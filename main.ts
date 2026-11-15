// BTC race bot: tranche washout buyer on 1-minute cbBTC bars, long only.
//
// Idea (from sweeping the dev window, NOT from the previous race's bot):
// the dev day is a choppy downtrend (67216 -> 64524, buy-and-hold -$44)
// with three separate -3% washouts, each followed by a partial V-recovery.
// Single-shot dip buys lose: they catch the knife mid-waterfall and get
// stopped out before the turn. This bot instead scales INTO a washout in
// four equal tranches as price falls further from its 60-bar high, then
// holds up to 200 bars for the recovery, exiting on a take-profit-armed
// trailing stop (arm at +50 bps vs average cost, exit 150 bps below the
// peak close) or max hold. No hard stop: in this regime the washouts
// recover within hours, and stops just lock in the bottom.
//
//   - Entry, tranche 1: close >= 120 bps below the max high of the last
//     60 bars (excluding the current bar) and RSI(14) <= 45. Buys 1/4 of
//     cash. Anchor = that 60-bar high, frozen for the whole position.
//   - Tranches 2-4: each further 150 bps below the anchor (i.e. -270,
//     -420, -570 bps), each buying 1/4 of the original cash (remaining
//     cash split over the tranches left). No RSI gate on later tranches.
//   - Exit, whole position: take-profit-armed trailing stop, or 200 bars
//     after the first tranche, or after bar 1380 (no new entries then).
//   - 30-bar cooldown after an exit before a new tranche 1.
//   - Max slippage 100 bps per order; buys sized from cash on hand.
//
// Everything is deterministic: integer/bigint order amounts, float64
// RSI on + - * / alone (IEEE 754, same bars give the same orders run
// after run), no clock, no environment, no unseeded random.
//
// Sweep (local integer backtester replicating the harness fill math,
// full 24h run with mark-to-market equity at each 6h boundary):
//   n=4, t1=120, step=150, maxHold=200: +$10.35, 4 round trips,
//   11 filled orders, max drawdown 1.3%, 6h blocks
//   [+0.52, +0.32, +7.14, +2.38] (all positive).
// Neighbours agree (n=3: +$12.61; t1=100/n=4: +$8.50; step=120/n=4:
// +$7.49), so the edge is the washout-tranche shape, not one tuned
// number. Candidates compared and rejected on this window: EMA-cross
// pullback (best -$23.71, 2 trades), RSI/Bollinger mean reversion
// (-$120 or worse), RSI bounce (-$90 or worse), Donchian breakout
// (-$20.41), single-shot dip buys (-$33 to -$281). Buy-and-hold: -$44.

import { BPS, runBars, type BarsState, type BuyInput, type SellInput } from './vendor/makefun-bars.ts';

const SYM = 'CBBTC';
const MAX_SLIPPAGE_BPS = 100;

// Strategy parameters (swept on the dev window; see header).
const N_TRANCHES = 4;
const T1_BPS = 120;
const TSTEP_BPS = 150;
const DIP_N = 60;
const RSI_BUY = 45;
const RSI_LEN = 14;
const TP_BPS = 50;
const TRAIL_BPS = 150;
const MAX_HOLD = 200;
const NO_ENTRY_AFTER = 1380;
const COOLDOWN_BARS = 30;

const closes: number[] = [];
const highs: number[] = [];
const lows: number[] = [];

// Wilder RSI(14) state.
let rsiAvgGain: number | null = null;
let rsiAvgLoss: number | null = null;
let rsiPrevClose: number | null = null;
let rsi: number | null = null;
const rsiSeedG: number[] = [];
const rsiSeedL: number[] = [];

// Open-position state (mirrors the backtester).
let tranches = 0;
let anchor = 0;
let entrySeq = -1;
let peakClose = 0;
let lastExitSeq = -1000000;
let armed = false;

function pushBar(c: number, h: number, l: number): void {
  closes.push(c);
  highs.push(h);
  lows.push(l);
  if (rsiPrevClose !== null) {
    const chg = c - rsiPrevClose;
    const gain = chg > 0 ? chg : 0;
    const loss = chg < 0 ? -chg : 0;
    if (rsiAvgGain === null || rsiAvgLoss === null) {
      rsiSeedG.push(gain);
      rsiSeedL.push(loss);
      if (rsiSeedG.length === RSI_LEN) {
        rsiAvgGain = rsiSeedG.reduce((a, x) => a + x, 0) / RSI_LEN;
        rsiAvgLoss = rsiSeedL.reduce((a, x) => a + x, 0) / RSI_LEN;
        rsi = rsiAvgLoss === 0 ? 100 : 100 - 100 / (1 + rsiAvgGain / rsiAvgLoss);
      }
    } else {
      rsiAvgGain = (rsiAvgGain * (RSI_LEN - 1) + gain) / RSI_LEN;
      rsiAvgLoss = (rsiAvgLoss * (RSI_LEN - 1) + loss) / RSI_LEN;
      rsi = rsiAvgLoss === 0 ? 100 : 100 - 100 / (1 + rsiAvgGain / rsiAvgLoss);
    }
  }
  rsiPrevClose = c;
}

/** Max high of the DIP_N bars BEFORE the current one. */
function maxHighBefore(n: number): number | null {
  if (highs.length <= n) return null;
  let m = -Infinity;
  for (let i = highs.length - n - 1; i < highs.length - 1; i++) if (highs[i] > m) m = highs[i];
  return m;
}

function resetPosition(): void {
  tranches = 0;
  anchor = 0;
  entrySeq = -1;
  peakClose = 0;
  armed = false;
}

await runBars({
  onBatch(state, batch) {
    for (const bar of batch.bars) {
      if (bar.symbol !== SYM) continue;
      pushBar(Number(bar.c), Number(bar.h), Number(bar.l));
    }
    if (state.warmingUp) return [];
    if (closes.length === 0) return [];
    const seq = batch.seq;
    const close = closes[closes.length - 1];
    const held = state.positions.get(SYM);
    const { caps, fees } = state.hello;

    // If a tranche buy was rejected last bar, the position never opened:
    // drop the half-built tranche state so tranche 1 can trigger again.
    if (held === undefined && tranches > 0) {
      const rejectedBuy = batch.fills.some((f) => f.side === 'buy' && f.status === 'rejected');
      if (rejectedBuy) resetPosition();
    }

    // Past the entry horizon: flatten if holding, otherwise stay out.
    if (seq > NO_ENTRY_AFTER) {
      if (held !== undefined) {
        lastExitSeq = seq;
        resetPosition();
        return [{ side: 'sell', symbol: SYM, qty: held.qty, max_slippage_bps: MAX_SLIPPAGE_BPS }];
      }
      return [];
    }

    if (held !== undefined) {
      if (close > peakClose) peakClose = close;
      const value = state.value(SYM, held.qty);
      // Arm the trailing stop once the position is TP_BPS above its cost.
      if (value * 10000n >= held.cost * BigInt(10000 + TP_BPS)) armed = true;
      let sell = false;
      if (armed && close <= peakClose * (1 - TRAIL_BPS / 10000)) sell = true;
      if (seq - entrySeq >= MAX_HOLD) sell = true;
      if (sell) {
        lastExitSeq = seq;
        resetPosition();
        return [{ side: 'sell', symbol: SYM, qty: held.qty, max_slippage_bps: MAX_SLIPPAGE_BPS }];
      }
      // Add the next tranche if price fell another step below the anchor.
      if (tranches < N_TRANCHES && anchor > 0) {
        const level = -T1_BPS - tranches * TSTEP_BPS;
        if ((close - anchor) * 10000 / anchor <= level) {
          let amt = state.cash / BigInt(N_TRANCHES - tranches);
          // Defensive: never let one buy push the position past the
          // weight cap (cash-funded buys cannot in practice, but the cap
          // is a protocol rejection if it ever binds).
          const room = state.weightRoom(SYM);
          if (amt > room) amt = room;
          if (amt >= BigInt(fees.min_notional_quote_units)) {
            tranches += 1;
            return [{ side: 'buy', symbol: SYM, notional_quote_units: amt, max_slippage_bps: MAX_SLIPPAGE_BPS }];
          }
        }
      }
      return [];
    }

    // Flat: tranche 1 entry on a washout with RSI not overbought.
    if (seq - lastExitSeq < COOLDOWN_BARS) return [];
    const mh = maxHighBefore(DIP_N);
    if (mh === null) return [];
    if ((close - mh) * 10000 / mh <= -T1_BPS && (rsi === null || rsi <= RSI_BUY)) {
      let amt = state.cash / BigInt(N_TRANCHES);
      const room = state.weightRoom(SYM);
      if (amt > room) amt = room;
      if (amt === 0n || amt < BigInt(fees.min_notional_quote_units)) return [];
      anchor = mh;
      tranches = 1;
      entrySeq = seq;
      peakClose = close;
      return [{ side: 'buy', symbol: SYM, notional_quote_units: amt, max_slippage_bps: MAX_SLIPPAGE_BPS }];
    }
    return [];
  },
});
