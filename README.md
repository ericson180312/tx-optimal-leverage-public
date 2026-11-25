# Optimal Leverage on Taiwan Index Futures — Research → Live Trading

Six research rounds asking *how much leverage, and how to control it*, and a live Shioaji
trading system that runs the surviving strategy on real capital. Synced from the private
repository as of 2026-11-24.

**What makes this repo unusual is not the strategy** — a 200-day moving-average trend
filter is textbook. It is the audit trail: every round states the one question it asked,
freezes its conclusion, and hands it to the next; the live signal logic is bit-identical to
the backtest by test, and the places where the live rules depart from it are listed; and the things that were *not* proven are written down as plainly as the
things that were.

📄 **[RETROSPECTIVE.md](RETROSPECTIVE.md)** — the long-form story: original goal, how the
question kept being redefined, design principles, and every defect that only real trading
revealed. Written 2026-09-22 and left as is: it predates round 6, and the 5-day check and T+1
execution it describes were the rules at that time.

---

## What is *not* claimed

Putting this first, because it is the part most easily overstated.

- **No optimal leverage is declared.** Round 5 swept the target from 2.0 → 2.5 and found
  CAGR +0.8pp against MDD +6.6pp, with per-unit-leverage return decaying 12.1%
  (sub-linear). That establishes a poor exchange rate — it does **not** identify an optimum,
  and the data cannot. The live system's `L_HIGH = 2.0` is **a risk preference, not a
  statistical result.**
- **Parameters are not the edge.** Round 1+2 found the Sharpe surface essentially flat
  (full-grid range ≈ 0.07, within noise). The first-order factors were only *effective
  average leverage* and *whether a de-leveraging mechanism exists at all*.
- **MA=200 is a plateau, not a peak.** Round 4 was pre-registered (decision rules written
  before results). The raw median-Sharpe range across the window grid was 0.076 < the 0.08
  threshold; 200 is not an isolated spike. It was kept because nothing better was shown,
  not because it won.
- **Daily, same-day execution is a choice, not a finding.** Round 6 concluded that check
  frequency and execution timing sit on a plateau and that the original rule should stay.
  Since 2026-11-05 the live system nevertheless checks daily and trades the same day: the
  owner's decision. On real 13:20 prices it only breaks even, with about 70% more switches.
- **The lot hysteresis band is not a research result either.** It rests on an exploratory
  simulation (settlement prices standing in for 13:20, no costs).
- **No live performance claim.** Real-money trading began in September 2026. Equity change
  is dominated by market beta × leverage, not by strategy alpha. A sample this short
  supports no performance conclusion of any kind.
- **Single market, single instrument for live.** Research covered TAIEX / S&P 500 /
  Nikkei 225 / Hang Seng; live trades Taiwan micro index futures only.

---

## The research lineage

Each round answers exactly one question. Conclusions are frozen and never revisited —
later rounds reference earlier outputs read-only. Round 6 came after going live, to test two
timing settings that no earlier round had swept.

```
single-market grid → cross-market conditional exposure → parameter robustness
      (1+2)                      (3)                            (4)
                  → exposure trade-off curve → live → timing settings
                              (5)               d10         (6)
```

| Round | Question | Conclusion passed forward | Tag |
|---|---|---|---|
| **1+2** | For 2× leverage with threshold rebalancing, where should the bounds [X, Y] sit? | Surface is flat; differences are noise. First-order factors are only effective average leverage and presence of a ruin-prevention mechanism. Buy-and-hold 2× ruined on 12 of 93 rolling 5-year paths; every rebalancing variant survived. | — |
| **3** | Which exposure-control mechanism is most robust across markets? | Nine mechanisms × 4 markets × stationary bootstrap. **D_10 (200MA ± 1% hysteresis, L toggling 2.0 / 1.0) is the maximin winner** → becomes the live strategy; its outputs are frozen as the golden-test baseline. | `round3-complete` |
| **4** | Is MA=200 convention or overfitting? | Pre-registered study. Window is a plateau (raw Sharpe range 0.076 < 0.08); 200 is not an isolated peak → keep. Also surfaced a methodological trap: percentile quantisation over a 10-member cohort made the literal rule near-binary, so a non-quantised cross-check was added. | `round4-complete` |
| **5** | Is raising target leverage 2.0 → 2.5 worth it? | CAGR +0.8pp, MDD +6.6pp, marginal ΔCAGR/ΔMDD only 0.06–0.22 throughout. Sharpe is flat across the range (Sharpe is near-invariant to leverage — the penalty shows up in geometric quantities, not Sharpe). **Declines to name an optimum**; ships the frontier and the cost list instead. | `round5-complete` |
| **6** | The 5-day check and T+1 were never swept. Were they ever validated, and would de-leveraging the same day help? | Pre-registered. 123 variants × 16 market/period cells. Raw Sharpe range across check frequencies 0.009; the same-day *upper bound*, which includes look-ahead, gains 0.011. Both far below τ = 0.08 → **plateau, keep K=5 and T+1**. This means "every 5 days is as good as other frequencies", not "best". Does not feed the live system. | `round6-complete` |

**Live changes made outside the rounds** (choices, recorded as such):

| Date | Change | Basis | Rollback |
|---|---|---|---|
| 2026-11-05 | Check daily; decide and execute at 13:20 the same day | Owner's choice, against the round 6 default | `SIGNAL_CHECK_EVERY = 5`, `T_PLUS_1 = True` |
| 2026-11-24 | Within a regime, change lots only when raw lots leave N ± 0.6 | Exploratory simulation: non-switch trades 4.39 → 0.67 per year | `LOT_HYSTERESIS = 0` |

---

## Packages

| Package | Contents |
|---|---|
| `backtest_threshold/` | Rounds 1+2 — TX 2× threshold-rebalancing [X, Y] grid, single market |
| `backtest_multimarket/` | Round 3 — conditional-exposure strategies × 4 markets × stationary bootstrap. **`output/` is the frozen golden-test baseline; never regenerate it.** |
| `backtest_ma_robustness/` | Round 4 — MA-window robustness, pre-registered rules evaluated mechanically |
| `backtest_leverage_sweep/` | Round 5 — target-leverage sweep with a maintenance-margin / forced-liquidation model |
| `backtest_signal_timing/` | Round 6 — check frequency × execution lag × phase, pre-registered rules evaluated mechanically |
| `d10_trading/` | The live Shioaji trading system — staged gates G0–G3, golden test, fail-closed design |

Coupling is deliberately narrow: `d10_trading` touches only `backtest_multimarket/signals.py`
(bit-identical port) and the frozen output CSVs — it does **not** import the backtest engine.
Shared-engine changes are additive only, with new parameters defaulting to disabled.

---

## The live system

The strategy is small: every trading day at 13:20, compare the live price with the 200MA ± 1%
band; L = 2.0 above, 1.0 below; a switch executes **the same day**; contracts =
`round(L × equity ÷ contract value)`, with a 0.1-lot hysteresis band inside a regime. (Until
2026-11-05: check every 5 series-days, execute the next trading day. That path is kept, tested,
and a config change away.)

History is always rebuilt from official settlements; only today's row uses the 13:20 price, so a
wrong 13:20 call is overwritten the next day instead of persisting. With no live price, or when
TX and same-month TMF quotes disagree by more than 0.2%, nothing switches.

Everything else is about executing it safely:

```
every trading day, 13:20 Taipei (systemd oneshot + timer)

  kill switch → idempotency → reconcile (mismatch ⇒ stop, never auto-correct) → fuse
  → data rebuild + dual-source cross-check → today's provisional row → signal
  → flip quote check → position sizing (hysteresis band, hard cap) → rollover decision
  → limit orders → state, audit trail, notifications, heartbeat
```

| Principle | How |
|---|---|
| **Fail-closed** | Anything uncertain stops the day and alerts. No guessing, no auto-repair. |
| **Golden test (gate G0)** | The live signal module is bit-identical to the backtest's: with the 5-day check it matches the frozen round 3 output day by day, and with the daily check it matches the round 3 code. A mismatch fails the test suite. |
| **Staged gates** | G0 offline → G1 simulated → G2′ supervised live takeover → G3 unattended. Each gate requires an explicit human pass; live trading defaults to off. |
| **Append-only audit trail** | Every order, decision, alert, manual tool run and manual correction is logged, hooked at two unavoidable funnels (broker wrapper + batch orchestrator). |
| **Decision records** | Where the spec and reality disagreed, the judgement and its trade-off are written into the docs and the PR body — so the reasoning outlives the memory of it. |

### What only real trading revealed

Roughly a dozen defects, and **not one of those was a strategy-logic error** — all were in data
conventions, broker API behaviour, exchange rules, monitoring blind spots, and timing
boundaries. A representative few:

- A monitoring query that reported "all clear" because log levels never reached the
  severity it filtered on — the check itself was broken, silently.
- Combo (calendar-spread) fills reported under a different data shape than single-leg
  fills, so the executed price was recorded as zero for a real trade.
- The expiring contract stops trading 15 minutes earlier than a normal day, and the
  worst-case order flow could cross that boundary — where failure is *silent*, because
  cash settlement then closes the position and the next day's reconciliation looks fine.
- A quote that looks like a calendar spread on settlement day but is mechanically
  something else entirely, differing from the official figure by ~200 index points.
- A typhoon closure that is in no yearly calendar: the 13:20 batch ran as usual on a closed
  market. No trade was due, so nothing was lost.

These are the class of problem no backtest can surface. They are the reason the gates
exist, and the gates did catch them one layer at a time.

One later finding is of a different kind. Plain rounding of lots, believed to be a natural dead
band, trades back and forth when the raw number sits next to N + 0.5; at L = 1 it does so even on
a steadily falling price. That is a sizing-logic flaw. It had not occurred live; it was found by
analysis in November 2026.

---

## Reproducing

Requires Python 3.13 and the dependencies pinned in `pyproject.toml` / `uv.lock`.

```bash
uv sync --frozen

uv run python -m backtest_threshold.run           # rounds 1+2
uv run python -m backtest_multimarket.run3        # round 3
uv run python -m backtest_ma_robustness.run4      # round 4
uv run python -m backtest_leverage_sweep.run5     # round 5
uv run python -m backtest_signal_timing.run6      # round 6 (reuses cached data, never refreshes it)

uv run pytest d10_trading/tests -q                # live-system test suite
```

### Market data

Price data comes from [FinMind](https://finmind.github.io/). **The raw data cache is not
redistributed here** — set `FINMIND_TOKEN` in a `.env` file and the data layer will fetch
and cache it on first run.

One consequence: the **golden test needs that cache** to rebuild the continuous futures
series, so it *skips* without it (it does not fail):

```
216 passed, 3 skipped        # the 3 skips are the golden test
```

Every other test runs fully offline against synthetic fixtures. With a FinMind token and
one data fetch, the 3 skips become the strongest artifact in the repo: a bit-identical
comparison between the live signal module and the frozen backtest output.

### Not included

Live operations material is deliberately absent: deployment runbooks, server addresses,
credentials, and all account figures (equity, fills, positions). Numbers appearing in the
documentation as examples are rounded illustrations, not account values. Secrets were never
committed to this repository's history.

---

## Disclaimer

**This is not investment advice, and not a recommendation to trade.** Leveraged futures can
lose more than the capital committed. The research here documents one person's process on
one market over a limited sample; the live results cover a period far too short to mean
anything. If you run any of this with real money, you are choosing to accept a complete
loss of that money.

The code is provided as-is under the terms in [LICENSE](LICENSE), with no warranty of
correctness, fitness, or profitability.
