# 🗂️ Adjust Asset Prices 🗂️

Adjust raw historical prices for gaps, splits (and optionally dividends), with an opt-in options pass.

Follow-up to the [`fetch-asset-prices`](https://github.com/sukiboo/fetch-asset-prices) repo,
which produces the raw input files this tool reads from `./data/files/<asset_type>/`.

## Scope

Built for **multi-year price histories** of assets where long histories are hard to source
elsewhere. Accuracy is calibrated for long horizons; short windows can show small
divergences from yfinance's daily close, because yfinance's snapshot doesn't precisely
line up with our 1-min UTC bars (an offset of a couple of hours of FX market state for
forex, fractions of a minute for crypto). If you need a week of data, other providers serve
it more easily; the value here is in multi-year histories you can't get cleanly elsewhere.

## Setup

```bash
pyenv virtualenv 3.12 adjust-asset-prices
pyenv activate adjust-asset-prices
pip install -r requirements.txt
```

The repo's `.python-version` pins the venv, so `cd`-ing in auto-activates it if you have
`pyenv virtualenv-init` in your shell.

The pipeline reads raw daily price files from `./data/files/<asset_type>/`. Download the
pre-built files (all asset types, through 2026) from the
[data archive](https://www.dropbox.com/scl/fo/xd5a5s5cwa0imf6gvplzv/AL1ffzRw3_AEfeEwRoKLQms?rlkey=ah6c8ps5zvco29npoeoro831k&dl=0),
or generate them with [`fetch-asset-prices`](https://github.com/sukiboo/fetch-asset-prices).
Running `run.py` without local data prints this same pointer.

## Usage

```bash
python run.py <TICKER> [OPTIONS]
```

End-to-end pipeline for one ticker: load raw bars → backfill missing minutes → adjust for
splits (and optionally cash dividends; stocks only, using yfinance metadata) → run sanity
checks → save to `./data/prices/<TICKER>_<start>_<end>.<format>` (the date range is taken
from the saved series' own index) → reload and verify the round-trip.

Tickers are unprefixed and the asset type is auto-detected from your local data: `BTC-USD`
(crypto), `EUR-USD` (forex), `AAPL` (stock). For `--options`, pass the **underlying** (`AAPL`),
not an individual contract.

Options:

| flag            | default          | description                                          |
|-----------------|------------------|------------------------------------------------------|
| `--data-dir`    | `./data/files`   | raw input directory                                  |
| `--save-dir`    | `./data/prices`  | adjusted output directory                            |
| `--format`      | `parquet`        | output format: `parquet` or `csv`                    |
| `--date-start`  | (earliest)       | inclusive start date, `YYYY-MM-DD`                   |
| `--date-end`    | (latest)         | inclusive end date, `YYYY-MM-DD`                     |
| `--dividends`   | off              | also back-adjust stocks for cash dividends           |
| `--options`     | off              | also fetch the underlying's option contracts         |
| `--plot`        | off              | display the price-comparison plot against yfinance   |
| `--force`       | off              | save even if the price checks fail                  |

`--dividends` and `--options` are mutually exclusive: the default output is the actual
split-adjusted (not dividend-adjusted) price, so options align with their underlying.

When the sanity checks fail the data is **not** saved by default. In an interactive terminal
you're prompted `[y/N]` to save it anyway; `--force` skips that prompt (and is the only way to
override in a non-interactive run). Overridden output is saved as-is and unvalidated — useful
for long crypto histories where the early, low-liquidity years diverge from yfinance's own
sparse early data far more than the threshold allows (see [Scope](#scope)).

Examples:

```bash
# crypto, full available history, default parquet output
python run.py BTC-USD

# split-adjusted stock over an explicit window, written as CSV
python run.py AAPL --date-start 2020-01-01 --date-end 2024-12-31 --format csv

# dividend-adjusted (total-return) series from a start date, with the comparison plot shown
python run.py SPY --dividends --date-start 2015-01-01 --plot

# forex pair bounded both ends, reading/writing custom directories
python run.py EUR-USD --date-start 2022-01-01 --date-end 2023-12-31 \
  --data-dir /mnt/polygon/files --save-dir /mnt/out/prices

# options companion pass on an underlying for a one-month window
python run.py NVDA --options --date-start 2024-06-01 --date-end 2024-06-30
```

## Output

Files are written to `--save-dir` (default `./data/prices/`), then reloaded and compared against
memory to verify the round-trip. The `<start>_<end>` range (`YYYYMMDD`) is taken from the saved
frame's own index.

- **Stocks / crypto / forex** — one file `<TICKER>_<start>_<end>.<format>`
  (e.g. `BTC-USD_20240101_20241231.parquet`): a 1-minute `timestamp_utc` index (UTC) and a single
  price column named after the ticker.
- **Options** — two files under an `options/` subfolder,
  `options/<UNDERLYING>_<start>_<end>_{calls,puts}.<format>`: a `(timestamp_utc, ticker)`
  multi-index and a single `close` column (one row per contract per minute).

Parquet preserves the UTC timezone natively; CSV is re-localized to UTC on load. Read a file back
with pandas:

```python
import pandas as pd

# single-series
prices = pd.read_parquet("data/prices/BTC-USD_20240101_20241231.parquet")

# options (one side)
calls = pd.read_parquet("data/prices/options/NVDA_20240601_20240630_calls.parquet")
```

## Caveats

Two things hold across every asset type:

- **Backfilled minutes are synthetic and unmarked in the saved files.** Gaps in liquid assets
  (*stocks, crypto, forex*) are filled by log-space linear interpolation between the surrounding
  real prints; *options* instead hold the last traded price flat (no price discovery
  happens between option trades, so an interpolated path would be fabricated). For thin-volume
  assets with long gaps, synthetic values can drift meaningfully from what really traded.
- **Corporate-event metadata comes from yfinance** (`.splits`, `.dividends`) and is applied
  across the entire history — matching yfinance's `Adj Close` convention — so a saved file is a
  point-in-time snapshot; re-run if new events are announced later. All timestamps are UTC.

### Crypto

- **Hours:** continuous 24/7 — a 1-minute grid over 00:00–23:59 UTC every day.
- **No corporate adjustments** — passed through as-is (no splits or dividends).
- **Hard forks are not adjusted for.** When a chain splits (BCH from BTC, ETC from ETH, …),
  pre-fork prices of the surviving ticker are inflated by the value of the spun-off coin —
  analogous to a stock split, but uncorrected here. Irrelevant for most majors in typical
  ranges; worth knowing if you backtest across a known fork date.

### Forex

- **Hours:** continuous 1-minute grid over 00:00–23:59 UTC every calendar day; weekend gaps,
  when the FX market is closed, are filled by interpolation like any other gap.
- **No corporate adjustments** — passed through as-is (no splits or dividends).
- **Daily-close alignment is London midnight** (yfinance's forex convention). On a single
  high-volatility day the few-hours offset from our UTC bars can look like a divergence in the
  sanity check; it averages out over long windows (see [Scope](#scope)).

### Stocks

- **Hours:** NYSE extended hours, `[04:00, 19:59]` ET on session days — matching Polygon's
  emission exactly, half-days handled via the exchange calendar. No overnight, weekend, or
  holiday rows. When the raw data doesn't reach a session's 04:00 open or 19:59 close, the
  missing edge minutes are flat-filled from the nearest real print.
- **Splits** are always applied — pre-split prices ÷ ratio.
- **Dividends** are opt-in via `--dividends` (off by default, so the default output is the
  actual split-adjusted traded price). With the flag, the series is back-adjusted to a
  total-return basis matching yfinance's `Adj Close`.
- **Ticker renames** are stitched automatically. Request the *current* symbol (the one valid
  at `--date-end`) and a long gap in its raw history — interior (e.g. QQQ traded as QQQQ
  2004–2011) or leading vs `--date-start` (e.g. META's pre-2022 history under FB) — triggers a
  search of the local files for the predecessor symbol, which is spliced in so the output is
  one continuous series under the requested ticker. The match is local-data-only (the symbol
  that stops trading right where the requested series resumes, traded with comparable liquidity,
  and is nearest in price); the compare-to-yfinance check is the independent verification — it
  rejects the result if yfinance can't corroborate a stitched span, so a wrong match can't slip
  through. No external rename API is used or needed. A genuinely new ticker with no predecessor is
  left alone.
- **Reused tickers** are handled too. When a symbol was held by an unrelated company before the
  current one (e.g. `META` was Meta Materials at ~$12 until early 2022, then Facebook took it on
  its FB→META rename), the loader picks up that foreign history along with the real series. The
  stitcher detects it: a genuine rename connects on *both* edges of the gap, but a reused ticker
  connects only on the resume side — so the discontinuous pre-gap segment is dropped as foreign.
  `META` thus loads as clean Facebook history (and `--date-start` before the rename pulls FB's
  earlier history back as a normal stitch).

### Options

- **Hours:** NYSE regular hours, `[09:30, 15:59]` ET on session days. Index/ETF options with an
  extended 16:15 ET close (SPX, SPXW, SPY, QQQ, IWM, …) have their 16:00–16:14 ET bars dropped
  on reindex — `[09:30, 15:59]` is correct for vanilla equity contracts, and extending it
  universally would fabricate synthetic rows for all of them.
- **Opt-in companion pass** (`--options`), not a standalone mode: you pass an underlying; the
  pipeline runs its stocks pass first, then loads every OSI contract on it (including OCC
  numeric-suffix roots like `AAPL7`), OCC-adjusts for splits (premium ÷ ratio *and* OSI symbol
  rewrite), backfills by holding the last trade flat to expiry, validates, and saves.
- **Never dividend-adjusted**, and `--options`/`--dividends` are mutually exclusive: the quality
  gate needs the underlying as it actually traded, and dividends are priced into premiums rather
  than back-adjusted out.
- **Quality gate is structural**, not a yfinance comparison (yfinance has no historical
  per-contract data): a no-arb self-consistency check (price > 0, calls ≤ underlying, puts ≤
  strike, intrinsic floors, no bars past expiry), scored on real (traded), near-the-money bars
  only.
- **Spinoff-style corporate actions** (where the OCC changes the deliverable, not the strike)
  are skipped.
- **Heavy on time and memory** — a run loads *every* contract on the underlying for the window
  (millions of contract-bars for a popular name over even a single month — e.g. ~12M for NVDA in
  June 2024), so expect tens of seconds to a few minutes and substantial RAM. Narrow the date
  window if it's too much.

## Testing

End-to-end integration tests in `tests/integration/` run the full pipeline against real
Polygon flat files and live yfinance data — no mocks. The full sweep takes about 30 mins,
run with:

```bash
pytest -m integration -v -s
```

`-s` disables stdout capture so each test's scenario header, bar counts, price range, and
boundary spot-checks print to the terminal as it runs. Drop `-s` if you want a quieter pass.
Tests skip per-asset when `data/files/<asset>/` is empty, so wire up the data the same way
you would for `run.py`.

### How outputs are validated

The two regimes differ in one crucial way — whether an **independent ground truth** exists to
check against:

- **Stocks, crypto, forex are checked against an external reference.** Each output's daily close
  is compared to yfinance's daily close over the window (`compare_to_yf`), gated on percentile
  bands of the absolute relative difference (median and p99, with per-asset thresholds). Event
  tests add a boundary **spot-check** — the close-to-close ratio across a split or dividend
  ex-date must land in an expected band (≈1 after a correct split back-adjustment; just under 1
  across a dividend). A regression therefore shows up as drift from a source we don't control.
- **Options cannot be verified this way — there is no historical ground truth to compare
  against.** yfinance, the reference everywhere else in this project, exposes only
  *current-snapshot* option chains and returns nothing for historical OSI tickers (expired or
  active), so there is nothing authoritative to compare a 2014 contract's premium to. Instead the
  gate (`check_options`) enforces **structural self-consistency** — no-arbitrage bounds the prices
  must satisfy regardless of source: price > 0, no bars past expiry, calls ≤ underlying, puts ≤
  strike, and intrinsic floors, scored on real (traded), near-the-money bars. This reliably
  catches a *broken* split adjustment (it corrupts real premiums across all moneyness and blows
  past the bounds), and forward splits add a continuity check — but it **cannot confirm that an
  un-broken premium equals what actually traded**, only that it is internally consistent. Read
  the option outputs as "structurally sound," not "independently verified."

### Coverage

- **Stocks** — vs yfinance Close (or Adj Close under `--dividends`): AAPL 7:1 split (split-only
  path), AAPL 2023 split-only-vs-dividend-adjusted both-ways (convention switch), NVDA 4:1 + 10:1
  splits with dividends, GE 1:8 reverse split, SPY 2020–2023 dividends multi-year, QYLD monthly
  ROC distributions, MSFT 2004 $3.08 special dividend (~10% drop), BBBY pre-bankruptcy (a
  known-divergence canary: yfinance serves a heavily re-adjusted ~$17–25 history vs the real
  ~$0.2–7 traded prices, so the test asserts the compare *fails* — and flags if yfinance ever
  changes its delisting handling).
- **Crypto** — vs yfinance daily Close, with looser thresholds since "daily close" is ill-defined
  across sources: BTC-USD Luna/Terra crash (threshold stress), BTC-USD 2020–2022 multi-year,
  ETH-USD 2021–2022 (Merge + FTX), SOL-USD 2022–2023 (FTX collapse + recovery).
- **Forex** — vs yfinance daily close aligned at London midnight (its forex convention): EUR-USD
  2025 (spring DST transition + tariff-day shock), USD-JPY 2022 (BoJ intervention, high-nominal
  pair), EUR-GBP 2023 (cross-pair, no USD leg), EUR-USD 2022–2024 multi-year.
- **Options** — structural gate only (no external source; see above): NVDA 10:1 2024
  (clean-strike unification), AAPL 7:1 2014 (non-clean / suffixed-root unification), TSLA 3:1 2022
  (clean integer ratio), AAPL 2023 no-split (clean-window gate), NVDA 4:1 2021 and TSLA 5:1 2020
  (cumulative forward multi-split, ÷40 / ÷15), GRPN 1:20 2020 and VXX two-1:4 (reverse and
  cumulative-reverse, ×20 / ×16). Forward splits — and GRPN's reverse split — pass the no-arb gate
  against the split-only underlying, and forward splits additionally check split-successor
  **continuity** on a *deep-ITM* probe (real-trade close before vs after the split, within ~10%;
  deep-ITM is low-elasticity, so the ratio reflects the adjustment rather than gamma, and
  real-trade endpoints avoid the ffill artifact that makes fixed-time bars vacuously ~1). Reverse
  splits get **no continuity probe** (thin/volatile names have no deep-ITM contract that traded
  both sides — only ATM/OTM, which swing wildly on gamma); they verify the cumulative
  back-adjustment **factor** instead. VXX additionally **skips the gate**: as a steep-contango vol
  ETP its forward sits far below spot, so ITM calls legitimately trade below the spot-based
  intrinsic floor — a genuine carry effect the gate can't model, independent of split correctness.
