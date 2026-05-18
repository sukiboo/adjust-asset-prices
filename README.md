# adjusted-asset-prices

Adjust raw historical prices for splits and dividends.

Follow-up to the [`historical-asset-prices`](https://github.com/sukiboo/historical-asset-prices) repo,
which produces the raw input files this tool reads from `./data/files/<asset_type>/`.

## Setup

```bash
pyenv virtualenv 3.12 adjusted-asset-prices
pyenv activate adjusted-asset-prices
pip install -r requirements.txt
```

The repo's `.python-version` pins the venv, so `cd`-ing in auto-activates it if you have
`pyenv virtualenv-init` in your shell.

## Usage

```bash
python main.py <TICKER> [OPTIONS]
```

End-to-end pipeline for one ticker: load raw bars → backfill missing minutes → adjust
(splits/dividends — currently stubbed) → run sanity checks → save to
`./data/prices/<TICKER>.<format>` → reload and verify the round-trip.

Options:

| flag            | default          | description                                  |
|-----------------|------------------|----------------------------------------------|
| `--format`      | `parquet`        | output format: `parquet` or `csv`            |
| `--date-start`  | (earliest)       | inclusive start date, `YYYY-MM-DD`           |
| `--date-end`    | (latest)         | inclusive end date, `YYYY-MM-DD`             |
| `--data-dir`    | `./data/files`   | raw input directory                          |
| `--save-dir`    | `./data/prices`  | adjusted output directory                    |
| `--debug`       | off              | print extra info during load                 |

Examples:

```bash
python main.py BTC-USD --debug
python main.py SPY --format csv --date-start 2024-01-01
```
