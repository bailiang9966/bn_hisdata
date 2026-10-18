"""
Download Binance public market data from https://data.binance.vision (S3).

Supports
  markets   : spot, futures/um  (--market both downloads both)
  symbols   : BTCUSDT, ETHUSDT  (--symbol all downloads both; comma-separated list supported)
  categories: klines (needs --interval), metrics (um only), aggTrades.
              (--category all downloads klines + metrics; comma-separated list supported)
  granularity: daily  -> one zip per day   (good for testing / a few days)
               monthly -> one zip per month (much faster for bulk backfills)
  workers   : parallel download & processing via --workers / -j (default: 4)

Disk layout (under <module>/data by default):
  data/{market}/{granularity}/{category}/{symbol}/{interval?}/{symbol}-{type}-{period}.{parquet|csv}
  The source .zip is downloaded to the same folder, processed into this cleaned
  parquet (default) or CSV (--format csv), and then deleted (the zip is only a
  transient intermediate). Resume is keyed on the cleaned output file, so
  re-running is a safe no-op once it exists.
"""
import argparse
import concurrent.futures
import hashlib
import threading
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE = "https://data.binance.vision/data"

VALID_INTERVALS = {
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
    "6h", "8h", "12h", "1d", "3d", "1w", "1mo",
}
MARKETS = {"spot": "spot", "um": "futures/um"}
VALID_CATEGORIES = {"klines", "aggTrades", "metrics"}

ALL_CATS = ["klines", "metrics"]

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume",
              "taker_buy_quote_volume", "ignore"]

KLINE_KEEP = ["open_time", "open", "high", "low", "close", "close_time",
              "count", "buy_vol", "sell_vol"]

METRICS_COLS = ["create_time", "symbol", "sum_open_interest",
                "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                "sum_taker_long_short_vol_ratio"]

AGGTRADES_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id",
                  "last_trade_id", "transact_time", "is_buyer_maker",
                  "is_best_match"]

CANONICAL = {"klines": KLINE_COLS, "metrics": METRICS_COLS,
             "aggTrades": AGGTRADES_COLS}

EXPECTED_FIRST = {"klines": "open_time", "metrics": "create_time",
                  "aggTrades": "agg_trade_id"}


def create_session(max_workers: int = 4) -> requests.Session:
    """Create a requests Session with HTTP connection pooling and automatic retries."""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    pool_size = max(10, max_workers * 2)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (bn_hisdata/binance_vision)"})
    return s


def period_strings(granularity: str, start: date, end: date):
    if granularity == "daily":
        d = start
        while d <= end:
            yield d.strftime("%Y-%m-%d")
            d += timedelta(days=1)
    else:  # monthly
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            yield f"{y:04d}-{m:02d}"
            m += 1
            if m > 12:
                m, y = 1, y + 1


def url_for(market_path: str, granularity: str, category: str,
            symbol: str, interval: str, period: str) -> str:
    if category == "klines":
        return (f"{BASE}/{market_path}/{granularity}/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{period}.zip")
    return (f"{BASE}/{market_path}/{granularity}/{category}/{symbol}/"
            f"{symbol}-{category}-{period}.zip")


def dest_path(data_root: Path, market_key: str, granularity: str, category: str,
              symbol: str, interval: str, period: str) -> Path:
    if category == "klines":
        return (data_root / market_key / granularity / category / symbol /
                interval / f"{symbol}-{interval}-{period}.zip")
    return (data_root / market_key / granularity / category / symbol /
            f"{symbol}-{category}-{period}.zip")


def fetch_expected_sha256(session: requests.Session, url_checksum: str, log) -> str | None:
    try:
        r = session.get(url_checksum, timeout=30)
        if r.status_code != 200:
            log(f"no checksum file (HTTP {r.status_code}), skipped")
            return None
        text = r.text.strip()
        if not text:
            log("empty checksum file, skipped")
            return None
        return text.split()[0].lower()
    except Exception as e:
        log(f"checksum check skipped: {e}")
        return None


def download_one(session: requests.Session, url: str, dest: Path, verify: bool, log) -> str:
    if dest.exists():
        log("skip (exists)")
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".part")
    last = None
    for attempt in range(1, 4):
        try:
            h = hashlib.sha256()
            size = 0
            with session.get(url, timeout=(30, 120), stream=True) as r:
                if r.status_code == 200:
                    with tmp.open("wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                fh.write(chunk)
                                h.update(chunk)
                                size += len(chunk)
                    expected = (fetch_expected_sha256(session, url + ".CHECKSUM", log)
                                if verify else None)
                    if expected is not None and expected != h.hexdigest():
                        log(f"CHECKSUM MISMATCH: expected {expected}, "
                            f"got {h.hexdigest()}")
                        last = "checksum mismatch"
                    else:
                        if expected is not None:
                            log("checksum OK")
                        tmp.replace(dest)
                        log(f"saved {size / 1e6:.1f} MB")
                        return "ok"
                elif r.status_code == 404:
                    log(f"MISSING (HTTP 404): {url}")
                    return "missing"
                else:
                    last = r.status_code
        except Exception as e:
            last = e
        tmp.unlink(missing_ok=True)
        if attempt < 3:
            time.sleep(2 ** attempt)
            log(f"attempt {attempt} failed ({last}); retrying")
    log(f"FAILED after retries: {url}")
    return "failed"


def get_schema_overrides(category: str, fdtype: pl.DataType) -> dict[str, pl.DataType]:
    """Return explicit schema overrides for Polars scan_csv to optimize CSV parsing."""
    if category == "klines":
        return {
            "open_time": pl.Int64,
            "open": fdtype,
            "high": fdtype,
            "low": fdtype,
            "close": fdtype,
            "volume": fdtype,
            "close_time": pl.Int64,
            "quote_volume": fdtype,
            "count": pl.Int64,
            "taker_buy_volume": fdtype,
            "taker_buy_quote_volume": fdtype,
            "ignore": pl.String,
        }
    elif category == "aggTrades":
        return {
            "agg_trade_id": pl.Int64,
            "price": fdtype,
            "quantity": fdtype,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
            "transact_time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
            "is_best_match": pl.Boolean,
        }
    elif category == "metrics":
        return {
            "sum_open_interest": fdtype,
            "sum_open_interest_value": fdtype,
            "count_toptrader_long_short_ratio": fdtype,
            "sum_toptrader_long_short_ratio": fdtype,
            "count_long_short_ratio": fdtype,
            "sum_taker_long_short_vol_ratio": fdtype,
        }
    return {}


def process(zip_path: Path, category: str, fmt: str, fdtype: pl.DataType, log) -> None:
    out = zip_path.with_suffix(f".{fmt}")
    if out.exists():
        return
    tmp_out = out.parent / (out.name + ".part")
    tmp_csv = out.parent / (zip_path.stem + ".src.part")
    try:
        with zipfile.ZipFile(zip_path) as z:
            csv_in = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_in:
                log("no csv inside zip")
                return
            with z.open(csv_in[0]) as src, tmp_csv.open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
        with tmp_csv.open("rb") as fh:
            first = fh.readline().decode("utf-8", errors="replace")
            if not first.strip():
                log("empty csv inside zip")
                return
            exp = EXPECTED_FIRST.get(category)
            is_header = bool(exp) and first.split(",")[0].strip() == exp

        schema_ovr = get_schema_overrides(category, fdtype)

        if is_header:
            lf = pl.scan_csv(tmp_csv, has_header=True, schema_overrides=schema_ovr)
        else:
            lf = pl.scan_csv(tmp_csv, has_header=False,
                             new_columns=CANONICAL[category],
                             schema_overrides=schema_ovr)

        if category == "klines":
            lf = lf.with_columns([
                pl.col("taker_buy_volume").alias("buy_vol"),
                (pl.col("volume") - pl.col("taker_buy_volume")).alias("sell_vol"),
                pl.when(pl.col("close_time") > 1e14)
                  .then(pl.col("close_time") // 1000 + 1)
                  .otherwise(pl.col("close_time") + 1)
                  .alias("close_time"),
                pl.when(pl.col("open_time") > 1e14)
                  .then(pl.col("open_time") // 1000)
                  .otherwise(pl.col("open_time"))
                  .alias("open_time"),
            ]).select(KLINE_KEEP)
        elif category == "aggTrades":
            pass
        elif category == "metrics":
            if lf.collect_schema()["create_time"].is_numeric():
                create_time = pl.col("create_time").cast(pl.Int64)
            else:
                create_time = (pl.col("create_time")
                               .str.to_datetime(format="%Y-%m-%d %H:%M:%S")
                               .dt.epoch("ms").cast(pl.Int64))
            lf = lf.with_columns(create_time)

        if fmt == "parquet":
            lf.sink_parquet(str(tmp_out), compression="zstd")
        else:
            lf.sink_csv(str(tmp_out))
        tmp_out.replace(out)
        log(f"cleaned -> {out.name}")
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


def parse_list_arg(val: str) -> list[str]:
    return [item.strip() for item in val.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["spot", "um", "both"], default="both")
    ap.add_argument("--symbol", default="BTCUSDT",
                    help="BTCUSDT / ETHUSDT / all / comma-separated list (e.g. BTCUSDT,ETHUSDT,SOLUSDT)")
    ap.add_argument("--category", default="all",
                    help="klines / aggTrades / metrics / all / comma-separated list (e.g. klines,metrics)")
    ap.add_argument("--interval", default="1m", help="klines interval")
    ap.add_argument("--granularity", choices=["daily", "monthly"], default="daily")
    ap.add_argument("--start-date", dest="start_date", required=True)
    ap.add_argument("--end-date", dest="end_date", required=True)
    ap.add_argument("--dest-root", default=str(DATA))
    ap.add_argument("--no-checksum", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--workers", "-j", type=int, default=4,
                    help="并发下载和处理的工作线程数（默认: 4）")
    args = ap.parse_args()

    if args.interval not in VALID_INTERVALS:
        ap.error(f"--interval must be one of {sorted(VALID_INTERVALS)}")

    data_root = Path(args.dest_root)
    markets = ["spot", "um"] if args.market == "both" else [args.market]

    if args.symbol.lower() == "all":
        symbols = ["BTCUSDT", "ETHUSDT"]
    else:
        symbols = [s.upper() for s in parse_list_arg(args.symbol)]

    if args.category.lower() == "all":
        cats = ALL_CATS
    else:
        cats = parse_list_arg(args.category)
        for cat in cats:
            if cat not in VALID_CATEGORIES:
                ap.error(f"Invalid category '{cat}'. Must be one of {sorted(VALID_CATEGORIES)}")

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        ap.error("--start-date must be <= --end-date")

    fdtype = pl.Float32 if args.dtype == "float32" else pl.Float64

    print_lock = threading.Lock()
    totals_lock = threading.Lock()

    def log(tag: str, msg: str) -> None:
        with print_lock:
            print(f"[{tag}] {msg}", flush=True)

    totals = {"ok": 0, "skip": 0, "missing": 0, "failed": 0}
    skipped = []

    def record_total(status: str, count: int = 1) -> None:
        with totals_lock:
            totals[status] = totals.get(status, 0) + count

    def handle_task(task: dict, session: requests.Session) -> None:
        tag = task["tag"]
        url = task["url"]
        dp = task["dp"]
        cat = task["cat"]
        out = task["out"]

        if out.exists():
            log(tag, f"skip (cleaned .{args.format} exists)")
            record_total("skip")
            return

        status = download_one(session, url, dp, not args.no_checksum, lambda m: log(tag, m))
        if status not in ("ok", "skip"):
            record_total(status)
            return
        if args.no_extract:
            record_total(status)
            return
        try:
            process(dp, cat, args.format, fdtype, lambda m: log(tag, m))
        except Exception as e:
            log(tag, f"process FAILED: {e}; forcing re-download and retrying once")
            if dp.exists():
                dp.unlink()
            if download_one(session, url, dp, not args.no_checksum, lambda m: log(tag, m)) != "ok":
                record_total("failed")
                return
            try:
                process(dp, cat, args.format, fdtype, lambda m: log(tag, m))
            except Exception as e2:
                log(tag, f"process FAILED again: {e2}; giving up on this period")
                if dp.exists():
                    dp.unlink()
                record_total("failed")
                return

        if dp.exists():
            dp.unlink()
            log(tag, "removed source zip")
        record_total("ok")

    tasks = []
    for market_key in markets:
        market_path = MARKETS[market_key]
        for symbol in symbols:
            for cat in cats:
                if cat == "metrics" and market_key == "spot":
                    log(f"{market_key}/{cat}/{symbol}", "skip: spot has no metrics")
                    skipped.append(f"{market_key}/metrics/{symbol}: spot 无 metrics 数据")
                    continue
                if cat == "metrics" and args.granularity == "monthly":
                    log(f"{market_key}/{cat}/{symbol}",
                        "skip: metrics is daily-only")
                    skipped.append(f"{market_key}/metrics/{symbol}: "
                                   "metrics 只有日度文件，monthly 自动跳过")
                    continue
                for period in period_strings(args.granularity, start, end):
                    tag = f"{market_key}/{cat}/{symbol}/{period}"
                    url = url_for(market_path, args.granularity, cat, symbol,
                                  args.interval, period)
                    dp = dest_path(data_root, market_key, args.granularity, cat,
                                   symbol, args.interval, period)
                    out = dp.with_suffix(f".{args.format}")
                    if out.exists():
                        log(tag, f"skip (cleaned .{args.format} exists)")
                        totals["skip"] += 1
                        continue
                    tasks.append({
                        "tag": tag, "url": url, "dp": dp, "cat": cat, "out": out
                    })

    if tasks:
        session = create_session(max_workers=args.workers)
        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(handle_task, task, session) for task in tasks]
                concurrent.futures.wait(futures)
        else:
            for task in tasks:
                handle_task(task, session)

    print("\n==== summary ====")
    print(f"ok={totals['ok']} skip={totals['skip']} "
          f"missing={totals['missing']} failed={totals['failed']}")


if __name__ == "__main__":
    main()
