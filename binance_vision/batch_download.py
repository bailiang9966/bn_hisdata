"""
Batch automation script to backfill data from 2025-06-01 to latest date.
Runs task stages (monthly klines/aggTrades, daily metrics, and recent daily klines/aggTrades).
"""
import datetime
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNLOAD_SCRIPT = HERE / "download_binance_vision.py"

START_DATE = "2025-06-01"
TODAY = datetime.date.today()
YESTERDAY = TODAY - datetime.timedelta(days=1)
PREV_MONTH_END = TODAY.replace(day=1) - datetime.timedelta(days=1)
CURRENT_MONTH_START = TODAY.replace(day=1)

LOG_FILE = HERE / "batch_download.log"


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str]):
    log(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log(res.stdout)
    if res.returncode != 0:
        log(f"Command failed with code {res.returncode}")
    else:
        log("Stage completed successfully.")


def main():
    log("=== Starting Batch Backfill ===")
    log(f"Date range: {START_DATE} -> {YESTERDAY}")

    workers = "8"

    # Stage 1: Monthly Klines (2025-06 to prev month end)
    log("--- Stage 1: Monthly Klines ---")
    run_cmd([
        sys.executable, str(DOWNLOAD_SCRIPT),
        "--market", "both",
        "--symbol", "all",
        "--category", "klines",
        "--interval", "1m",
        "--granularity", "monthly",
        "--start-date", START_DATE,
        "--end-date", PREV_MONTH_END.strftime("%Y-%m-%d"),
        "--workers", workers,
    ])

    # Stage 2: Monthly aggTrades (2025-06 to prev month end)
    log("--- Stage 2: Monthly aggTrades ---")
    run_cmd([
        sys.executable, str(DOWNLOAD_SCRIPT),
        "--market", "both",
        "--symbol", "all",
        "--category", "aggTrades",
        "--granularity", "monthly",
        "--start-date", START_DATE,
        "--end-date", PREV_MONTH_END.strftime("%Y-%m-%d"),
        "--workers", workers,
    ])

    # Stage 3: Daily Metrics (2025-06 to yesterday)
    log("--- Stage 3: Daily Metrics ---")
    run_cmd([
        sys.executable, str(DOWNLOAD_SCRIPT),
        "--market", "um",
        "--symbol", "all",
        "--category", "metrics",
        "--granularity", "daily",
        "--start-date", START_DATE,
        "--end-date", YESTERDAY.strftime("%Y-%m-%d"),
        "--workers", workers,
    ])

    # Stage 4: Current Month Daily Klines & aggTrades
    log("--- Stage 4: Current Month Daily Klines & aggTrades ---")
    run_cmd([
        sys.executable, str(DOWNLOAD_SCRIPT),
        "--market", "both",
        "--symbol", "all",
        "--category", "klines",
        "--interval", "1m",
        "--granularity", "daily",
        "--start-date", CURRENT_MONTH_START.strftime("%Y-%m-%d"),
        "--end-date", YESTERDAY.strftime("%Y-%m-%d"),
        "--workers", workers,
    ])
    run_cmd([
        sys.executable, str(DOWNLOAD_SCRIPT),
        "--market", "both",
        "--symbol", "all",
        "--category", "aggTrades",
        "--granularity", "daily",
        "--start-date", CURRENT_MONTH_START.strftime("%Y-%m-%d"),
        "--end-date", YESTERDAY.strftime("%Y-%m-%d"),
        "--workers", workers,
    ])

    log("=== Batch Backfill Completed ===")


if __name__ == "__main__":
    main()
