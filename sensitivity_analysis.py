"""
Sensitivity analysis for Atlas.

This script runs the same Atlas backtest under different assumptions:
portfolio size, rebalance frequency, factor weights, individual factors,
transaction costs, and portfolio weighting method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys
import types

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ATLAS_APP_PATH = BASE_DIR / "atlas_app.py"
OUTPUT_DIR = BASE_DIR.parent / "outputs" / "sensitivity_results"

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
VIX_SMOOTH_DAYS = 63
DEFAULT_TC_BPS = 10.0
DEFAULT_WEIGHTING = "equal"

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ",
    "ORCL", "MRK", "ABBV", "CVX", "NFLX", "KO", "CRM", "BAC", "WMT", "PEP",
    "AMD", "ADBE", "TMO", "MCD", "QCOM", "NKE", "LIN", "DIS", "CSCO", "ABT",
    "ACN", "VZ", "TXN", "DHR", "INTC", "NEE", "PM", "UPS", "MS", "AMGN",
]

DEFAULT_WEIGHTS = {
    "value_pe": 0.20,
    "profit_roe": 0.20,
    "growth_rev": 0.20,
    "risk_vol": 0.20,
    "risk_de": 0.20,
}


class _CacheDecorator:
    def __call__(self, *args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


class _Progress:
    def progress(self, *args, **kwargs):
        return None

    def empty(self):
        return None


class _StreamlitStub(types.SimpleNamespace):
    cache_data = _CacheDecorator()

    def progress(self, *args, **kwargs):
        return _Progress()


def load_atlas_engine() -> dict:
    """Load Atlas helper functions without executing the Streamlit UI section."""
    if not ATLAS_APP_PATH.exists():
        raise FileNotFoundError(f"Expected Atlas app at {ATLAS_APP_PATH}")

    source = ATLAS_APP_PATH.read_text(encoding="utf-8")
    marker = "# STREAMLIT APP"
    if marker not in source:
        raise ValueError("Could not find the Streamlit app marker in atlas_app.py")

    engine_source = source.split(marker, 1)[0]
    namespace = {}
    previous_streamlit = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _StreamlitStub()
    try:
        exec(compile(engine_source, str(ATLAS_APP_PATH), "exec"), namespace)
    finally:
        if previous_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = previous_streamlit
    return namespace


atlas = load_atlas_engine()
download_prices = atlas["download_prices"]
download_close = atlas["download_close"]
build_pit_fund_matrix = atlas["build_pit_fund_matrix"]
backtest = atlas["backtest"]
compute_vix_regimes = atlas["compute_vix_regimes"]
annualized_return = atlas["annualized_return"]
info_ratio = atlas["info_ratio"]


def run_backtest(
    prices: pd.DataFrame,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    bench_px: pd.Series,
    top_n: int = 30,
    rebalance: str = "ME",
    weights: Dict[str, float] | None = None,
    tc_bps: float = DEFAULT_TC_BPS,
    weighting: str = DEFAULT_WEIGHTING,
):
    return backtest(
        prices=prices,
        pit_fund_matrix=pit_fund_matrix,
        bench_px=bench_px,
        top_n=top_n,
        rebalance=rebalance,
        mom_lb=252,
        vol_lb=252,
        weights=weights or DEFAULT_WEIGHTS,
        tc_bps_per_100_turnover=tc_bps,
        weighting=weighting,
    )


def regime_result_row(label: str, out, vix_regimes: pd.Series) -> dict:
    df = pd.DataFrame({
        "net": out.net,
        "bench": out.bench,
        "regime": vix_regimes.reindex(out.net.index).ffill(),
    }).dropna()

    low = df[df["regime"] == "low_vol"]
    high = df[df["regime"] == "high_vol"]

    low_ir = info_ratio(low["net"], low["bench"]) if len(low) > 30 else np.nan
    high_ir = info_ratio(high["net"], high["bench"]) if len(high) > 30 else np.nan

    return {
        "test_case": label,
        "low_vix_days": int(len(low)),
        "high_vix_days": int(len(high)),
        "low_vix_info_ratio": low_ir,
        "high_vix_info_ratio": high_ir,
        "ir_differential": low_ir - high_ir if np.isfinite(low_ir) and np.isfinite(high_ir) else np.nan,
        "ir_ratio": low_ir / high_ir if np.isfinite(low_ir) and np.isfinite(high_ir) and high_ir != 0 else np.nan,
    }


def test_portfolio_size(
    prices: pd.DataFrame,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    bench_px: pd.Series,
    vix_regimes: pd.Series,
    sizes: List[int] | None = None,
) -> pd.DataFrame:
    results = []
    for size in sizes or [10, 15, 20, 30, 50]:
        if size > len(prices.columns):
            continue
        print(f"  Testing portfolio size: {size}...", end=" ")
        out = run_backtest(prices, pit_fund_matrix, bench_px, top_n=size)
        row = regime_result_row(str(size), out, vix_regimes)
        row["portfolio_size"] = size
        results.append(row)
        print("done")
    return pd.DataFrame(results)


def test_rebalance_frequency(
    prices: pd.DataFrame,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    bench_px: pd.Series,
    vix_regimes: pd.Series,
) -> pd.DataFrame:
    frequencies = {
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE",
        "Semi-annual": "2QE",
        "Annual": "YE",
    }
    results = []
    for label, freq in frequencies.items():
        print(f"  Testing rebalance frequency: {label}...", end=" ")
        out = run_backtest(prices, pit_fund_matrix, bench_px, rebalance=freq)
        row = regime_result_row(label, out, vix_regimes)
        row["rebalance_code"] = freq
        results.append(row)
        print("done")
    return pd.DataFrame(results)


def test_factor_weighting(
    prices: pd.DataFrame,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    bench_px: pd.Series,
    vix_regimes: pd.Series,
) -> pd.DataFrame:
    weighting_schemes = {
        "Equal Weight": DEFAULT_WEIGHTS,
        "Value Tilt": {"value_pe": 0.35, "profit_roe": 0.15, "growth_rev": 0.15, "risk_vol": 0.20, "risk_de": 0.15},
        "Profitability Tilt": {"value_pe": 0.15, "profit_roe": 0.35, "growth_rev": 0.15, "risk_vol": 0.20, "risk_de": 0.15},
        "Risk Focus": {"value_pe": 0.15, "profit_roe": 0.15, "growth_rev": 0.15, "risk_vol": 0.35, "risk_de": 0.20},
        "Growth Tilt": {"value_pe": 0.15, "profit_roe": 0.15, "growth_rev": 0.35, "risk_vol": 0.20, "risk_de": 0.15},
    }
    results = []
    for label, weights in weighting_schemes.items():
        print(f"  Testing factor weights: {label}...", end=" ")
        out = run_backtest(prices, pit_fund_matrix, bench_px, weights=weights)
        results.append(regime_result_row(label, out, vix_regimes))
        print("done")
    return pd.DataFrame(results)


def test_single_factors(
    prices: pd.DataFrame,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    bench_px: pd.Series,
    vix_regimes: pd.Series,
) -> pd.DataFrame:
    factor_configs = {
        "Value Only": {"value_pe": 1.0, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 0.0, "risk_de": 0.0},
        "Profitability Only": {"value_pe": 0.0, "profit_roe": 1.0, "growth_rev": 0.0, "risk_vol": 0.0, "risk_de": 0.0},
        "Growth Only": {"value_pe": 0.0, "profit_roe": 0.0, "growth_rev": 1.0, "risk_vol": 0.0, "risk_de": 0.0},
        "Volatility Only": {"value_pe": 0.0, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 1.0, "risk_de": 0.0},
        "Debt Only": {"value_pe": 0.0, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 0.0, "risk_de": 1.0},
        "Value + Volatility": {"value_pe": 0.5, "profit_roe": 0.0, "growth_rev": 0.0, "risk_vol": 0.5, "risk_de": 0.0},
        "Profitability + Growth": {"value_pe": 0.0, "profit_roe": 0.5, "growth_rev": 0.5, "risk_vol": 0.0, "risk_de": 0.0},
    }
    results = []
    for label, weights in factor_configs.items():
        print(f"  Testing factors: {label}...", end=" ")
        out = run_backtest(prices, pit_fund_matrix, bench_px, weights=weights)
        row = regime_result_row(label, out, vix_regimes)
        df = pd.DataFrame({
            "net": out.net,
            "regime": vix_regimes.reindex(out.net.index).ffill(),
        }).dropna()
        low = df[df["regime"] == "low_vol"]
        high = df[df["regime"] == "high_vol"]
        row["low_vix_return"] = annualized_return(low["net"]) if len(low) > 30 else np.nan
        row["high_vix_return"] = annualized_return(high["net"]) if len(high) > 30 else np.nan
        results.append(row)
        print("done")
    return pd.DataFrame(results)


def test_transaction_costs(
    prices: pd.DataFrame,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    bench_px: pd.Series,
    vix_regimes: pd.Series,
    costs: List[float] | None = None,
) -> pd.DataFrame:
    results = []
    for cost in costs or [0.0, 5.0, 10.0, 15.0, 20.0, 50.0]:
        print(f"  Testing transaction cost: {cost} bps...", end=" ")
        out = run_backtest(prices, pit_fund_matrix, bench_px, tc_bps=cost)
        row = regime_result_row(str(cost), out, vix_regimes)
        row["transaction_cost_bps"] = cost
        results.append(row)
        print("done")
    return pd.DataFrame(results)


def test_portfolio_weighting(
    prices: pd.DataFrame,
    pit_fund_matrix: Dict[str, pd.DataFrame],
    bench_px: pd.Series,
    vix_regimes: pd.Series,
) -> pd.DataFrame:
    methods = {
        "Equal weight": "equal",
        "Risk parity": "risk_parity",
        "Min variance": "min_variance",
    }
    results = []
    for label, method in methods.items():
        print(f"  Testing portfolio weighting: {label}...", end=" ")
        out = run_backtest(prices, pit_fund_matrix, bench_px, weighting=method)
        row = regime_result_row(label, out, vix_regimes)
        row["weighting_method"] = method
        results.append(row)
        print("done")
    return pd.DataFrame(results)


def build_inputs():
    print("\n[1/4] Downloading price data...")
    prices = download_prices(DEFAULT_TICKERS, START_DATE, END_DATE)
    if prices.empty:
        raise ValueError("No price data downloaded.")

    bench_df = download_prices([BENCHMARK], START_DATE, END_DATE)
    if BENCHMARK not in bench_df.columns:
        raise ValueError(f"Benchmark {BENCHMARK} was not found in downloaded data.")
    bench_px = bench_df[BENCHMARK]

    print("\n[2/4] Building point-in-time fundamental matrix...")
    rebal_dates = prices.resample("ME").last().index
    rebal_dates = rebal_dates[rebal_dates.isin(prices.index)]
    pit_fund_matrix = build_pit_fund_matrix(
        tickers=tuple(sorted(prices.columns.tolist())),
        rebal_dates=tuple(rebal_dates.tolist()),
        prices=prices,
        budget_s=240.0,
    )

    print("\n[3/4] Computing VIX regimes...")
    vix_close = download_close(VIX_TICKER, START_DATE, END_DATE)
    regimes, lo_thr, hi_thr = compute_vix_regimes(
        vix_close=vix_close,
        smooth_days=VIX_SMOOTH_DAYS,
        low_q=0.33,
        high_q=0.67,
        split_date=None,
    )
    regimes = regimes.reindex(prices.index).ffill()

    print(f"  Prices: {prices.shape}")
    print(f"  Monthly rebalance dates: {len(rebal_dates)}")
    print(f"  VIX thresholds: low <= {lo_thr:.2f}, high >= {hi_thr:.2f}")
    return prices, pit_fund_matrix, bench_px, regimes


def save_results(all_results: Dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for test_name, df in all_results.items():
        out_path = OUTPUT_DIR / f"sensitivity_{test_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")


def print_summary(all_results: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, df in all_results.items():
        if "ir_differential" not in df.columns:
            continue
        diffs = df["ir_differential"].dropna()
        if diffs.empty:
            print(f"\n{name}: no valid IR differential values")
            continue
        print(f"\n{name}:")
        print(f"  IR differential range: {diffs.min():.2f} to {diffs.max():.2f}")
        print(f"  Mean IR differential: {diffs.mean():.2f}")


def main() -> None:
    print("\n" + "=" * 80)
    print("ATLAS SENSITIVITY ANALYSIS")
    print("=" * 80)

    prices, pit_fund_matrix, bench_px, regimes = build_inputs()

    print("\n[4/4] Running robustness tests...")
    all_results = {
        "portfolio_size": test_portfolio_size(prices, pit_fund_matrix, bench_px, regimes),
        "rebalance_frequency": test_rebalance_frequency(prices, pit_fund_matrix, bench_px, regimes),
        "factor_weighting": test_factor_weighting(prices, pit_fund_matrix, bench_px, regimes),
        "single_factors": test_single_factors(prices, pit_fund_matrix, bench_px, regimes),
        "transaction_costs": test_transaction_costs(prices, pit_fund_matrix, bench_px, regimes),
        "portfolio_weighting": test_portfolio_weighting(prices, pit_fund_matrix, bench_px, regimes),
    }

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for test_name, df in all_results.items():
        print(f"\n{test_name.upper()}:")
        print(df.to_string(index=False))

    save_results(all_results)
    print_summary(all_results)


if __name__ == "__main__":
    main()
