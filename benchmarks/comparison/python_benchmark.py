#!/usr/bin/env python3
"""
Python Benchmark Suite for QuantNova Comparison

Compares against:
- vectorbt: Fast vectorized backtesting
- pandas: Data manipulation and rolling statistics
- statsmodels: Factor regressions, statistical tests
- numpy: Numerical operations

Run: python python_benchmark.py
Requirements: pip install vectorbt pandas numpy statsmodels scipy
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try optional imports
try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False
    print("vectorbt not installed, skipping backtest benchmarks")

try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("statsmodels not installed, skipping regression benchmarks")


def benchmark(func, n_runs=100, n_warmup=5):
    """Time a function, return median time in microseconds."""
    # Warmup
    for _ in range(n_warmup):
        func()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to μs

    return {
        'median': np.median(times),
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


# =============================================================================
# Backtesting Benchmarks
# =============================================================================

def run_backtest_benchmarks():
    """Benchmark backtesting operations."""
    print("\n" + "=" * 70)
    print("BACKTESTING BENCHMARKS (vectorbt / pandas)")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic price data (5 years daily)
    n_days = 252 * 5
    n_assets = 10
    dates = pd.date_range('2019-01-01', periods=n_days, freq='D')

    # Random walk prices
    returns = np.random.randn(n_days, n_assets) * 0.02
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=[f'ASSET_{i}' for i in range(n_assets)]
    )

    results = {}

    # -------------------------------------------------------------------------
    # Simple Moving Average Crossover
    # -------------------------------------------------------------------------
    print("\n[1/4] SMA Crossover Strategy (single asset, 5 years)")

    single_price = prices.iloc[:, 0]

    def sma_crossover_pandas():
        fast = single_price.rolling(20).mean()
        slow = single_price.rolling(50).mean()
        signal = (fast > slow).astype(int)
        returns = single_price.pct_change() * signal.shift(1)
        cum_returns = (1 + returns).cumprod()
        return cum_returns.iloc[-1]

    t = benchmark(sma_crossover_pandas, n_runs=100)
    print(f"  pandas:     {t['median']:.1f} μs")
    results['sma_crossover_pandas'] = t['median']

    if HAS_VBT:
        def sma_crossover_vbt():
            fast = vbt.MA.run(single_price, 20)
            slow = vbt.MA.run(single_price, 50)
            entries = fast.ma_crossed_above(slow)
            exits = fast.ma_crossed_below(slow)
            pf = vbt.Portfolio.from_signals(single_price, entries, exits)
            return pf.total_return()

        t = benchmark(sma_crossover_vbt, n_runs=20)
        print(f"  vectorbt:   {t['median']:.1f} μs")
        results['sma_crossover_vbt'] = t['median']

    # -------------------------------------------------------------------------
    # Portfolio Rebalancing
    # -------------------------------------------------------------------------
    print("\n[2/4] Monthly Rebalancing (10 assets, 5 years)")

    def monthly_rebalance_pandas():
        monthly = prices.resample('M').last()
        weights = np.ones(n_assets) / n_assets
        returns = monthly.pct_change()
        port_returns = (returns * weights).sum(axis=1)
        cum_returns = (1 + port_returns).cumprod()
        return cum_returns.iloc[-1]

    t = benchmark(monthly_rebalance_pandas, n_runs=100)
    print(f"  pandas:     {t['median']:.1f} μs")
    results['rebalance_pandas'] = t['median']

    # -------------------------------------------------------------------------
    # Rolling Sharpe Ratio
    # -------------------------------------------------------------------------
    print("\n[3/4] Rolling Sharpe (252-day window, 5 years)")

    returns_series = single_price.pct_change().dropna()

    def rolling_sharpe_pandas():
        rolling_mean = returns_series.rolling(252).mean()
        rolling_std = returns_series.rolling(252).std()
        sharpe = (rolling_mean * 252) / (rolling_std * np.sqrt(252))
        return sharpe.iloc[-1]

    t = benchmark(rolling_sharpe_pandas, n_runs=100)
    print(f"  pandas:     {t['median']:.1f} μs")
    results['rolling_sharpe_pandas'] = t['median']

    # -------------------------------------------------------------------------
    # Full Backtest Metrics
    # -------------------------------------------------------------------------
    print("\n[4/4] Full Metrics Calculation (Sharpe, MaxDD, etc.)")

    def compute_metrics_pandas():
        rets = returns_series
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
        cum = (1 + rets).cumprod()
        drawdown = cum / cum.cummax() - 1
        max_dd = drawdown.min()
        vol = rets.std() * np.sqrt(252)
        total_ret = cum.iloc[-1] - 1
        return {'sharpe': sharpe, 'max_dd': max_dd, 'vol': vol, 'total_ret': total_ret}

    t = benchmark(compute_metrics_pandas, n_runs=100)
    print(f"  pandas:     {t['median']:.1f} μs")
    results['metrics_pandas'] = t['median']

    return results


# =============================================================================
# Factor Model Benchmarks
# =============================================================================

def run_factor_benchmarks():
    """Benchmark factor model operations."""
    print("\n" + "=" * 70)
    print("FACTOR MODEL BENCHMARKS (statsmodels / numpy)")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic factor data
    n = 252 * 5  # 5 years daily

    # Factors
    mkt = np.random.randn(n) * 0.01 + 0.0003
    smb = np.random.randn(n) * 0.005 + 0.0001
    hml = np.random.randn(n) * 0.005 + 0.00005

    # Strategy returns with known loadings
    alpha = 0.0002
    beta_mkt, beta_smb, beta_hml = 1.1, 0.3, -0.2
    noise = np.random.randn(n) * 0.005
    returns = alpha + beta_mkt * mkt + beta_smb * smb + beta_hml * hml + noise

    results = {}

    # -------------------------------------------------------------------------
    # OLS Regression (CAPM)
    # -------------------------------------------------------------------------
    print("\n[1/4] CAPM Regression (OLS)")

    def capm_numpy():
        X = np.column_stack([np.ones(n), mkt])
        beta = np.linalg.lstsq(X, returns, rcond=None)[0]
        return beta

    t = benchmark(capm_numpy, n_runs=1000)
    print(f"  numpy:      {t['median']:.1f} μs")
    results['capm_numpy'] = t['median']

    if HAS_SM:
        def capm_statsmodels():
            X = sm.add_constant(mkt)
            model = sm.OLS(returns, X).fit()
            return model.params

        t = benchmark(capm_statsmodels, n_runs=100)
        print(f"  statsmodels: {t['median']:.1f} μs")
        results['capm_sm'] = t['median']

    # -------------------------------------------------------------------------
    # Fama-French 3-Factor
    # -------------------------------------------------------------------------
    print("\n[2/4] Fama-French 3-Factor Regression")

    factors = np.column_stack([mkt, smb, hml])

    def ff3_numpy():
        X = np.column_stack([np.ones(n), factors])
        beta = np.linalg.lstsq(X, returns, rcond=None)[0]
        y_hat = X @ beta
        residuals = returns - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((returns - returns.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return beta, r2

    t = benchmark(ff3_numpy, n_runs=1000)
    print(f"  numpy:      {t['median']:.1f} μs")
    results['ff3_numpy'] = t['median']

    if HAS_SM:
        def ff3_statsmodels():
            X = sm.add_constant(factors)
            model = sm.OLS(returns, X).fit()
            return model.params, model.rsquared

        t = benchmark(ff3_statsmodels, n_runs=100)
        print(f"  statsmodels: {t['median']:.1f} μs")
        results['ff3_sm'] = t['median']

    # -------------------------------------------------------------------------
    # Rolling Beta
    # -------------------------------------------------------------------------
    print("\n[3/4] Rolling Beta (60-day window)")

    def rolling_beta_numpy():
        window = 60
        betas = np.full(n, np.nan)
        for i in range(window, n):
            y = returns[i-window:i]
            x = mkt[i-window:i]
            cov = np.cov(y, x)[0, 1]
            var = np.var(x)
            betas[i] = cov / var
        return betas

    t = benchmark(rolling_beta_numpy, n_runs=10)
    print(f"  numpy loop: {t['median']/1000:.1f} ms")
    results['rolling_beta_numpy'] = t['median']

    def rolling_beta_pandas():
        ret_s = pd.Series(returns)
        mkt_s = pd.Series(mkt)
        cov = ret_s.rolling(60).cov(mkt_s)
        var = mkt_s.rolling(60).var()
        return cov / var

    t = benchmark(rolling_beta_pandas, n_runs=100)
    print(f"  pandas:     {t['median']:.1f} μs")
    results['rolling_beta_pandas'] = t['median']

    # -------------------------------------------------------------------------
    # Information Coefficient
    # -------------------------------------------------------------------------
    print("\n[4/4] Information Coefficient (rank correlation)")

    predictions = np.random.randn(1000)
    outcomes = 0.3 * predictions + 0.7 * np.random.randn(1000)

    def ic_scipy():
        return stats.spearmanr(predictions, outcomes)[0]

    t = benchmark(ic_scipy, n_runs=1000)
    print(f"  scipy:      {t['median']:.1f} μs")
    results['ic_scipy'] = t['median']

    return results


# =============================================================================
# Statistical Testing Benchmarks
# =============================================================================

def run_statistics_benchmarks():
    """Benchmark statistical testing operations."""
    print("\n" + "=" * 70)
    print("STATISTICAL TESTING BENCHMARKS")
    print("=" * 70)

    np.random.seed(42)
    returns = np.random.randn(252 * 5) * 0.01 + 0.0004  # 5 years daily

    results = {}

    # -------------------------------------------------------------------------
    # Sharpe Ratio
    # -------------------------------------------------------------------------
    print("\n[1/3] Sharpe Ratio Calculation")

    def sharpe_numpy():
        return returns.mean() / returns.std() * np.sqrt(252)

    t = benchmark(sharpe_numpy, n_runs=10000)
    print(f"  numpy:      {t['median']:.2f} μs")
    results['sharpe_numpy'] = t['median']

    # -------------------------------------------------------------------------
    # Sharpe Confidence Interval
    # -------------------------------------------------------------------------
    print("\n[2/3] Sharpe Confidence Interval (Lo 2002)")

    def sharpe_ci_numpy():
        n = len(returns)
        sr = returns.mean() / returns.std() * np.sqrt(252)
        se = np.sqrt((1 + 0.5 * sr**2) / n) * np.sqrt(252)
        z = 1.96
        return sr - z * se, sr + z * se

    t = benchmark(sharpe_ci_numpy, n_runs=10000)
    print(f"  numpy:      {t['median']:.2f} μs")
    results['sharpe_ci_numpy'] = t['median']

    # -------------------------------------------------------------------------
    # T-test
    # -------------------------------------------------------------------------
    print("\n[3/3] T-test (returns > 0)")

    def ttest_scipy():
        return stats.ttest_1samp(returns, 0)

    t = benchmark(ttest_scipy, n_runs=1000)
    print(f"  scipy:      {t['median']:.1f} μs")
    results['ttest_scipy'] = t['median']

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PYTHON BENCHMARK SUITE FOR QUANTNOVA COMPARISON")
    print("=" * 70)
    print(f"\nnumpy version:  {np.__version__}")
    print(f"pandas version: {pd.__version__}")
    if HAS_VBT:
        print(f"vectorbt version: {vbt.__version__}")
    if HAS_SM:
        print(f"statsmodels version: {sm.__version__}")

    all_results = {}

    bt_results = run_backtest_benchmarks()
    all_results.update(bt_results)

    factor_results = run_factor_benchmarks()
    all_results.update(factor_results)

    stats_results = run_statistics_benchmarks()
    all_results.update(stats_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (all times in μs unless noted)")
    print("=" * 70)
    for name, time in sorted(all_results.items()):
        if time > 1000:
            print(f"  {name:30s} {time/1000:10.2f} ms")
        else:
            print(f"  {name:30s} {time:10.2f} μs")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
