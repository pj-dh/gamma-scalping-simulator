# =============================================================================
#  src/price_simulation.py
#  Geometric Brownian Motion (GBM) Stock Price Simulator
# =============================================================================
#
#  Geometric Brownian Motion (GBM) is the standard model for stock prices.
#  The key idea: each day's price = yesterday's price × a random multiplier.
#
#  The multiplier has two parts:
#    1. DRIFT   (mu)   – the average direction (stocks tend to go up slowly)
#    2. SHOCKS  (sigma) – random daily surprises (up or down)
#
#  The formula per time step:
#    S(t+dt) = S(t) * exp( (mu - sigma²/2)*dt  +  sigma * sqrt(dt) * Z )
#    where Z ~ Normal(0,1) – a standard random draw
#
#  This is the same model Black-Scholes assumes underneath.
# =============================================================================

import numpy as np
import pandas as pd


def simulate_gbm(S0, mu, sigma, T_days, dt_minutes=1, seed=42):
    """
    Simulate a stock price path using Geometric Brownian Motion.

    Parameters
    ----------
    S0         : float – starting stock price (e.g. 100.0)
    mu         : float – annual drift/return (e.g. 0.05 = 5% per year)
    sigma      : float – annual volatility (e.g. 0.20 = 20% per year)
    T_days     : int   – total number of trading DAYS to simulate
    dt_minutes : int   – time step size in minutes (1 = tick-by-tick, 60 = hourly, etc.)
    seed       : int   – random seed for reproducibility

    Returns
    -------
    prices : np.array – array of simulated stock prices
    times  : np.array – corresponding time index (in years)
    """
    np.random.seed(seed)

    # Convert annual parameters to per-minute scale
    # There are 390 trading minutes in a day (6.5 hour session)
    minutes_per_day  = 390
    total_minutes    = T_days * minutes_per_day
    n_steps          = total_minutes // dt_minutes   # how many steps we simulate

    # dt = size of one time step in years
    dt = dt_minutes / (minutes_per_day * 252)

    # Pre-allocate array for speed (faster than appending in a loop)
    prices = np.zeros(n_steps + 1)
    prices[0] = S0

    # GBM formula step by step
    for i in range(1, n_steps + 1):
        Z = np.random.standard_normal()          # random standard-normal number
        drift_term    = (mu - 0.5 * sigma**2) * dt  # deterministic component
        diffusion_term = sigma * np.sqrt(dt) * Z    # random component
        prices[i] = prices[i-1] * np.exp(drift_term + diffusion_term)

    # Build a time array from 0 to T (in years)
    times = np.linspace(0, T_days / 252, n_steps + 1)

    return prices, times


def simulate_gbm_fast(S0, mu, sigma, T_days, dt_minutes=1, seed=42):
    """
    Vectorized (faster) version of GBM simulation.
    Same result as simulate_gbm but uses numpy array operations
    instead of a Python for-loop — much faster for large simulations.
    """
    np.random.seed(seed)

    minutes_per_day = 390
    n_steps = (T_days * minutes_per_day) // dt_minutes
    dt = dt_minutes / (minutes_per_day * 252)

    # Generate ALL random numbers at once (vectorized)
    Z = np.random.standard_normal(n_steps)

    # Calculate all log-returns in one go
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum of log-returns then exponentiate = price path
    log_prices = np.log(S0) + np.concatenate([[0], np.cumsum(log_returns)])
    prices = np.exp(log_prices)

    times = np.linspace(0, T_days / 252, n_steps + 1)

    return prices, times


def compute_realized_volatility(prices, window_steps):
    """
    Calculate REALIZED volatility over a rolling window.

    Realized vol = what the stock ACTUALLY did (backwards-looking).
    Implied vol  = what the market EXPECTS the stock to do (forward-looking).

    The difference between realized and implied vol is the core
    profit driver of gamma scalping.

    Parameters
    ----------
    prices       : np.array – price series
    window_steps : int      – number of steps in the rolling window

    Returns
    -------
    np.array – annualized realized volatility at each step
    """
    # Calculate log-returns: ln(S[t] / S[t-1])
    log_returns = np.diff(np.log(prices))

    realized_vol = np.zeros(len(prices))

    for i in range(window_steps, len(log_returns) + 1):
        window = log_returns[i - window_steps : i]
        # std of log-returns then annualize
        # 252 * 390 = total minutes in a year (for minute-level data)
        realized_vol[i] = np.std(window) * np.sqrt(252 * 390)

    # Fill the initial window with NaN (not enough data yet)
    realized_vol[:window_steps] = np.nan

    return realized_vol


def downsample_prices(prices, times, original_dt_min, target_dt_min):
    """
    Downsample a high-frequency price series to a lower frequency.
    e.g. convert 1-minute prices to hourly or daily prices.

    Parameters
    ----------
    prices         : np.array – original price series
    times          : np.array – original time series
    original_dt_min: int – original time step in minutes
    target_dt_min  : int – desired time step in minutes

    Returns
    -------
    sampled_prices, sampled_times
    """
    step = target_dt_min // original_dt_min
    sampled_prices = prices[::step]
    sampled_times  = times[::step]
    return sampled_prices, sampled_times


# ── Quick test ─────────────────────────────────────────────
if __name__ == '__main__':
    prices, times = simulate_gbm_fast(
        S0=100, mu=0.05, sigma=0.20, T_days=30, dt_minutes=1
    )

    print("=" * 45)
    print("  GBM Stock Price Simulation")
    print("=" * 45)
    print(f"  Total steps simulated : {len(prices):,}")
    print(f"  Starting price        : ${prices[0]:.2f}")
    print(f"  Final price           : ${prices[-1]:.2f}")
    print(f"  Min price             : ${prices.min():.2f}")
    print(f"  Max price             : ${prices.max():.2f}")
    print(f"  Actual return         : {(prices[-1]/prices[0]-1)*100:.2f}%")
    print("=" * 45)
