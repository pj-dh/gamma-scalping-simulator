# =============================================================================
#  src/pnl_analysis.py
#  P&L Analysis, Gamma Scalping Profit Attribution, Frequency Experiment
# =============================================================================
#
#  This module:
#   1. Breaks down total P&L into its components
#   2. Explains WHERE gamma scalping profits come from
#   3. Runs the optimal hedge frequency experiment
#   4. Compares realized vs implied volatility
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from price_simulation  import simulate_gbm_fast, compute_realized_volatility
from delta_hedging     import run_delta_hedge


# =============================================================================
#  PART A: GAMMA SCALPING PROFIT EXPLANATION
# =============================================================================

def theoretical_gamma_pnl(prices, times, K, r, sigma_implied, hedge_dt_minutes):
    """
    Estimate the THEORETICAL gamma scalping P&L using the classic formula.

    The gamma scalping P&L formula is:
        Profit per step ≈ 0.5 * Gamma * S² * (realized_vol² - implied_vol²) * dt

    This formula shows the core insight:
      - If realized_vol > implied_vol: each step gives POSITIVE profit
      - If realized_vol < implied_vol: each step gives NEGATIVE profit (loss)
      - The SIZE of profit/loss depends on gamma (bigger gamma = bigger moves)

    Parameters
    ----------
    prices           : simulated price array (1-min resolution)
    times            : time array
    K, r             : strike and risk-free rate
    sigma_implied    : the implied vol we PAID for (from the option premium)
    hedge_dt_minutes : rebalancing frequency

    Returns
    -------
    dict with theoretical P&L breakdown
    """
    from black_scholes import gamma as calc_gamma

    minutes_per_day  = 390
    rebalance_indices = list(range(0, len(prices), hedge_dt_minutes))

    theoretical_pnl_steps = []
    realized_vols         = []
    implied_vols          = []
    gammas                = []

    for j in range(1, len(rebalance_indices)):
        idx_now  = rebalance_indices[j]
        idx_prev = rebalance_indices[j-1]

        S_now   = prices[idx_now]
        S_prev  = prices[idx_prev]
        t_now   = times[idx_now]

        # Time remaining to expiry
        T_rem = max(times[-1] - t_now, 1e-8)

        # dt in years for this step
        dt_years = hedge_dt_minutes / (minutes_per_day * 252)

        # Gamma at midpoint
        S_mid   = 0.5 * (S_now + S_prev)
        g       = calc_gamma(S_mid, K, T_rem, r, sigma_implied)

        # Realized return squared for this step
        ret         = (S_now - S_prev) / S_prev
        ret_sq      = ret ** 2
        # Annualize: multiply by (minutes_per_year / dt_minutes)
        realized_var = ret_sq / dt_years      # realized variance rate (annualized)
        realized_v   = np.sqrt(max(realized_var, 0))

        # Theoretical P&L for this step (per share of underlying)
        # = 0.5 * Gamma * S^2 * (realized_var - implied_var) * dt
        implied_var  = sigma_implied ** 2
        step_pnl     = 0.5 * g * (S_mid ** 2) * (realized_var - implied_var) * dt_years

        theoretical_pnl_steps.append(step_pnl)
        realized_vols.append(realized_v)
        implied_vols.append(sigma_implied)
        gammas.append(g)

    cumulative_pnl = np.cumsum(theoretical_pnl_steps)

    return {
        'step_pnl'       : np.array(theoretical_pnl_steps),
        'cumulative_pnl' : cumulative_pnl,
        'realized_vols'  : np.array(realized_vols),
        'implied_vols'   : np.array(implied_vols),
        'gammas'         : np.array(gammas),
        'total_pnl'      : cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0,
    }


# =============================================================================
#  PART B: HEDGE FREQUENCY EXPERIMENT
# =============================================================================

def frequency_experiment(S0=100, K=100, mu=0.05, sigma_realized=0.25,
                          sigma_implied=0.20, T_days=30,
                          cost_per_share=0.01, seed=42):
    """
    The Hedge Frequency Experiment.

    Compare gamma scalping results at three different rebalancing frequencies:
      1. Every minute   (most frequent – captures more gamma but high costs)
      2. Every hour     (moderate frequency)
      3. Every day      (least frequent – low costs but misses moves)

    The optimal frequency balances:
      GAMMA PROFIT  (increases with more frequent rebalancing)
      TRANS COSTS   (also increases with more frequent rebalancing)

    Returns
    -------
    dict with results for each frequency
    """
    frequencies = {
        'Every Minute' : 1,
        'Every Hour'   : 60,
        'Every Day'    : 390,
    }

    # Simulate a SINGLE price path that all three strategies share
    # (Important: same path = fair comparison)
    prices, times = simulate_gbm_fast(
        S0=S0, mu=mu, sigma=sigma_realized,
        T_days=T_days, dt_minutes=1, seed=seed
    )

    results = {}

    for freq_name, freq_minutes in frequencies.items():
        hedge_results = run_delta_hedge(
            prices=prices, times=times,
            K=K, r=0.05,
            sigma_implied=sigma_implied,
            hedge_dt_minutes=freq_minutes,
            cost_per_share=cost_per_share,
        )

        theo = theoretical_gamma_pnl(
            prices=prices, times=times,
            K=K, r=0.05,
            sigma_implied=sigma_implied,
            hedge_dt_minutes=freq_minutes,
        )

        results[freq_name] = {
            'hedge_results'    : hedge_results,
            'theoretical'      : theo,
            'freq_minutes'     : freq_minutes,
            'n_rebalances'     : hedge_results['n_rebalances'],
            'final_total_pnl'  : hedge_results['total_pnl'][-1],
            'final_hedge_pnl'  : hedge_results['hedge_pnl'][-1],
            'final_option_pnl' : hedge_results['option_pnl'][-1],
            'total_costs'      : hedge_results['transaction_costs'][-1],
            'theo_pnl'         : theo['total_pnl'],
        }

    return results, prices, times


def compute_realized_vs_implied(prices, times, window_minutes=390, sigma_implied=0.20):
    """
    Calculate rolling realized volatility vs constant implied volatility.

    This comparison is the heart of gamma scalping logic:
      - If realized vol > implied vol for most of the period -> should profit
      - If realized vol < implied vol -> likely to lose money

    Parameters
    ----------
    prices         : price array (minute resolution)
    times          : time array
    window_minutes : rolling window in minutes
    sigma_implied  : constant implied vol (the vol we 'bought' via the option)

    Returns
    -------
    dict with realized_vol array, implied_vol array, times
    """
    log_returns = np.diff(np.log(prices))

    # Minutes per year for annualizing
    minutes_per_year = 390 * 252

    realized_vol = np.full(len(prices), np.nan)

    for i in range(window_minutes, len(log_returns) + 1):
        window = log_returns[i - window_minutes : i]
        realized_vol[i] = np.std(window) * np.sqrt(minutes_per_year)

    return {
        'realized_vol' : realized_vol,
        'implied_vol'  : np.full(len(prices), sigma_implied),
        'times'        : times,
        'avg_realized' : np.nanmean(realized_vol),
        'avg_implied'  : sigma_implied,
        'vol_premium'  : sigma_implied - np.nanmean(realized_vol),
    }


def pnl_breakdown_summary(results_dict):
    """
    Print a formatted P&L breakdown table comparing all hedge frequencies.
    """
    print("\n" + "=" * 70)
    print(f"  {'Frequency':<15} {'Rebalances':>12} {'Hedge P&L':>12} "
          f"{'Option P&L':>12} {'Trans Cost':>12} {'Net P&L':>10}")
    print("=" * 70)
    for name, res in results_dict.items():
        print(
            f"  {name:<15}"
            f" {res['n_rebalances']:>12,}"
            f" ${res['final_hedge_pnl']:>10.2f}"
            f" ${res['final_option_pnl']:>10.2f}"
            f" ${res['total_costs']:>10.2f}"
            f" ${res['final_total_pnl']:>8.2f}"
        )
    print("=" * 70)


# =============================================================================
#  PART C: SCENARIO ANALYSIS
# =============================================================================

def scenario_analysis(S0=100, K=100, T_days=30, r=0.05,
                       sigma_implied=0.20, cost_per_share=0.01):
    """
    Run gamma scalping under three volatility scenarios:
      1. High vol: realized > implied  -> should PROFIT (the ideal scenario)
      2. Fair vol: realized ≈ implied  -> roughly break-even
      3. Low vol:  realized < implied  -> should LOSE (paid too much for option)

    This demonstrates the core risk of gamma scalping:
    you are essentially BETTING that realized vol will exceed implied vol.
    """
    scenarios = {
        'High Vol (rv=30%, iv=20%)' : {'sigma_realized': 0.30, 'seed': 1},
        'Fair Vol (rv=20%, iv=20%)' : {'sigma_realized': 0.20, 'seed': 2},
        'Low  Vol (rv=12%, iv=20%)' : {'sigma_realized': 0.12, 'seed': 3},
    }

    results = {}
    all_prices = {}
    all_times  = {}

    for scenario_name, params in scenarios.items():
        prices, times = simulate_gbm_fast(
            S0=S0, mu=0.05,
            sigma=params['sigma_realized'],
            T_days=T_days, dt_minutes=1,
            seed=params['seed']
        )

        hedge_res = run_delta_hedge(
            prices=prices, times=times,
            K=K, r=r,
            sigma_implied=sigma_implied,
            hedge_dt_minutes=60,   # hourly hedging for all scenarios
            cost_per_share=cost_per_share,
        )

        results[scenario_name] = {
            'hedge_results'   : hedge_res,
            'sigma_realized'  : params['sigma_realized'],
            'sigma_implied'   : sigma_implied,
            'final_pnl'       : hedge_res['total_pnl'][-1],
            'final_hedge_pnl' : hedge_res['hedge_pnl'][-1],
            'total_costs'     : hedge_res['transaction_costs'][-1],
        }
        all_prices[scenario_name] = prices
        all_times[scenario_name]  = times

    return results, all_prices, all_times


# ── Quick test ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  P&L Analysis Module Test")
    print("=" * 55)

    print("\nRunning Hedge Frequency Experiment...")
    freq_results, prices, times = frequency_experiment(
        sigma_realized=0.25, sigma_implied=0.20, T_days=30
    )
    pnl_breakdown_summary(freq_results)

    print("\nRunning Scenario Analysis...")
    scen_results, _, _ = scenario_analysis()
    print(f"\n  {'Scenario':<35} {'Net P&L':>10}")
    print("  " + "-" * 47)
    for name, res in scen_results.items():
        print(f"  {name:<35} ${res['final_pnl']:>8.2f}")
    print("=" * 55)
