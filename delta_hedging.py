# =============================================================================
#  src/delta_hedging.py
#  Delta Hedging Simulator
# =============================================================================
#
#  Delta Hedging means keeping your total portfolio delta = 0.
#  ("Delta neutral" = insensitive to small stock price moves)
#
#  How it works:
#    1. You BUY a call option (delta = e.g. 0.52, gamma = e.g. 0.04)
#    2. To hedge: SHORT 0.52 shares of the stock
#    3. Total delta = +0.52 (from option) - 0.52 (from short) = 0  ✓
#
#    4. Stock moves up $2. New delta = e.g. 0.56 (gamma kicked in)
#    5. You need to short MORE shares: sell 0.04 more shares
#    6. Each rebalance like this locks in a small profit
#
#  That locked-in profit is GAMMA SCALPING!
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from black_scholes import black_scholes_call, delta, gamma, all_greeks


def run_delta_hedge(prices, times, K, r, sigma_implied,
                    hedge_dt_minutes=60, cost_per_share=0.01,
                    n_options=1, minutes_per_day=390):
    """
    Run a full delta hedging simulation over a simulated price path.

    The strategy:
      - We OWN n_options call options (bought at t=0 at fair value)
      - We SHORT delta * n_options * 100 shares at each rebalance
        (1 option contract = 100 shares in real markets)
      - We rebalance every hedge_dt_minutes minutes

    Parameters
    ----------
    prices           : np.array – simulated stock price path (minute-by-minute)
    times            : np.array – time in years for each price step
    K                : float   – option strike price
    r                : float   – risk-free rate
    sigma_implied    : float   – implied volatility (used for pricing)
    hedge_dt_minutes : int     – how often to rebalance hedge (in minutes)
    cost_per_share   : float   – transaction cost per share traded ($)
    n_options        : int     – number of option contracts held
    minutes_per_day  : int     – trading minutes per day (390 standard)

    Returns
    -------
    results : dict with arrays of:
      - hedge_times      : times at which we rebalanced
      - hedge_prices     : stock prices at rebalance points
      - delta_values     : delta at each rebalance
      - gamma_values     : gamma at each rebalance
      - option_values    : option price at each rebalance
      - hedge_position   : shares held short (negative = short)
      - hedge_pnl        : cumulative P&L from hedge adjustments
      - option_pnl       : cumulative P&L from option value change
      - transaction_costs: cumulative transaction costs paid
      - total_pnl        : hedge_pnl + option_pnl - transaction_costs
      - shares_traded    : shares bought/sold at each rebalance
    """
    # ── Identify rebalance timestamps ────────────────────────
    # We rebalance every hedge_dt_minutes steps in the price array
    # (price array is at 1-minute resolution)
    rebalance_indices = list(range(0, len(prices), hedge_dt_minutes))
    if rebalance_indices[-1] != len(prices) - 1:
        rebalance_indices.append(len(prices) - 1)  # always include final step

    n_rebalances = len(rebalance_indices)

    # ── Pre-allocate output arrays ───────────────────────────
    hedge_times       = np.zeros(n_rebalances)
    hedge_prices      = np.zeros(n_rebalances)
    delta_values      = np.zeros(n_rebalances)
    gamma_values      = np.zeros(n_rebalances)
    option_values     = np.zeros(n_rebalances)
    hedge_position    = np.zeros(n_rebalances)   # shares short
    shares_traded     = np.zeros(n_rebalances)   # shares bought(+) or sold(-) at each step
    hedge_pnl         = np.zeros(n_rebalances)   # P&L from hedge rebalancing
    option_pnl        = np.zeros(n_rebalances)   # P&L from option value change
    transaction_costs = np.zeros(n_rebalances)   # cumulative costs paid

    # ── Step 0: Initial setup ────────────────────────────────
    S0         = prices[0]
    T0         = times[-1] - times[0]    # total time to expiry (years)
    option0    = black_scholes_call(S0, K, T0, r, sigma_implied)
    delta0     = delta(S0, K, T0, r, sigma_implied)
    gamma0     = gamma(S0, K, T0, r, sigma_implied)
    shares_per_contract = 100   # standard: 1 contract = 100 shares

    # Initial hedge: short delta0 * shares_per_contract * n_options shares
    initial_shares = delta0 * shares_per_contract * n_options
    current_position = -initial_shares   # negative = short position

    # Record initial state
    hedge_times[0]    = times[0]
    hedge_prices[0]   = S0
    delta_values[0]   = delta0
    gamma_values[0]   = gamma0
    option_values[0]  = option0
    hedge_position[0] = current_position
    shares_traded[0]  = initial_shares   # we sold these shares to set up the hedge

    cum_hedge_pnl   = 0.0
    cum_option_pnl  = 0.0
    cum_costs       = abs(initial_shares) * cost_per_share  # cost of initial hedge

    hedge_pnl[0]         = cum_hedge_pnl
    option_pnl[0]        = cum_option_pnl
    transaction_costs[0] = cum_costs

    prev_option_value = option0
    prev_price        = S0

    # ── Main loop: rebalance at each scheduled time ──────────
    for j in range(1, n_rebalances):
        idx = rebalance_indices[j]
        S   = prices[idx]
        t   = times[idx]

        # Time remaining to expiry (decreases as we move forward)
        T_remaining = times[-1] - t
        T_remaining = max(T_remaining, 1e-8)   # avoid division by zero at expiry

        # Calculate current Greeks and option value
        opt_val   = black_scholes_call(S, K, T_remaining, r, sigma_implied)
        d         = delta(S, K, T_remaining, r, sigma_implied)
        g         = gamma(S, K, T_remaining, r, sigma_implied)

        # ── P&L from hedge position ──────────────────────────
        # The hedge is SHORT shares, so profit = -position * (S_new - S_old)
        # If stock goes up by $1 and we are short 50 shares -> we lose $50
        price_change = S - prev_price
        hedge_profit_this_step = current_position * price_change
        cum_hedge_pnl += hedge_profit_this_step

        # ── P&L from option value change ─────────────────────
        # We are LONG the option, so profit = option_value_now - option_value_before
        option_profit_this_step = (opt_val - prev_option_value) * shares_per_contract * n_options
        cum_option_pnl += option_profit_this_step

        # ── Rebalance the hedge ──────────────────────────────
        # Target position = -delta * shares_per_contract * n_options (short delta shares)
        target_position = -d * shares_per_contract * n_options
        shares_to_trade  = target_position - current_position

        # Transaction costs = |shares traded| * cost_per_share
        cost_this_step   = abs(shares_to_trade) * cost_per_share
        cum_costs       += cost_this_step

        # Update position
        current_position = target_position

        # ── Store results ────────────────────────────────────
        hedge_times[j]       = t
        hedge_prices[j]      = S
        delta_values[j]      = d
        gamma_values[j]      = g
        option_values[j]     = opt_val
        hedge_position[j]    = current_position
        shares_traded[j]     = shares_to_trade
        hedge_pnl[j]         = cum_hedge_pnl
        option_pnl[j]        = cum_option_pnl
        transaction_costs[j] = cum_costs

        prev_option_value = opt_val
        prev_price        = S

    # ── Final P&L ────────────────────────────────────────────
    # Total P&L = option gains + hedge gains - transaction costs
    # Note: we also subtract the initial premium paid for the option
    initial_premium = option0 * shares_per_contract * n_options
    total_pnl = hedge_pnl + option_pnl - transaction_costs

    return {
        'hedge_times'       : hedge_times,
        'hedge_prices'      : hedge_prices,
        'delta_values'      : delta_values,
        'gamma_values'      : gamma_values,
        'option_values'     : option_values,
        'hedge_position'    : hedge_position,
        'shares_traded'     : shares_traded,
        'hedge_pnl'         : hedge_pnl,
        'option_pnl'        : option_pnl,
        'transaction_costs' : transaction_costs,
        'total_pnl'         : total_pnl,
        'initial_premium'   : initial_premium,
        'n_rebalances'      : n_rebalances,
    }


def summarize_hedge(results):
    """
    Print a human-readable summary of the delta hedging simulation.
    """
    r = results
    print(f"  Rebalance events         : {r['n_rebalances']}")
    print(f"  Final Hedge P&L          : ${r['hedge_pnl'][-1]:>8.2f}")
    print(f"  Final Option P&L         : ${r['option_pnl'][-1]:>8.2f}")
    print(f"  Total Transaction Costs  : ${r['transaction_costs'][-1]:>8.2f}")
    print(f"  NET Total P&L            : ${r['total_pnl'][-1]:>8.2f}")


# ── Quick test ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from price_simulation import simulate_gbm_fast

    print("=" * 50)
    print("  Delta Hedging Simulator Test")
    print("=" * 50)

    # Simulate prices at 1-minute resolution for 30 days
    prices, times = simulate_gbm_fast(100, 0.05, 0.25, T_days=30, dt_minutes=1)

    # Run delta hedge with hourly rebalancing
    results = run_delta_hedge(
        prices=prices, times=times,
        K=100, r=0.05,
        sigma_implied=0.20,
        hedge_dt_minutes=60,
        cost_per_share=0.01,
    )

    summarize_hedge(results)
    print("=" * 50)
