# =============================================================================
#  src/black_scholes.py
#  Black-Scholes Option Pricing + Greeks
# =============================================================================
#
#  The Black-Scholes model prices European options using 5 inputs:
#    S  = current stock price
#    K  = strike price (the price you have the RIGHT to buy/sell at)
#    T  = time to expiration (in years)
#    r  = risk-free interest rate (e.g. 0.05 = 5%)
#    sigma = implied volatility (e.g. 0.20 = 20%)
#
#  Greeks measure how sensitive the option price is to each input.
#  The two we care about most for gamma scalping are:
#    Delta  = how much option price changes when stock moves $1
#    Gamma  = how much DELTA changes when stock moves $1
# =============================================================================

import numpy as np
from scipy.stats import norm   # standard normal distribution functions


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the price of a European CALL option using Black-Scholes.

    A CALL option gives the holder the RIGHT (not obligation) to BUY
    the stock at price K before expiry T.

    Parameters
    ----------
    S     : float  – current stock price
    K     : float  – strike price
    T     : float  – time to expiry in years (e.g. 30 days = 30/252)
    r     : float  – annual risk-free rate (0.05 = 5%)
    sigma : float  – implied volatility (0.20 = 20%)

    Returns
    -------
    float – fair value of the call option
    """
    # Clamp T to avoid division by zero on expiry day
    T = max(T, 1e-8)

    # d1 and d2 are intermediate values used in Black-Scholes
    # Think of them as "how far in the money" the option is in std-dev units
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # norm.cdf(x) = probability that a standard normal variable is <= x
    # N(d1) and N(d2) represent risk-adjusted probabilities
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price


def delta(S, K, T, r, sigma):
    """
    Delta = how much the option price changes per $1 move in the stock.

    For a call option, delta ranges from 0 to 1:
      delta = 0   -> deep out-of-the-money (stock far below strike)
      delta = 0.5 -> at-the-money (stock ≈ strike)
      delta = 1   -> deep in-the-money (stock far above strike)

    If delta = 0.60, and stock goes up $1, option goes up ~$0.60.
    To be 'delta-neutral' you short 0.60 shares per option you own.
    """
    T = max(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def gamma(S, K, T, r, sigma):
    """
    Gamma = how much DELTA changes per $1 move in the stock.

    Gamma is highest for at-the-money options near expiry.
    This is the key metric for gamma scalping:
      - High gamma means delta changes a LOT as stock moves
      - Each delta rebalance captures a bit of profit
      - The more the stock moves (higher realized vol), the more profit

    Think of it like this:
      If gamma = 0.02, and stock moves $1:
        -> delta changes by 0.02
        -> you need to buy/sell 0.02 shares to stay hedged
        -> that rebalance locks in a small profit from the price move
    """
    T = max(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    # norm.pdf = probability density function of standard normal
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Vega = how much option price changes per 1% change in implied volatility.
    (Not used in hedging but useful for understanding P&L attribution.)
    """
    T = max(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma):
    """
    Theta = time decay — how much the option loses per day passing.
    This is the COST of owning the option (gamma scalpers pay theta).
    Gamma scalping profit must exceed theta to be worthwhile.
    """
    T = max(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Theta expressed as daily decay (divided by 365)
    th = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
          - r * K * np.exp(-r * T) * norm.cdf(d2))
    return th / 365   # per calendar day


def all_greeks(S, K, T, r, sigma):
    """
    Return all greeks as a dictionary for easy access.
    """
    return {
        'delta' : delta(S, K, T, r, sigma),
        'gamma' : gamma(S, K, T, r, sigma),
        'vega'  : vega(S, K, T, r, sigma),
        'theta' : theta(S, K, T, r, sigma),
    }


# ── Quick test ─────────────────────────────────────────────
if __name__ == '__main__':
    # Example: stock at $100, strike $100, 30 days to expiry,
    # 5% risk-free rate, 20% implied volatility
    S, K, T, r, sigma = 100, 100, 30/252, 0.05, 0.20

    price = black_scholes_call(S, K, T, r, sigma)
    greeks = all_greeks(S, K, T, r, sigma)

    print("=" * 45)
    print("  Black-Scholes Option Pricing")
    print("=" * 45)
    print(f"  Stock Price (S)     : ${S}")
    print(f"  Strike Price (K)    : ${K}")
    print(f"  Days to Expiry      : 30")
    print(f"  Implied Volatility  : {sigma*100:.0f}%")
    print(f"  Risk-Free Rate      : {r*100:.0f}%")
    print("-" * 45)
    print(f"  Call Price          : ${price:.4f}")
    print(f"  Delta               : {greeks['delta']:.4f}")
    print(f"  Gamma               : {greeks['gamma']:.6f}")
    print(f"  Vega                : {greeks['vega']:.4f}")
    print(f"  Theta (daily)       : ${greeks['theta']:.4f}")
    print("=" * 45)
