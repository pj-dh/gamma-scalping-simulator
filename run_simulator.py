# =============================================================================
#  dashboard/run_simulator.py
#  Gamma Scalping Training Simulator — Main Entry Point
# =============================================================================
#
#  Run this file to execute the full simulation and generate all charts.
#  Charts are saved to the data/ folder.
#
#  Usage:
#    python dashboard/run_simulator.py
#    (or from the project root: python -m dashboard.run_simulator)
# =============================================================================

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Make sure Python can find our src/ modules
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

from black_scholes    import black_scholes_call, all_greeks
from price_simulation import simulate_gbm_fast
from delta_hedging    import run_delta_hedge, summarize_hedge
from pnl_analysis     import (frequency_experiment, scenario_analysis,
                               compute_realized_vs_implied,
                               theoretical_gamma_pnl, pnl_breakdown_summary)

# ── Matplotlib style ─────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.facecolor' : '#0F1923',
    'axes.facecolor'   : '#16202A',
    'axes.edgecolor'   : '#2D4356',
    'grid.color'       : '#1E2E3D',
    'grid.alpha'       : 0.8,
    'text.color'       : '#E0E0E0',
    'axes.labelcolor'  : '#B0BEC5',
    'xtick.color'      : '#78909C',
    'ytick.color'      : '#78909C',
    'axes.titlecolor'  : '#E0F2F1',
    'legend.facecolor' : '#1A2535',
    'legend.edgecolor' : '#2D4356',
    'font.size'        : 9,
})

# ── Colour palette ────────────────────────────────────────────
C_BLUE     = '#42A5F5'
C_GREEN    = '#66BB6A'
C_RED      = '#EF5350'
C_ORANGE   = '#FFA726'
C_TEAL     = '#26C6DA'
C_PURPLE   = '#AB47BC'
C_YELLOW   = '#FFEE58'
C_GRAY     = '#78909C'


# =============================================================================
#  SIMULATION PARAMETERS
# =============================================================================
print("=" * 60)
print("   GAMMA SCALPING TRAINING SIMULATOR")
print("   Quantitative Finance Learning Tool")
print("=" * 60)

# --- Core Parameters ---
S0             = 100.0    # starting stock price
K              = 100.0    # option strike (at-the-money)
T_DAYS         = 30       # simulation length in trading days
MU             = 0.05     # stock drift (5% annual)
SIGMA_REALIZED = 0.25     # what the stock ACTUALLY does (25% vol)
SIGMA_IMPLIED  = 0.20     # what the option ASSUMES (20% vol)
R              = 0.05     # risk-free rate
COST_PER_SHARE = 0.01     # $0.01 transaction cost per share traded
SEED           = 42

print(f"\n  Simulation Parameters:")
print(f"  Stock Price (S0)      : ${S0}")
print(f"  Strike Price (K)      : ${K}")
print(f"  Duration              : {T_DAYS} trading days")
print(f"  Realized Volatility   : {SIGMA_REALIZED*100:.0f}%  (what stock actually does)")
print(f"  Implied Volatility    : {SIGMA_IMPLIED*100:.0f}%  (what option assumes)")
print(f"  Vol Premium (rv - iv) : {(SIGMA_REALIZED - SIGMA_IMPLIED)*100:.0f}%  (positive -> gamma scalper profits)")
print(f"  Transaction Cost      : ${COST_PER_SHARE}/share")

# =============================================================================
#  STEP 1: Show Black-Scholes option pricing
# =============================================================================
print("\n" + "─" * 60)
print("  STEP 1: Black-Scholes Option Pricing")
print("─" * 60)

T0 = T_DAYS / 252
option_price = black_scholes_call(S0, K, T0, R, SIGMA_IMPLIED)
greeks = all_greeks(S0, K, T0, R, SIGMA_IMPLIED)

print(f"  Call Option Price     : ${option_price:.4f}")
print(f"  Delta                 : {greeks['delta']:.4f}")
print(f"  Gamma                 : {greeks['gamma']:.6f}")
print(f"  Vega                  : {greeks['vega']:.4f}")
print(f"  Theta (daily)         : ${greeks['theta']:.4f}")
print(f"\n  Delta = {greeks['delta']:.2f} means: option moves ${greeks['delta']:.2f} per $1 stock move")
print(f"  Gamma = {greeks['gamma']:.4f} means: delta changes by {greeks['gamma']:.4f} per $1 stock move")

# =============================================================================
#  STEP 2: Simulate prices
# =============================================================================
print("\n" + "─" * 60)
print("  STEP 2: Simulating Stock Price Path (GBM)")
print("─" * 60)

prices, times = simulate_gbm_fast(S0, MU, SIGMA_REALIZED, T_DAYS, dt_minutes=1, seed=SEED)
print(f"  Price steps simulated : {len(prices):,}")
print(f"  Start price           : ${prices[0]:.2f}")
print(f"  End price             : ${prices[-1]:.2f}")
print(f"  Min price             : ${prices.min():.2f}")
print(f"  Max price             : ${prices.max():.2f}")

# =============================================================================
#  STEP 3: Run delta hedge with hourly rebalancing
# =============================================================================
print("\n" + "─" * 60)
print("  STEP 3: Delta Hedging Simulation (Hourly Rebalancing)")
print("─" * 60)

hedge_results = run_delta_hedge(
    prices=prices, times=times,
    K=K, r=R,
    sigma_implied=SIGMA_IMPLIED,
    hedge_dt_minutes=60,
    cost_per_share=COST_PER_SHARE,
)
summarize_hedge(hedge_results)

# =============================================================================
#  STEP 4: Hedge frequency experiment
# =============================================================================
print("\n" + "─" * 60)
print("  STEP 4: Hedge Frequency Experiment")
print("─" * 60)

freq_results, _, _ = frequency_experiment(
    S0=S0, K=K, mu=MU,
    sigma_realized=SIGMA_REALIZED,
    sigma_implied=SIGMA_IMPLIED,
    T_days=T_DAYS,
    cost_per_share=COST_PER_SHARE,
    seed=SEED
)
pnl_breakdown_summary(freq_results)

# =============================================================================
#  STEP 5: Scenario analysis
# =============================================================================
print("\n" + "─" * 60)
print("  STEP 5: Volatility Scenario Analysis")
print("─" * 60)

scen_results, scen_prices, scen_times = scenario_analysis(
    S0=S0, K=K, T_days=T_DAYS, r=R,
    sigma_implied=SIGMA_IMPLIED,
    cost_per_share=COST_PER_SHARE
)
print(f"\n  {'Scenario':<38} {'Net P&L':>10} {'Vol Premium':>14}")
print("  " + "─" * 64)
for name, res in scen_results.items():
    vp = (res['sigma_realized'] - res['sigma_implied']) * 100
    print(f"  {name:<38} ${res['final_pnl']:>8.2f}   {vp:>+6.0f}% vol premium")

# =============================================================================
#  STEP 6: Realized vs Implied volatility
# =============================================================================
vol_data = compute_realized_vs_implied(
    prices=prices, times=times,
    window_minutes=390,
    sigma_implied=SIGMA_IMPLIED
)
print(f"\n  Average Realized Vol  : {vol_data['avg_realized']*100:.1f}%")
print(f"  Implied Vol           : {vol_data['avg_implied']*100:.1f}%")
print(f"  Vol Premium (iv-rv)   : {vol_data['vol_premium']*100:.1f}%")


# =============================================================================
#  VISUALIZATIONS
# =============================================================================
print("\n" + "─" * 60)
print("  Generating Charts...")
print("─" * 60)


# ─────────────────────────────────────────────────────────────
# CHART 1: Stock Price Simulation + Greeks over time
# ─────────────────────────────────────────────────────────────
fig1, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig1.suptitle('Chart 1: Stock Price Simulation & Option Greeks', 
              fontsize=13, fontweight='bold', color='#E0F2F1', y=0.98)

# Panel A: Stock price
ax = axes[0]
ax.plot(times, prices, color=C_BLUE, linewidth=0.8, alpha=0.9, label='Stock Price (GBM)')
ax.axhline(K, color=C_RED, linestyle='--', alpha=0.7, linewidth=1, label=f'Strike K=${K}')
ax.fill_between(times, prices, K, where=(prices >= K), alpha=0.15, color=C_GREEN)
ax.fill_between(times, prices, K, where=(prices < K),  alpha=0.15, color=C_RED)
ax.set_ylabel('Stock Price ($)', fontsize=9)
ax.set_title(f'Simulated Price Path  (σ_realized={SIGMA_REALIZED*100:.0f}%,  σ_implied={SIGMA_IMPLIED*100:.0f}%)',
             fontsize=9, pad=4)
ax.legend(fontsize=8, loc='upper left')
ax.set_facecolor('#16202A')

# Panel B: Delta over hedge time
ht = hedge_results['hedge_times']
ax2 = axes[1]
ax2.plot(ht, hedge_results['delta_values'],  color=C_TEAL,  linewidth=1.5, label='Delta')
ax2.axhline(0.5, color=C_GRAY, linestyle=':', alpha=0.6, linewidth=1)
ax2.set_ylabel('Delta', fontsize=9)
ax2.set_title('Option Delta over Time  (ranges 0 → 1 as stock moves relative to strike)', fontsize=9, pad=4)
ax2.legend(fontsize=8)
ax2.set_ylim(0, 1)
ax2.set_facecolor('#16202A')

# Panel C: Gamma over hedge time
ax3 = axes[2]
ax3.plot(ht, hedge_results['gamma_values'], color=C_ORANGE, linewidth=1.5, label='Gamma')
ax3.set_ylabel('Gamma', fontsize=9)
ax3.set_xlabel('Time (years)', fontsize=9)
ax3.set_title('Option Gamma over Time  (gamma increases as stock nears strike near expiry)', fontsize=9, pad=4)
ax3.legend(fontsize=8)
ax3.set_facecolor('#16202A')

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart1_path = os.path.join(DATA_DIR, '01_price_and_greeks.png')
plt.savefig(chart1_path, dpi=130, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.close()
print(f"  Saved: {chart1_path}")


# ─────────────────────────────────────────────────────────────
# CHART 2: Delta Hedge Position & Trades
# ─────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig2.suptitle('Chart 2: Delta Hedge Position Over Time', 
              fontsize=13, fontweight='bold', color='#E0F2F1', y=0.98)

# Top: stock price with hedge adjustments marked
ax = axes2[0]
ax.plot(times, prices, color=C_BLUE, linewidth=0.7, alpha=0.8, zorder=1)
# Mark rebalance points
ax.scatter(ht, hedge_results['hedge_prices'],
           c=np.sign(hedge_results['shares_traded']),
           cmap='RdYlGn', s=15, alpha=0.7, zorder=2)
ax.set_ylabel('Stock Price ($)', fontsize=9)
ax.set_title('Stock Price with Hedge Rebalance Points  (green=buy shares, red=sell shares)', fontsize=9, pad=4)
ax.set_facecolor('#16202A')

# Bottom: hedge position (shares short)
ax2 = axes2[1]
ax2.step(ht, -hedge_results['hedge_position'], color=C_PURPLE, linewidth=1.2,
          label='Shares Short (positive = selling)')
ax2.fill_between(ht, -hedge_results['hedge_position'], 0,
                  alpha=0.2, color=C_PURPLE, step='pre')
ax2.axhline(0, color=C_GRAY, linestyle=':', alpha=0.6)
ax2.set_ylabel('Shares Short', fontsize=9)
ax2.set_xlabel('Time (years)', fontsize=9)
ax2.set_title('Hedge Position  (tracks –delta × 100 shares, rebalanced hourly)', fontsize=9, pad=4)
ax2.legend(fontsize=8)
ax2.set_facecolor('#16202A')

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart2_path = os.path.join(DATA_DIR, '02_delta_hedge_position.png')
plt.savefig(chart2_path, dpi=130, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print(f"  Saved: {chart2_path}")


# ─────────────────────────────────────────────────────────────
# CHART 3: Realized vs Implied Volatility
# ─────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig3.suptitle('Chart 3: Realized Volatility vs Implied Volatility',
              fontsize=13, fontweight='bold', color='#E0F2F1', y=0.98)

rv  = vol_data['realized_vol']
iv  = vol_data['implied_vol']
t   = vol_data['times']

ax = axes3[0]
ax.plot(t, rv * 100, color=C_ORANGE, linewidth=1.2, alpha=0.9, label='Realized Vol (rolling 1-day window)')
ax.plot(t, iv * 100, color=C_TEAL,   linewidth=1.5, linestyle='--', label=f'Implied Vol ({SIGMA_IMPLIED*100:.0f}%)')
ax.fill_between(t, rv * 100, iv * 100,
                where=(rv >= iv), alpha=0.2, color=C_GREEN, label='rv > iv  (gamma scalper profits)')
ax.fill_between(t, rv * 100, iv * 100,
                where=(rv < iv),  alpha=0.2, color=C_RED,   label='rv < iv  (gamma scalper loses)')
ax.set_ylabel('Volatility (%)', fontsize=9)
ax.set_title('Rolling Realized Volatility vs Implied Volatility  '
             '(green = profitable zone for gamma scalping)', fontsize=9, pad=4)
ax.legend(fontsize=8)
ax.set_facecolor('#16202A')

# Volatility premium (rv - iv) as a bar chart
ax2 = axes3[1]
vol_prem = rv - iv
valid = ~np.isnan(vol_prem)
bar_colors = [C_GREEN if x >= 0 else C_RED for x in vol_prem[valid]]
ax2.bar(t[valid], vol_prem[valid] * 100, color=bar_colors, width=0.0005, alpha=0.7)
ax2.axhline(0, color=C_GRAY, linestyle='-', alpha=0.8, linewidth=1)
ax2.set_ylabel('Vol Premium (rv - iv) %', fontsize=9)
ax2.set_xlabel('Time (years)', fontsize=9)
ax2.set_title('Volatility Premium  (positive = gamma scalping opportunity)', fontsize=9, pad=4)
ax2.set_facecolor('#16202A')

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart3_path = os.path.join(DATA_DIR, '03_realized_vs_implied_vol.png')
plt.savefig(chart3_path, dpi=130, bbox_inches='tight', facecolor=fig3.get_facecolor())
plt.close()
print(f"  Saved: {chart3_path}")


# ─────────────────────────────────────────────────────────────
# CHART 4: Cumulative P&L Breakdown
# ─────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 9))
fig4.suptitle('Chart 4: Cumulative P&L Breakdown (Hourly Hedging)',
              fontsize=13, fontweight='bold', color='#E0F2F1', y=0.98)

ht = hedge_results['hedge_times']

# Subplot 1: All P&L components
ax = axes4[0, 0]
ax.plot(ht, hedge_results['hedge_pnl'],          color=C_BLUE,   linewidth=1.5, label='Hedge P&L')
ax.plot(ht, hedge_results['option_pnl'],         color=C_TEAL,   linewidth=1.5, label='Option P&L')
ax.plot(ht, -hedge_results['transaction_costs'], color=C_RED,    linewidth=1.2, linestyle='--', label='–Transaction Costs')
ax.plot(ht, hedge_results['total_pnl'],          color=C_GREEN,  linewidth=2.0, label='Net P&L', zorder=5)
ax.axhline(0, color=C_GRAY, linestyle=':', alpha=0.6)
ax.set_title('P&L Components', fontsize=9, pad=4)
ax.set_ylabel('P&L ($)', fontsize=9)
ax.legend(fontsize=7)
ax.set_facecolor('#16202A')

# Subplot 2: Option value over time
ax2 = axes4[0, 1]
ax2.plot(ht, hedge_results['option_values'], color=C_ORANGE, linewidth=1.5, label='Option Price')
ax2.set_title('Option Price Decay (Theta)', fontsize=9, pad=4)
ax2.set_ylabel('Option Value ($)', fontsize=9)
ax2.legend(fontsize=7)
ax2.set_facecolor('#16202A')

# Subplot 3: Cumulative transaction costs
ax3 = axes4[1, 0]
ax3.plot(ht, hedge_results['transaction_costs'], color=C_RED, linewidth=1.5)
ax3.fill_between(ht, hedge_results['transaction_costs'], alpha=0.2, color=C_RED)
ax3.set_title('Cumulative Transaction Costs', fontsize=9, pad=4)
ax3.set_ylabel('Costs ($)', fontsize=9)
ax3.set_xlabel('Time (years)', fontsize=9)
ax3.set_facecolor('#16202A')

# Subplot 4: Shares traded per rebalance
ax4 = axes4[1, 1]
traded = hedge_results['shares_traded']
colors = [C_GREEN if t >= 0 else C_RED for t in traded]
ax4.bar(range(len(traded)), np.abs(traded), color=colors, alpha=0.7)
ax4.set_title('Shares Traded per Rebalance (abs value)', fontsize=9, pad=4)
ax4.set_ylabel('|Shares Traded|', fontsize=9)
ax4.set_xlabel('Rebalance #', fontsize=9)
ax4.set_facecolor('#16202A')

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart4_path = os.path.join(DATA_DIR, '04_cumulative_pnl.png')
plt.savefig(chart4_path, dpi=130, bbox_inches='tight', facecolor=fig4.get_facecolor())
plt.close()
print(f"  Saved: {chart4_path}")


# ─────────────────────────────────────────────────────────────
# CHART 5: Transaction Cost Impact (Frequency Experiment)
# ─────────────────────────────────────────────────────────────
fig5, axes5 = plt.subplots(1, 3, figsize=(15, 6))
fig5.suptitle('Chart 5: Impact of Hedge Frequency on P&L',
              fontsize=13, fontweight='bold', color='#E0F2F1', y=0.98)

freq_colors = {'Every Minute': C_ORANGE, 'Every Hour': C_TEAL, 'Every Day': C_BLUE}

# Subplot 1: Net P&L comparison
ax = axes5[0]
names  = list(freq_results.keys())
values = [freq_results[n]['final_total_pnl'] for n in names]
bar_c  = [C_GREEN if v > 0 else C_RED for v in values]
bars   = ax.bar(names, values, color=bar_c, edgecolor='#2D4356', linewidth=0.8)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.1 if val >= 0 else -1.5),
            f'${val:.2f}', ha='center', fontsize=8, color='white', fontweight='bold')
ax.axhline(0, color=C_GRAY, linestyle=':', alpha=0.7)
ax.set_title('Net P&L by Hedge Frequency', fontsize=9, pad=4)
ax.set_ylabel('Net P&L ($)', fontsize=9)
ax.set_facecolor('#16202A')

# Subplot 2: P&L breakdown stacked
ax2 = axes5[1]
x = np.arange(len(names))
w = 0.25
hedge_pnls  = [freq_results[n]['final_hedge_pnl']  for n in names]
option_pnls = [freq_results[n]['final_option_pnl'] for n in names]
costs       = [-freq_results[n]['total_costs']      for n in names]
ax2.bar(x - w, hedge_pnls,  width=w, color=C_BLUE,   alpha=0.85, label='Hedge P&L')
ax2.bar(x,     option_pnls, width=w, color=C_TEAL,   alpha=0.85, label='Option P&L')
ax2.bar(x + w, costs,       width=w, color=C_RED,    alpha=0.85, label='–Trans Costs')
ax2.axhline(0, color=C_GRAY, linestyle=':', alpha=0.7)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7)
ax2.set_title('P&L Component Breakdown', fontsize=9, pad=4)
ax2.set_ylabel('P&L ($)', fontsize=9)
ax2.legend(fontsize=7)
ax2.set_facecolor('#16202A')

# Subplot 3: Cumulative P&L paths
ax3 = axes5[2]
for name, color in freq_colors.items():
    r_data = freq_results[name]['hedge_results']
    ax3.plot(r_data['hedge_times'], r_data['total_pnl'],
              color=color, linewidth=1.5, label=name)
ax3.axhline(0, color=C_GRAY, linestyle=':', alpha=0.6)
ax3.set_title('Cumulative P&L Over Time', fontsize=9, pad=4)
ax3.set_ylabel('Net P&L ($)', fontsize=9)
ax3.set_xlabel('Time (years)', fontsize=9)
ax3.legend(fontsize=7)
ax3.set_facecolor('#16202A')

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart5_path = os.path.join(DATA_DIR, '05_transaction_cost_impact.png')
plt.savefig(chart5_path, dpi=130, bbox_inches='tight', facecolor=fig5.get_facecolor())
plt.close()
print(f"  Saved: {chart5_path}")


# ─────────────────────────────────────────────────────────────
# CHART 6: Scenario Analysis (High / Fair / Low Vol)
# ─────────────────────────────────────────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 6))
fig6.suptitle('Chart 6: Gamma Scalping P&L Under Different Volatility Scenarios',
              fontsize=13, fontweight='bold', color='#E0F2F1', y=0.98)

scen_colors = list(scen_results.keys())
for i, (scen_name, scen_data) in enumerate(scen_results.items()):
    ax = axes6[i]
    hr_data = scen_data['hedge_results']
    rv_val  = scen_data['sigma_realized'] * 100
    iv_val  = scen_data['sigma_implied']  * 100
    pnl_end = scen_data['final_pnl']
    color   = C_GREEN if pnl_end > 0 else C_RED

    ax.plot(hr_data['hedge_times'], hr_data['total_pnl'],
             color=color, linewidth=2, label=f'Net P&L: ${pnl_end:.2f}')
    ax.plot(hr_data['hedge_times'], hr_data['hedge_pnl'],
             color=C_BLUE, linewidth=1, linestyle='--', alpha=0.6, label='Hedge P&L')
    ax.fill_between(hr_data['hedge_times'], hr_data['total_pnl'], 0,
                     alpha=0.15, color=color)
    ax.axhline(0, color=C_GRAY, linestyle=':', alpha=0.7)

    ax.set_title(f'{scen_name}\nrv={rv_val:.0f}%  iv={iv_val:.0f}%',
                  fontsize=9, pad=4)
    ax.set_ylabel('P&L ($)', fontsize=9)
    ax.set_xlabel('Time (years)', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_facecolor('#16202A')

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart6_path = os.path.join(DATA_DIR, '06_scenario_analysis.png')
plt.savefig(chart6_path, dpi=130, bbox_inches='tight', facecolor=fig6.get_facecolor())
plt.close()
print(f"  Saved: {chart6_path}")


# ─────────────────────────────────────────────────────────────
# CHART 7: Summary Dashboard
# ─────────────────────────────────────────────────────────────
fig7 = plt.figure(figsize=(18, 12))
fig7.patch.set_facecolor('#0A1520')
fig7.suptitle('GAMMA SCALPING TRAINING SIMULATOR — SUMMARY DASHBOARD',
               fontsize=14, fontweight='bold', color='#00E5FF', y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig7, hspace=0.52, wspace=0.38)

# 1: Stock price (wide)
ax1 = fig7.add_subplot(gs[0, :2])
ax1.plot(times, prices, color=C_BLUE, linewidth=0.7, alpha=0.9)
ax1.axhline(K, color=C_RED, linestyle='--', linewidth=1, alpha=0.7)
ax1.set_title(f'Stock Price Path  (σ_realized={SIGMA_REALIZED*100:.0f}%)', fontsize=8, pad=3)
ax1.set_ylabel('Price ($)', fontsize=8)
ax1.set_facecolor('#16202A')
ax1.tick_params(labelsize=7)

# 2: Delta
ax2 = fig7.add_subplot(gs[0, 2])
ax2.plot(ht, hedge_results['delta_values'], color=C_TEAL, linewidth=1.2)
ax2.axhline(0.5, color=C_GRAY, linestyle=':', alpha=0.5)
ax2.set_title('Delta over Time', fontsize=8, pad=3)
ax2.set_facecolor('#16202A')
ax2.tick_params(labelsize=7)

# 3: Gamma
ax3 = fig7.add_subplot(gs[0, 3])
ax3.plot(ht, hedge_results['gamma_values'], color=C_ORANGE, linewidth=1.2)
ax3.set_title('Gamma over Time', fontsize=8, pad=3)
ax3.set_facecolor('#16202A')
ax3.tick_params(labelsize=7)

# 4: Realized vs Implied vol (wide)
ax4 = fig7.add_subplot(gs[1, :2])
ax4.plot(t, rv * 100, color=C_ORANGE, linewidth=1, alpha=0.9, label='Realized Vol')
ax4.plot(t, iv * 100, color=C_TEAL,   linewidth=1.5, linestyle='--', label='Implied Vol')
ax4.fill_between(t, rv*100, iv*100, where=(rv >= iv), alpha=0.2, color=C_GREEN)
ax4.fill_between(t, rv*100, iv*100, where=(rv < iv),  alpha=0.2, color=C_RED)
ax4.set_title('Realized vs Implied Volatility', fontsize=8, pad=3)
ax4.set_ylabel('Vol (%)', fontsize=8)
ax4.legend(fontsize=7, loc='upper right')
ax4.set_facecolor('#16202A')
ax4.tick_params(labelsize=7)

# 5: Net P&L by frequency (bar)
ax5 = fig7.add_subplot(gs[1, 2])
names_short = ['1min', '1hr', '1day']
vals = [freq_results[n]['final_total_pnl'] for n in freq_results.keys()]
colors_bar = [C_GREEN if v > 0 else C_RED for v in vals]
ax5.bar(names_short, vals, color=colors_bar, alpha=0.85)
ax5.axhline(0, color=C_GRAY, linestyle=':', alpha=0.7)
ax5.set_title('Net P&L by Frequency', fontsize=8, pad=3)
ax5.set_ylabel('$', fontsize=8)
ax5.set_facecolor('#16202A')
ax5.tick_params(labelsize=7)

# 6: Cumulative P&L (hourly hedge)
ax6 = fig7.add_subplot(gs[1, 3])
ax6.plot(ht, hedge_results['total_pnl'],  color=C_GREEN, linewidth=1.5, label='Net P&L')
ax6.plot(ht, hedge_results['hedge_pnl'],  color=C_BLUE,  linewidth=1,   label='Hedge', alpha=0.7)
ax6.axhline(0, color=C_GRAY, linestyle=':', alpha=0.6)
ax6.set_title('Cumul. P&L (Hourly)', fontsize=8, pad=3)
ax6.legend(fontsize=6)
ax6.set_facecolor('#16202A')
ax6.tick_params(labelsize=7)

# 7: Scenario comparison (wide, bottom)
ax7 = fig7.add_subplot(gs[2, :3])
scen_cols = [C_GREEN, C_YELLOW, C_RED]
for (scen_name, scen_data), sc in zip(scen_results.items(), scen_cols):
    hr_d = scen_data['hedge_results']
    rv_v = scen_data['sigma_realized']*100
    ax7.plot(hr_d['hedge_times'], hr_d['total_pnl'],
              color=sc, linewidth=1.5,
              label=f"rv={rv_v:.0f}% → ${scen_data['final_pnl']:.2f}")
ax7.axhline(0, color=C_GRAY, linestyle=':', alpha=0.7)
ax7.set_title('Scenario Analysis: P&L for Different Realized Volatilities', fontsize=8, pad=3)
ax7.set_ylabel('Net P&L ($)', fontsize=8)
ax7.set_xlabel('Time (years)', fontsize=8)
ax7.legend(fontsize=7)
ax7.set_facecolor('#16202A')
ax7.tick_params(labelsize=7)

# 8: Stats box
ax8 = fig7.add_subplot(gs[2, 3])
ax8.axis('off')
ax8.set_facecolor('#16202A')
summary_text = (
    f"SIMULATION SUMMARY\n"
    f"─────────────────────\n"
    f"Duration:    {T_DAYS} days\n"
    f"σ_realized:  {SIGMA_REALIZED*100:.0f}%\n"
    f"σ_implied:   {SIGMA_IMPLIED*100:.0f}%\n"
    f"Vol Premium: {(SIGMA_REALIZED-SIGMA_IMPLIED)*100:.0f}%\n\n"
    f"HOURLY HEDGE:\n"
    f"  Rebalances: {hedge_results['n_rebalances']}\n"
    f"  Hedge P&L:  ${hedge_results['hedge_pnl'][-1]:.2f}\n"
    f"  Option P&L: ${hedge_results['option_pnl'][-1]:.2f}\n"
    f"  Trans Cost: ${hedge_results['transaction_costs'][-1]:.2f}\n"
    f"  NET P&L:    ${hedge_results['total_pnl'][-1]:.2f}\n"
)
ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
          fontsize=7.5, verticalalignment='top',
          fontfamily='monospace', color='#B0BEC5',
          bbox=dict(facecolor='#1A2535', edgecolor='#2D4356', boxstyle='round,pad=0.5'))

plt.savefig(os.path.join(DATA_DIR, '07_dashboard.png'),
             dpi=130, bbox_inches='tight', facecolor=fig7.get_facecolor())
plt.close()
print(f"  Saved: {os.path.join(DATA_DIR, '07_dashboard.png')}")


# ─────────────────────────────────────────────────────────────
# SAVE SUMMARY DATA to CSV
# ─────────────────────────────────────────────────────────────
import pandas as pd

# Frequency experiment summary
freq_summary = []
for name, res in freq_results.items():
    freq_summary.append({
        'Frequency'     : name,
        'Rebalances'    : res['n_rebalances'],
        'Hedge_PnL'     : round(res['final_hedge_pnl'], 2),
        'Option_PnL'    : round(res['final_option_pnl'], 2),
        'Trans_Costs'   : round(res['total_costs'], 2),
        'Net_PnL'       : round(res['final_total_pnl'], 2),
    })
pd.DataFrame(freq_summary).to_csv(
    os.path.join(DATA_DIR, 'frequency_experiment_results.csv'), index=False)

# Scenario summary
scen_summary = []
for name, res in scen_results.items():
    scen_summary.append({
        'Scenario'       : name,
        'Realized_Vol'   : res['sigma_realized'],
        'Implied_Vol'    : res['sigma_implied'],
        'Vol_Premium'    : round(res['sigma_realized'] - res['sigma_implied'], 3),
        'Net_PnL'        : round(res['final_pnl'], 2),
        'Hedge_PnL'      : round(res['final_hedge_pnl'], 2),
        'Trans_Costs'    : round(res['total_costs'], 2),
    })
pd.DataFrame(scen_summary).to_csv(
    os.path.join(DATA_DIR, 'scenario_analysis_results.csv'), index=False)

print(f"  Saved: {os.path.join(DATA_DIR, 'frequency_experiment_results.csv')}")
print(f"  Saved: {os.path.join(DATA_DIR, 'scenario_analysis_results.csv')}")


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   SIMULATION COMPLETE")
print("=" * 60)
print(f"\n  Key Insight:")
print(f"  Realized vol ({SIGMA_REALIZED*100:.0f}%) > Implied vol ({SIGMA_IMPLIED*100:.0f}%)")
print(f"  -> Positive vol premium of {(SIGMA_REALIZED-SIGMA_IMPLIED)*100:.0f}%")
print(f"  -> Gamma scalping should generate positive P&L (in expectation)")
print(f"\n  Best hedge frequency: {'Every Hour'} (balances gamma capture vs costs)")
print(f"\n  Charts saved to: {DATA_DIR}/")
print("=" * 60)
