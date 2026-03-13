# Gamma Scalping Training Simulator


---

## What Is Gamma Scalping?

**Gamma scalping** is an options trading strategy where a trader:

1. **Buys an option** (which gives them positive gamma exposure)
2. **Delta-hedges** the position — continuously buying/selling the underlying stock to stay "delta neutral"
3. **Profits from volatility** — each time the stock moves and they rebalance, they lock in a small gain

The core insight:
> If the stock moves MORE than the option priced in (realized vol > implied vol),
> gamma scalping is profitable. If it moves LESS, you lose money.

---

## How Delta Hedging Works

**Delta** measures how much an option price changes per $1 move in the stock.

- If delta = 0.60, and you own 1 call option (= 100 shares equivalent)
- You SHORT 60 shares to be delta-neutral (total delta = 0)
- When the stock moves up $1 → delta rises to e.g. 0.64 (gamma kicked in)
- You SELL 4 more shares to stay hedged
- That rebalance locks in a tiny profit

Repeat this hundreds of times and it adds up to a meaningful P&L.

---

## What Insights the Simulator Demonstrates

| Insight | What You'll See |
|---------|----------------|
| **Vol Premium** | When realized vol > implied vol, gamma scalping profits |
| **Theta cost** | Owning options costs you time decay every day (the break-even hurdle) |
| **Optimal frequency** | More frequent hedging captures more gamma BUT incurs more costs |
| **Transaction costs** | High-frequency hedging (every minute) can be eaten alive by costs |
| **Scenario analysis** | High/fair/low vol scenarios show how conditions affect P&L |

---

## Project Structure

```
gamma-scalping-simulator/
│
├── src/
│   ├── black_scholes.py     ← Option pricing + Greeks (Delta, Gamma, Vega, Theta)
│   ├── price_simulation.py  ← GBM stock price simulator
│   ├── delta_hedging.py     ← Delta hedging engine with P&L tracking
│   └── pnl_analysis.py      ← Gamma scalping theory, frequency experiment, scenarios
│
├── dashboard/
│   └── run_simulator.py     ← Main entry point — runs everything + generates charts
│
├── data/                    ← Generated charts and CSV results (auto-created)
│
├── notebooks/               ← Place for Jupyter notebooks (optional exploration)
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Option A: Google Colab (No Installation)
1. Upload all files to a Colab notebook
2. Run: `!python dashboard/run_simulator.py`

### Option B: Local Python
```bash
# Install dependencies
pip install numpy scipy matplotlib pandas

# Run the simulator
python dashboard/run_simulator.py
```

---

## Output Charts

| Chart | What It Shows |
|-------|--------------|
| `01_price_and_greeks.png` | GBM stock price simulation + Delta and Gamma over time |
| `02_delta_hedge_position.png` | Hedge position (shares short) and rebalance events |
| `03_realized_vs_implied_vol.png` | Rolling realized vol vs constant implied vol |
| `04_cumulative_pnl.png` | P&L breakdown: hedge gains, option gains, transaction costs |
| `05_transaction_cost_impact.png` | Effect of hedge frequency (every min / hr / day) on net P&L |
| `06_scenario_analysis.png` | High vol / fair vol / low vol scenario comparison |
| `07_dashboard.png` | All-in-one summary dashboard |

---

## Key Concepts Explained

### Black-Scholes Model
The industry-standard formula for pricing European options.
Uses 5 inputs: Stock Price, Strike, Time to Expiry, Risk-Free Rate, Implied Volatility.

### The Greeks
- **Delta** (Δ): Option sensitivity to stock price moves (0 to 1 for calls)
- **Gamma** (Γ): Rate of change of delta per $1 stock move (highest near ATM)
- **Theta** (Θ): Daily time decay — the cost of holding the option
- **Vega** (ν): Sensitivity to changes in implied volatility

### Geometric Brownian Motion (GBM)
The mathematical model for stock price movements used in Black-Scholes.
Each step: S(t+dt) = S(t) × exp((μ - σ²/2)dt + σ√dt × Z)

### The Gamma Scalping Formula
Per-step theoretical P&L ≈ ½ × Γ × S² × (σ_realized² − σ_implied²) × dt

This formula shows:
- Profit when `σ_realized > σ_implied`
- Loss when `σ_realized < σ_implied`
- Bigger gamma → bigger moves per rebalance

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical operations, GBM simulation |
| `scipy` | Normal distribution functions for Black-Scholes |
| `matplotlib` | All charts and visualizations |
| `pandas` | Saving results to CSV |
