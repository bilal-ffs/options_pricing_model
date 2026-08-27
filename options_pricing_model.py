"""
================================================================================
PROJECT 1 — OPTIONS PRICING MODEL
================================================================================
  1. Black-Scholes pricing (calls & puts)
  2. All Greeks: Delta, Gamma, Theta, Vega, Rho
  3. Where BS breaks: Volatility Smile & Skew
  4. Fix: Heston Stochastic Volatility Model (Monte Carlo)
  5. Implied Volatility surface
  6. P&L simulation of options strategies
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

DARK   = "#0f172a"
CARD   = "#1e293b"
ACCENT = "#3b82f6"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
RED    = "#ef4444"
LIGHT  = "#e2e8f0"
MUTED  = "#64748b"
GRID   = "#334155"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BLACK-SCHOLES MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option price.
    S=spot, K=strike, T=time to expiry (years), r=risk-free rate, sigma=vol
    """
    if T <= 0: return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """All 5 Greeks."""
    if T <= 1e-6:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100   # per 1% vol move

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "theta": theta,
            "vega": vega,  "rho": rho}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IMPLIED VOLATILITY (where BS breaks)
# ═══════════════════════════════════════════════════════════════════════════════

def implied_vol(market_price, S, K, T, r, option_type="call"):
    """Recover implied vol from market price via Brent's method."""
    try:
        iv = brentq(
            lambda sig: bs_price(S, K, T, r, sig, option_type) - market_price,
            1e-6, 10.0, xtol=1e-6
        )
        return iv
    except:
        return np.nan


def simulate_vol_smile(S=100, T=0.25, r=0.02):
    """
    Simulate a realistic vol smile/skew.
    Real markets show higher IV for low strikes (put skew) — BS assumes flat vol.
    We model this with a quadratic + skew term.
    """
    strikes   = np.linspace(70, 130, 50)
    moneyness = np.log(strikes / S)

    # True implied vol surface (what market prices imply)
    # Left skew: OTM puts more expensive than BS predicts
    iv_market = 0.20 + 0.08 * moneyness**2 - 0.05 * moneyness

    # BS assumes flat vol (constant 20%)
    iv_bs_flat = np.full_like(strikes, 0.20)

    return strikes, iv_market, iv_bs_flat


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HESTON MODEL (Stochastic Volatility — fixes the smile)
# ═══════════════════════════════════════════════════════════════════════════════

def heston_monte_carlo(S0, K, T, r, v0, kappa, theta_v, xi, rho_sv,
                        n_paths=10_000, n_steps=252, option_type="call"):
    """
    Heston (1993) stochastic vol model.
    v0      = initial variance
    kappa   = mean-reversion speed of variance
    theta_v = long-run variance
    xi      = vol of vol
    rho_sv  = correlation between asset and variance Brownian motions
    """
    dt      = T / n_steps
    S       = np.full(n_paths, S0, dtype=float)
    v       = np.full(n_paths, v0, dtype=float)

    for _ in range(n_steps):
        z1 = np.random.standard_normal(n_paths)
        z2 = rho_sv * z1 + np.sqrt(1 - rho_sv**2) * np.random.standard_normal(n_paths)

        v_pos = np.maximum(v, 0)
        S    *= np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1)
        v    += kappa * (theta_v - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z2
        v     = np.maximum(v, 0)

    if option_type == "call":
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    se    = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_paths)
    return price, se


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OPTIONS STRATEGY P&L (Bull Call Spread + Protective Put)
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_pnl(S0=100, T=0.25, r=0.02, sigma=0.20):
    """Simulate P&L at expiry for common strategies."""
    spots = np.linspace(60, 140, 300)

    # Bull Call Spread: Buy 100C, Sell 110C
    K1, K2 = 100, 110
    cost_spread = bs_price(S0, K1, T, r, sigma, "call") - \
                  bs_price(S0, K2, T, r, sigma, "call")
    pnl_spread  = (np.maximum(spots - K1, 0) - np.maximum(spots - K2, 0)) - cost_spread

    # Protective Put: Long stock + Buy 95P
    K_put = 95
    cost_pp  = bs_price(S0, K_put, T, r, sigma, "put")
    pnl_pp   = (spots - S0) + np.maximum(K_put - spots, 0) - cost_pp

    # Straddle: Buy ATM Call + Put
    cost_straddle = (bs_price(S0, S0, T, r, sigma, "call") +
                     bs_price(S0, S0, T, r, sigma, "put"))
    pnl_straddle  = (np.maximum(spots - S0, 0) +
                     np.maximum(S0 - spots, 0)) - cost_straddle

    return spots, pnl_spread, pnl_pp, pnl_straddle


# ═══════════════════════════════════════════════════════════════════════════════
# 5. IV SURFACE (3D)
# ═══════════════════════════════════════════════════════════════════════════════

def iv_surface(S=100, r=0.02):
    """Generate a realistic implied vol surface across strikes and maturities."""
    strikes     = np.linspace(80, 120, 20)
    maturities  = np.linspace(0.05, 2.0, 15)
    K_grid, T_grid = np.meshgrid(strikes, maturities)

    moneyness = np.log(K_grid / S)
    # Vol smile steeper at short maturities, flatter at long maturities
    iv = 0.20 + (0.08 * moneyness**2 - 0.05 * moneyness) / np.sqrt(T_grid)
    iv = np.clip(iv, 0.05, 0.80)
    return K_grid, T_grid, iv


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RUN HESTON vs BS COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_comparison(S=100, T=0.25, r=0.02):
    """Compare BS vs Heston across strike range."""
    strikes = np.arange(80, 122, 2)
    # Heston params calibrated to show smile
    params  = dict(v0=0.04, kappa=2.0, theta_v=0.04, xi=0.5, rho_sv=-0.7)

    bs_prices, h_prices, bs_ivs, h_ivs = [], [], [], []

    for K in strikes:
        bp = bs_price(S, K, T, r, 0.20, "call")
        hp, _ = heston_monte_carlo(S, K, T, r, n_paths=5000, n_steps=63,
                                   option_type="call", **params)
        bs_prices.append(bp)
        h_prices.append(hp)

        biv = 0.20
        hiv = implied_vol(hp, S, K, T, r, "call")
        bs_ivs.append(biv)
        h_ivs.append(hiv if hiv else np.nan)

    return strikes, np.array(bs_prices), np.array(h_prices), \
           np.array(bs_ivs),   np.array(h_ivs)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MASTER TEARSHEET
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tearsheet(save_path="/home/claude/tearsheet_options.png"):
    fig = plt.figure(figsize=(22, 26), facecolor=DARK)
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.55, wspace=0.38,
                            top=0.93, bottom=0.04,
                            left=0.06, right=0.97)

    def sa(ax):
        ax.set_facecolor(CARD); ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.4, ls="--", alpha=0.5)
        for sp in ax.spines.values(): sp.set_linewidth(0.6)

    tk = dict(color=LIGHT, fontweight="bold", fontsize=10)

    fig.text(0.5, 0.965, "PROJECT 1 — OPTIONS PRICING MODEL",
             ha="center", fontsize=18, fontweight="bold",
             color="#f8fafc", fontfamily="monospace")
    fig.text(0.5, 0.950,
             "Black-Scholes  |  Greeks  |  Volatility Smile  |  Heston Stochastic Vol  |  Strategy P&L",
             ha="center", fontsize=10, color=MUTED, fontfamily="monospace")

    # ── 1. Option price vs spot ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sa(ax1); ax1.set_title("Option Price vs Spot", **tk)
    spots  = np.linspace(60, 140, 200)
    S0, K, T_val, r_val, sig = 100, 100, 0.25, 0.02, 0.20
    calls  = [bs_price(s, K, T_val, r_val, sig, "call") for s in spots]
    puts   = [bs_price(s, K, T_val, r_val, sig, "put")  for s in spots]
    intrinsic_c = np.maximum(spots - K, 0)
    intrinsic_p = np.maximum(K - spots, 0)
    ax1.plot(spots, calls, color=ACCENT, lw=2, label="Call (BS)")
    ax1.plot(spots, puts,  color=RED,    lw=2, label="Put (BS)")
    ax1.plot(spots, intrinsic_c, color=ACCENT, lw=1, ls="--", alpha=0.5, label="Intrinsic Call")
    ax1.plot(spots, intrinsic_p, color=RED,    lw=1, ls="--", alpha=0.5, label="Intrinsic Put")
    ax1.axvline(K, color=AMBER, lw=1, ls=":", alpha=0.8)
    ax1.set_xlabel("Spot Price ($)", color=MUTED)
    ax1.set_ylabel("Option Price ($)", color=MUTED)
    ax1.legend(fontsize=7, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 2. Greeks vs Spot ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sa(ax2); ax2.set_title("Call Greeks vs Spot", **tk)
    deltas = [bs_greeks(s, K, T_val, r_val, sig, "call")["delta"] for s in spots]
    gammas = [bs_greeks(s, K, T_val, r_val, sig, "call")["gamma"] * 10 for s in spots]
    ax2.plot(spots, deltas, color=GREEN, lw=2, label="Delta")
    ax2.plot(spots, gammas, color=AMBER, lw=2, label="Gamma ×10")
    ax2.axvline(K, color=MUTED, lw=1, ls=":")
    ax2.set_xlabel("Spot Price ($)", color=MUTED)
    ax2.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 3. Theta decay ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sa(ax3); ax3.set_title("Theta Decay (Time Value Erosion)", **tk)
    times = np.linspace(0.01, 1.0, 200)
    for K_t, col, lbl in [(90, GREEN, "ITM K=90"),
                           (100, ACCENT, "ATM K=100"),
                           (110, RED,   "OTM K=110")]:
        thetas = [bs_greeks(S0, K_t, t, r_val, sig, "call")["theta"] for t in times]
        ax3.plot(times[::-1], thetas, color=col, lw=2, label=lbl)
    ax3.axvline(0, color=MUTED, lw=0.8)
    ax3.set_xlabel("Days to Expiry", color=MUTED)
    ax3.set_ylabel("Theta ($/day)", color=MUTED)
    ax3.set_xticklabels([f"{int(t*365)}" for t in ax3.get_xticks()])
    ax3.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 4. Volatility Smile (BS failure) ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    sa(ax4); ax4.set_title("Volatility Smile — Where BS Fails", **tk)
    strikes_sm, iv_mkt, iv_bs = simulate_vol_smile()
    ax4.plot(strikes_sm, iv_mkt * 100, color=ACCENT, lw=2.5, label="Market IV (smile/skew)")
    ax4.plot(strikes_sm, iv_bs  * 100, color=RED,    lw=2,   ls="--", label="BS assumption (flat)")
    ax4.fill_between(strikes_sm, iv_bs*100, iv_mkt*100,
                     color=ACCENT, alpha=0.15, label="Model error")
    ax4.set_xlabel("Strike Price ($)", color=MUTED)
    ax4.set_ylabel("Implied Volatility (%)", color=MUTED)
    ax4.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 5. BS vs Heston Prices ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sa(ax5); ax5.set_title("Black-Scholes vs Heston Pricing", **tk)
    print("  Running Heston MC (this takes ~15s)...")
    strikes_cmp, bs_p, h_p, bs_iv, h_iv = run_model_comparison()
    ax5.plot(strikes_cmp, bs_p, color=RED,   lw=2, marker="o", ms=4, label="Black-Scholes")
    ax5.plot(strikes_cmp, h_p,  color=GREEN, lw=2, marker="s", ms=4, label="Heston MC")
    ax5.fill_between(strikes_cmp, bs_p, h_p, alpha=0.2, color=AMBER, label="Pricing diff")
    ax5.set_xlabel("Strike ($)", color=MUTED)
    ax5.set_ylabel("Call Price ($)", color=MUTED)
    ax5.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 6. Heston Implied Vol Smile ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    sa(ax6); ax6.set_title("Heston Recovers the Vol Smile", **tk)
    ax6.plot(strikes_cmp, bs_iv * 100,
             color=RED,   lw=2, ls="--", label="BS (flat 20%)")
    valid = ~np.isnan(h_iv)
    ax6.plot(strikes_cmp[valid], h_iv[valid] * 100,
             color=GREEN, lw=2.5, marker="s", ms=4, label="Heston implied IV")
    ax6.set_xlabel("Strike ($)", color=MUTED)
    ax6.set_ylabel("Implied Vol (%)", color=MUTED)
    ax6.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 7. Strategy P&L ─────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :])
    sa(ax7); ax7.set_title("Options Strategy P&L at Expiry", **tk)
    spots_pl, pnl_spread, pnl_pp, pnl_straddle = strategy_pnl()
    ax7.plot(spots_pl, pnl_spread,   color=ACCENT, lw=2.5, label="Bull Call Spread (Buy 100C / Sell 110C)")
    ax7.plot(spots_pl, pnl_pp,       color=GREEN,  lw=2.5, label="Protective Put (Long Stock + Buy 95P)")
    ax7.plot(spots_pl, pnl_straddle, color=AMBER,  lw=2.5, label="Straddle (Buy ATM Call + Put)")
    ax7.axhline(0, color=MUTED, lw=1, ls="--")
    ax7.axvline(100, color=MUTED, lw=1, ls=":", alpha=0.5)
    ax7.fill_between(spots_pl, pnl_spread,   0, where=pnl_spread   > 0, alpha=0.12, color=ACCENT)
    ax7.fill_between(spots_pl, pnl_straddle, 0, where=pnl_straddle > 0, alpha=0.12, color=AMBER)
    ax7.set_xlabel("Spot Price at Expiry ($)", color=MUTED)
    ax7.set_ylabel("P&L ($)", color=MUTED)
    ax7.legend(fontsize=9, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 8. Metrics table ────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, :])
    ax8.set_facecolor(CARD); ax8.axis("off")
    ax8.set_title("Black-Scholes Greeks Summary  (S=100, K=100, T=0.25yr, r=2%, σ=20%)",
                  color=LIGHT, fontweight="bold", fontsize=10)

    greek_data = []
    for opt in ["call", "put"]:
        g = bs_greeks(100, 100, 0.25, 0.02, 0.20, opt)
        p = bs_price(100, 100, 0.25, 0.02, 0.20, opt)
        greek_data.append([
            opt.upper(),
            f"${p:.4f}",
            f"{g['delta']:.4f}",
            f"{g['gamma']:.4f}",
            f"${g['theta']:.4f}/day",
            f"${g['vega']:.4f}/1%vol",
            f"${g['rho']:.4f}/1%r"
        ])

    cols = ["Type","Price","Delta (Δ)","Gamma (Γ)","Theta (Θ)","Vega (ν)","Rho (ρ)"]
    tbl  = ax8.table(cellText=greek_data, colLabels=cols,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
    for (r2, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r2 == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="#93c5fd", fontweight="bold")
        else:
            cell.set_facecolor("#1e293b" if r2 % 2 == 0 else "#172032")
            cell.set_text_props(color=LIGHT)

    plt.savefig(save_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] Options tearsheet → {save_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PROJECT 1: OPTIONS PRICING MODEL")
    print("="*60)
    print(f"  BS Call Price (S=100,K=100,T=0.25,r=2%,σ=20%): "
          f"${bs_price(100,100,0.25,0.02,0.20,'call'):.4f}")
    print(f"  BS Put  Price:  ${bs_price(100,100,0.25,0.02,0.20,'put'):.4f}")
    g = bs_greeks(100,100,0.25,0.02,0.20,"call")
    print(f"  Call Greeks → Delta:{g['delta']:.3f}  Gamma:{g['gamma']:.4f}  "
          f"Theta:{g['theta']:.4f}  Vega:{g['vega']:.4f}")
    print(f"  Put-Call Parity check: "
          f"{abs(bs_price(100,100,0.25,0.02,0.20,'call') - bs_price(100,100,0.25,0.02,0.20,'put') - 100 + 100*np.exp(-0.02*0.25)):.6f} (should be ~0)")
    print("\n  Running Heston Monte Carlo & building tearsheet...")
    plot_tearsheet()
