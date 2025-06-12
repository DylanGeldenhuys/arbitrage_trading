import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Utilities ---
def generate_ar_series(n, mean, phi, sigma):
    x = np.zeros(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = mean + phi * (x[t-1] - mean) + np.random.normal(0, sigma)
    return x

# --- Shared Params ---
spread_thresh = 2.0
liq_thresh = 1_000_000
lat_thresh = 60

# --- Sidebar ---
st.sidebar.title("Simulation Controls")
view = st.sidebar.selectbox("Choose View", ["Phase Space Evolution", "Bifurcation Explorer"])

# --- Phase Space Evolution ---
if view == "Phase Space Evolution":
    st.title("Phase Space: Trajectory + Arbitrage Zones")

    np.random.seed(42)
    n = 1000
    spread = generate_ar_series(n, mean=2.5, phi=0.9, sigma=1.0)
    volatility = generate_ar_series(n, mean=0.02, phi=0.9, sigma=0.005)
    time = np.arange(n)
    arbitrage = ((spread > spread_thresh) & (volatility < 0.03)).astype(int)

    data = pd.DataFrame({
        'spread': spread,
        'volatility': volatility,
        'time': time,
        'arbitrage': arbitrage
    })

    t_start = st.slider("Time Window Start", 0, n - 50, 0, step=10)
    window = data[(data['time'] >= t_start) & (data['time'] < t_start + 50)]
    arb = window[window['arbitrage'] == 1]
    no_arb = window[window['arbitrage'] == 0]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Global KDE background
    sns.kdeplot(
        data=data, x='spread', y='volatility', fill=True,
        cmap='Greys', thresh=0.05, alpha=0.3, ax=ax, bw_adjust=1.2
    )

    # Overlay local window trajectory
    ax.plot(window['spread'], window['volatility'], color='black', alpha=0.4, linestyle='--', linewidth=1, label='Trajectory')
    ax.scatter(no_arb['spread'], no_arb['volatility'], color='blue', alpha=0.5, s=30, label='No Arbitrage')
    ax.scatter(arb['spread'], arb['volatility'], color='red', alpha=0.8, s=40, label='Arbitrage')

    ax.set_title(f"Phase Space: Time {t_start} to {t_start + 49}")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Volatility")
    ax.set_xlim(data['spread'].min() - 0.5, data['spread'].max() + 0.5)
    ax.set_ylim(data['volatility'].min() - 0.01, data['volatility'].max() + 0.01)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- Bifurcation Explorer ---
elif view == "Bifurcation Explorer":
    st.title("Bifurcation: Arbitrage Emergence via Structural Inputs")

    liquidity_mean = st.slider("Liquidity Mean", 800_000, 1_200_000, 950_000, step=10_000)
    latency_mean = st.slider("Latency Mean (ms)", 50, 80, 65, step=1)
    volatility_mean = st.slider("Volatility Mean", 0.005, 0.05, 0.02, step=0.001)

    n = 500
    spread = generate_ar_series(n, mean=2.5, phi=0.9, sigma=1.0)
    liquidity = generate_ar_series(n, mean=liquidity_mean, phi=0.95, sigma=50_000)
    volatility = generate_ar_series(n, mean=volatility_mean, phi=0.95, sigma=0.005)
    latency = generate_ar_series(n, mean=latency_mean, phi=0.9, sigma=3)

    arbitrage = ((spread > spread_thresh) & (liquidity > liq_thresh) & (latency < lat_thresh)).astype(int)
    df = pd.DataFrame({
        'spread': spread,
        'volatility': volatility,
        'arbitrage': arbitrage
    })

    arb = df[df['arbitrage'] == 1]
    spread_min, spread_max = df['spread'].min(), df['spread'].max()
    vol_min, vol_max = df['volatility'].min(), df['volatility'].max()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(
        data=df, x='spread', y='volatility', fill=True,
        cmap='Greys', thresh=0.05, alpha=0.4,
        ax=ax, clip=((spread_min, spread_max), (vol_min, vol_max)), bw_adjust=1.5
    )
    ax.scatter(arb['spread'], arb['volatility'], color='red', s=40, alpha=0.8, label='Arbitrage')

    arb_rate = df['arbitrage'].mean() * 100
    ax.set_title(f"Bifurcation View\nLiquidity = {int(liquidity_mean):,}, Latency = {int(latency_mean)} ms, Volatility = {volatility_mean:.3f} | Arbitrage Rate: {arb_rate:.1f}%")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Volatility")
    ax.set_xlim(spread_min - 0.5, spread_max + 0.5)
    ax.set_ylim(vol_min - 0.01, vol_max + 0.01)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)