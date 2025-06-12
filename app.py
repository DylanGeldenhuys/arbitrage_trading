import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# --- Generate Simulation Data ---
def generate_data():
    np.random.seed(42)
    n = 1000
    time = np.arange(n)
    spread = np.random.normal(2.5, 1.0, size=n)
    volatility = np.random.normal(0.02, 0.01, size=n)
    liquidity = np.random.normal(1_000_000, 100_000, size=n)
    latency = np.random.normal(65, 5, size=n)
    
    arbitrage = ((spread > 2.0) & (liquidity > 1_000_000) & (latency < 60)).astype(int)

    return pd.DataFrame({
        'time': time,
        'spread': spread,
        'volatility': volatility,
        'liquidity': liquidity,
        'latency': latency,
        'arbitrage': arbitrage
    })

# Load data
data = generate_data()

# --- Page Setup ---
st.set_page_config(layout="wide")

# --- Sidebar Navigation ---
page = st.sidebar.radio("Choose View", ["Time-Evolving Phase Space", "Bifurcation Explorer", "Static Attractor Views"])

# --- 1. Time-Evolving Phase Space ---
if page == "Time-Evolving Phase Space":
    st.title("Time-Evolving Arbitrage Phase Space")
    t_start = st.slider("Start Time", min_value=0, max_value=950, step=10, value=0)

    window = data[(data['time'] >= t_start) & (data['time'] < t_start + 50)]
    arb = window[window['arbitrage'] == 1]
    no_arb = window[window['arbitrage'] == 0]

    # KDE using scipy
    x = window['spread']
    y = window['volatility']
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=z, cmap="Greys", s=40, alpha=0.6, edgecolor='')
    ax.scatter(no_arb['spread'], no_arb['volatility'], color='blue', alpha=0.4, s=30, label='No Arbitrage')
    ax.scatter(arb['spread'], arb['volatility'], color='red', alpha=0.8, s=40, label='Arbitrage')
    ax.set_title(f"Arbitrage Phase Space | Time {t_start}-{t_start+49}")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- 2. Bifurcation Explorer ---
elif page == "Bifurcation Explorer":
    st.title("Bifurcation Explorer")
    liquidity_mean = st.slider("Liquidity Mean", 800_000, 1_200_000, 950_000, step=10_000)
    latency_mean = st.slider("Latency Mean (ms)", 50, 80, 65, step=1)
    volatility_mean = st.slider("Volatility Mean", 0.005, 0.05, 0.02, step=0.001)

    def generate_ar_series(n, mean, phi, sigma):
        x = np.zeros(n)
        x[0] = mean
        for t in range(1, n):
            x[t] = mean + phi * (x[t-1] - mean) + np.random.normal(0, sigma)
        return x

    n = 500
    spread = generate_ar_series(n, mean=2.5, phi=0.9, sigma=1.0)
    liquidity = generate_ar_series(n, mean=liquidity_mean, phi=0.95, sigma=50_000)
    volatility = generate_ar_series(n, mean=volatility_mean, phi=0.95, sigma=0.005)
    latency = generate_ar_series(n, mean=latency_mean, phi=0.9, sigma=3)

    arbitrage = ((spread > 2.0) & (liquidity > 1_000_000) & (latency < 60)).astype(int)
    df = pd.DataFrame({
        'spread': spread,
        'volatility': volatility,
        'arbitrage': arbitrage
    })
    arb = df[df['arbitrage'] == 1]

    x = df['spread']
    y = df['volatility']
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=z, cmap="Greys", s=40, alpha=0.6, edgecolor='')
    ax.scatter(arb['spread'], arb['volatility'], color='red', alpha=0.8, s=40, label='Arbitrage')
    arb_rate = df['arbitrage'].mean() * 100
    ax.set_title(f"Bifurcation Phase Space\nLiquidity={int(liquidity_mean):,}, Latency={latency_mean} ms, Volatility={volatility_mean:.3f}\nArbitrage Rate: {arb_rate:.1f}%")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- 3. Static Attractor Views ---
elif page == "Static Attractor Views":
    st.title("Static Attractor Views")

    # KDE by Regime: Low vs High Liquidity
    low_liq = data[data['liquidity'] < 1_000_000]
    high_liq = data[data['liquidity'] >= 1_000_000]

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    sns.kdeplot(data=low_liq, x='spread', y='volatility', fill=True, cmap='Blues', ax=axes1[0], thresh=0.05)
    axes1[0].set_title("Low Liquidity Regime")
    axes1[0].set_xlabel("Spread")
    axes1[0].set_ylabel("Volatility")

    sns.kdeplot(data=high_liq, x='spread', y='volatility', fill=True, cmap='Reds', ax=axes1[1], thresh=0.05)
    axes1[1].set_title("High Liquidity Regime")
    axes1[1].set_xlabel("Spread")
    axes1[1].set_ylabel("Volatility")
    st.pyplot(fig1)

    # KDE Before vs After Shock (t = 500)
    data['regime'] = pd.cut(data['time'], bins=[0, 499, 550, 1000], labels=['pre-shock', 'shock', 'post-shock'])

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for i, label in enumerate(['pre-shock', 'shock', 'post-shock']):
        subset = data[data['regime'] == label]
        sns.kdeplot(data=subset, x='spread', y='volatility', fill=True, cmap='coolwarm', ax=axes2[i], thresh=0.05)
        axes2[i].set_title(f"{label.title()} Regime")
        axes2[i].set_xlabel("Spread")
        axes2[i].set_ylabel("Volatility")

    st.pyplot(fig2)
