import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import gaussian_kde

# --- Simulated Data Generator ---
def generate_data(n=1000):
    np.random.seed(42)
    time = np.arange(n)
    spread = np.random.normal(2.5, 0.7, size=n)
    volatility = np.random.normal(0.02, 0.01, size=n)
    liquidity = np.random.normal(950_000, 100_000, size=n)
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

# --- KDE Plot Function ---
def plot_kde_frame(window):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = window['spread']
    y = window['volatility']

    # Density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    ax.scatter(x, y, c=z, cmap="Greys", s=40, alpha=0.6, edgecolor='none')

    # Arbitrage Overlay
    arb = window[window['arbitrage'] == 1]
    ax.scatter(arb['spread'], arb['volatility'], color='red', alpha=0.8, s=40, label='Arbitrage')

    ax.set_title("Phase Space (KDE + Arbitrage Overlay)")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Bifurcation KDE View ---
def generate_ar_series(n, mean, phi, sigma):
    x = np.zeros(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = mean + phi * (x[t-1] - mean) + np.random.normal(0, sigma)
    return x

def plot_bifurcation_kde(liquidity_mean, latency_mean, volatility_mean):
    n = 500
    spread = generate_ar_series(n, mean=2.5, phi=0.9, sigma=1.0)
    liquidity = generate_ar_series(n, mean=liquidity_mean, phi=0.95, sigma=50_000)
    volatility = generate_ar_series(n, mean=volatility_mean, phi=0.95, sigma=0.005)
    latency = generate_ar_series(n, mean=latency_mean, phi=0.9, sigma=3)

    arbitrage = ((spread > 2.0) & (liquidity > 1_000_000) & (latency < 60)).astype(int)
    df = pd.DataFrame({'spread': spread, 'volatility': volatility, 'arbitrage': arbitrage})

    fig, ax = plt.subplots(figsize=(8, 6))
    xy = np.vstack([df['spread'], df['volatility']])
    z = gaussian_kde(xy)(xy)

    ax.scatter(df['spread'], df['volatility'], c=z, cmap="Greys", s=40, alpha=0.6, edgecolor='none')
    ax.scatter(df[df['arbitrage'] == 1]['spread'], df[df['arbitrage'] == 1]['volatility'], color='red', s=40, alpha=0.8, label='Arbitrage')
    arb_rate = df['arbitrage'].mean() * 100

    ax.set_title(f"Bifurcation View\nLiquidity = {int(liquidity_mean):,}, Latency = {int(latency_mean)} ms, Volatility = {volatility_mean:.3f}\nArbitrage Rate: {arb_rate:.1f}%")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Volatility")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Arbitrage Regime Simulation Viewer")

# Sidebar Navigation
mode = st.sidebar.radio("Choose Simulation View", ["Time-Evolving Phase Space", "Bifurcation Explorer", "Static KDE Regimes"])

# Load data once
@st.cache_data
def load_data():
    return generate_data(n=1000)

data = load_data()

if mode == "Time-Evolving Phase Space":
    t = st.slider("Start Time", min_value=0, max_value=950, step=10, value=0)
    window = data[(data['time'] >= t) & (data['time'] < t + 50)]
    plot_kde_frame(window)

elif mode == "Bifurcation Explorer":
    lq = st.slider("Mean Liquidity", min_value=800_000, max_value=1_200_000, step=10_000, value=950_000)
    lt = st.slider("Mean Latency", min_value=50, max_value=80, step=1, value=65)
    vol = st.slider("Mean Volatility", min_value=0.005, max_value=0.05, step=0.001, value=0.02)
    plot_bifurcation_kde(lq, lt, vol)

elif mode == "Static KDE Regimes":
    st.subheader("Liquidity Regime Comparison")
    low_liq = data[data['liquidity'] < 1_000_000]
    high_liq = data[data['liquidity'] >= 1_000_000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.kdeplot(data=low_liq, x='spread', y='volatility', fill=True, cmap='Blues', ax=axes[0], thresh=0.05)
    axes[0].set_title("Low Liquidity Regime")
    axes[0].set_xlabel("Spread")
    axes[0].set_ylabel("Volatility")

    sns.kdeplot(data=high_liq, x='spread', y='volatility', fill=True, cmap='Reds', ax=axes[1], thresh=0.05)
    axes[1].set_title("High Liquidity Regime")
    axes[1].set_xlabel("Spread")
    axes[1].set_ylabel("Volatility")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Shock Regime Comparison")
    data['regime'] = pd.cut(data['time'], bins=[0, 499, 550, 1000], labels=['pre-shock', 'shock', 'post-shock'])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, label in enumerate(['pre-shock', 'shock', 'post-shock']):
        subset = data[data['regime'] == label]
        sns.kdeplot(data=subset, x='spread', y='volatility', fill=True, cmap='coolwarm', ax=axes[i], thresh=0.05)
        axes[i].set_title(f"{label.title()} Regime")
        axes[i].set_xlabel("Spread")
        axes[i].set_ylabel("Volatility")

    plt.tight_layout()
    st.pyplot(fig)
