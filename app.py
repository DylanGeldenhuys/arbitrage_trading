import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Market Regime Attractors", layout="wide")

# --- Simulate Market Data ---
np.random.seed(42)

def generate_ar_series(n, mean, phi, sigma):
    x = np.zeros(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = mean + phi * (x[t-1] - mean) + np.random.normal(0, sigma)
    return x

n = 1000
spread = generate_ar_series(n, mean=2.5, phi=0.9, sigma=1.0)
volatility = generate_ar_series(n, mean=0.02, phi=0.95, sigma=0.005)
liquidity = generate_ar_series(n, mean=950_000, phi=0.95, sigma=50_000)
time = np.arange(n)
arbitrage = ((spread > 2.0) & (liquidity > 1_000_000) & (np.random.rand(n) > 0.5)).astype(int)

data = pd.DataFrame({
    'time': time,
    'spread': spread,
    'volatility': volatility,
    'liquidity': liquidity,
    'arbitrage': arbitrage
})

# Label time-based regimes
data['regime'] = pd.cut(data['time'], bins=[0, 499, 550, 1000], labels=['pre-shock', 'shock', 'post-shock'])

# --- Streamlit App ---
st.title("📈 Market Regime Attractors")
st.markdown("""
Explore how arbitrage emerges under different structural market conditions using KDE attractor plots.  
We simulate how market variables evolve and observe density shifts across:
- **Liquidity Regimes** (Low vs High)
- **Temporal Regimes** (Pre-shock, Shock, Post-shock)
""")

# --- KDE by Liquidity Regime ---
st.subheader("Attractor Density by Liquidity Regime")

low_liq = data[data['liquidity'] < 1_000_000]
high_liq = data[data['liquidity'] >= 1_000_000]

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
sns.kdeplot(data=low_liq, x='spread', y='volatility', fill=True, cmap='Blues', ax=axes1[0], thresh=0.02)
axes1[0].set_title("Low Liquidity Regime")
axes1[0].set_xlabel("Spread")
axes1[0].set_ylabel("Volatility")

sns.kdeplot(data=high_liq, x='spread', y='volatility', fill=True, cmap='Reds', ax=axes1[1], thresh=0.02)
axes1[1].set_title("High Liquidity Regime")
axes1[1].set_xlabel("Spread")
axes1[1].set_ylabel("Volatility")

plt.tight_layout()
st.pyplot(fig1)

# --- KDE Over Shock Regimes ---
st.subheader("Attractor Evolution Across Shock Regimes")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
for i, label in enumerate(['pre-shock', 'shock', 'post-shock']):
    subset = data[data['regime'] == label]
    sns.kdeplot(data=subset, x='spread', y='volatility', fill=True, cmap='coolwarm', ax=axes2[i], thresh=0.02)
    axes2[i].set_title(f"{label.title()} Regime")
    axes2[i].set_xlabel("Spread")
    axes2[i].set_ylabel("Volatility")

plt.tight_layout()
st.pyplot(fig2)
