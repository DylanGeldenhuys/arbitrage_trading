import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set Streamlit config
st.set_page_config(page_title="Arbitrage Regime Explorer", layout="wide")

# --- Simulate Market Data ---
np.random.seed(42)
def generate_ar_series(n, mean, phi, sigma):
    x = np.zeros(n)
    x[0] = mean
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + np.random.normal(0, sigma)
    return x

n = 1000
spread = generate_ar_series(n, mean=2.5, phi=0.9, sigma=1.0)
volatility = generate_ar_series(n, mean=0.02, phi=0.95, sigma=0.005)
liquidity = generate_ar_series(n, mean=950_000, phi=0.95, sigma=50_000)
time = np.arange(n)
arbitrage = ((spread > 2.0) & (liquidity > 1_000_000) & (np.random.rand(n) > 0.5)).astype(int)

# Assemble DataFrame
data = pd.DataFrame({
    'time': time,
    'spread': spread,
    'volatility': volatility,
    'liquidity': liquidity,
    'arbitrage': arbitrage
})
data['regime'] = pd.cut(data['time'], bins=[0, 499, 550, 1000], labels=['pre-shock', 'shock', 'post-shock'])

st.title("📉 Arbitrage Attractor Explorer")
st.markdown("""
Use KDE plots and simulated structural conditions to explore how arbitrage regimes emerge and change.
""")

# --- Interactive Time-KDE View ---
st.subheader("📊 Time-Windowed KDE with Arbitrage Overlay")
t_start = st.slider("Select Time Window Start", 0, 950, 0, step=10)

window = data[(data['time'] >= t_start) & (data['time'] < t_start + 50)]
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Perform KDE manually with scipy for better control
xy = np.vstack([window['spread'], window['volatility']])
kde = gaussian_kde(xy, bw_method=0.1)
xgrid = np.linspace(spread.min(), spread.max(), 100)
ygrid = np.linspace(volatility.min(), volatility.max(), 100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

ax1.contourf(X, Y, Z, levels=50, cmap="Greys", alpha=0.4)

# Overlay arbitrage states
arb = window[window['arbitrage'] == 1]
no_arb = window[window['arbitrage'] == 0]
ax1.scatter(no_arb['spread'], no_arb['volatility'], color='blue', alpha=0.4, s=20, label='No Arbitrage')
ax1.scatter(arb['spread'], arb['volatility'], color='red', alpha=0.8, s=30, label='Arbitrage')

ax1.set_title(f"Phase Space + Arbitrage Overlay (t={t_start}-{t_start+49})")
ax1.set_xlabel("Spread")
ax1.set_ylabel("Volatility")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# --- Bifurcation Regime Explorer ---
st.subheader("🌀 Bifurcation View by Structural Drivers")
liq_val = st.slider("Mean Liquidity", 800_000, 1_200_000, 950_000, step=10_000)
lat_val = st.slider("Mean Latency (ms)", 50, 80, 65)
vol_val = st.slider("Mean Volatility", 5, 50, 20, step=1)
vol_val = vol_val / 1000

def simulate_bifurcation(liquidity_mean, latency_mean, volatility_mean):
    n = 500
    spread = generate_ar_series(n, 2.5, 0.9, 1.0)
    liquidity = generate_ar_series(n, liquidity_mean, 0.95, 50_000)
    volatility = generate_ar_series(n, volatility_mean, 0.95, 0.005)
    latency = generate_ar_series(n, latency_mean, 0.9, 3)
    arbitrage = ((spread > 2.0) & (liquidity > 1_000_000) & (latency < 60)).astype(int)
    df = pd.DataFrame({"spread": spread, "volatility": volatility, "arbitrage": arbitrage})
    return df

bif_data = simulate_bifurcation(liq_val, lat_val, vol_val)
fig2, ax2 = plt.subplots(figsize=(8, 6))

xy_bif = np.vstack([bif_data['spread'], bif_data['volatility']])
kde_bif = gaussian_kde(xy_bif, bw_method=0.1)
xb = np.linspace(spread.min(), spread.max(), 100)
yb = np.linspace(volatility.min(), volatility.max(), 100)
XB, YB = np.meshgrid(xb, yb)
ZB = kde_bif(np.vstack([XB.ravel(), YB.ravel()])).reshape(XB.shape)

ax2.contourf(XB, YB, ZB, levels=50, cmap="Greys", alpha=0.4)
ax2.scatter(bif_data[bif_data['arbitrage'] == 1]['spread'], bif_data[bif_data['arbitrage'] == 1]['volatility'],
            color='red', alpha=0.7, s=30, label="Arbitrage")
ax2.set_title(f"Bifurcation View\nLiquidity = {liq_val:,}, Latency = {lat_val} ms, Volatility = {vol_val:.3f}")
ax2.set_xlabel("Spread")
ax2.set_ylabel("Volatility")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("Built by Dylan exploring market dynamics through the lens of phase transitions and structural conditions for arbitrage.")
