import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Simulate data (pre-generated for demonstration)
@st.cache_data
def generate_simulation_data():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'time': np.tile(np.arange(n // 10), 10),
        'spread': np.random.normal(2.5, 0.5, n),
        'volatility': np.random.normal(0.02, 0.005, n)
    })
    data['arbitrage'] = ((data['spread'] > 2.7) & (data['volatility'] < 0.025)).astype(int)
    return data

data = generate_simulation_data()

# KDE function with imshow and contour overlay
def kde_with_imshow_and_contours(x, y, title=""):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    coords = np.vstack([x, y])
    kde = gaussian_kde(coords)
    zi = kde(np.vstack([xi.ravel(), yi.ravel()]))

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        zi.reshape(xi.shape),
        origin='lower',
        aspect='auto',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='magma',
        alpha=0.6
    )

    ax.contour(
        xi, yi, zi.reshape(xi.shape),
        colors='black',
        linewidths=0.8,
        levels=5
    )

    ax.set_xlabel("Spread")
    ax.set_ylabel("Volatility")
    ax.set_title(title)
    ax.grid(True)
    fig.colorbar(im, ax=ax)
    return fig

# Streamlit interface
st.title("Dynamic Arbitrage Phase Space")
st.write("This app visualizes simulated arbitrage zones evolving through a spread-volatility phase space.")

t_start = st.slider("Select time window start", min_value=0, max_value=950, step=10, value=0)

window = data[(data['time'] >= t_start) & (data['time'] < t_start + 50)]

fig = kde_with_imshow_and_contours(
    window['spread'].values,
    window['volatility'].values,
    title=f"KDE with Contours\nTime = {t_start} to {t_start + 49}"
)

st.pyplot(fig)
