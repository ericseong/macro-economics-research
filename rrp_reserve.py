import pandas as pd
from fredapi import Fred
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Load API key from external file
# ---------------------------------------------------------
with open("credentials/credential_fred_api.txt", "r") as f:
    API_KEY = f.read().strip()

fred = Fred(api_key=API_KEY)

# ---------------------------------------------------------
# 2. Download data from FRED
# ---------------------------------------------------------
rrp = fred.get_series("RRPONTSYD")     # Reverse Repo
reserves = fred.get_series("WRESBAL")  # Reserves (at Fed)

# ---------------------------------------------------------
# 3. Create DataFrame
# ---------------------------------------------------------
df = pd.DataFrame({
    "RRP": rrp,
    "Reserves": reserves
}).dropna()

# ---------------------------------------------------------
# 4. Plotly Dual-Axis Chart
# ---------------------------------------------------------
fig = go.Figure()

# Left y-axis: Reverse Repo
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["RRP"] / 1e12,  # convert to Trillion USD
    name="Reverse Repo (RRP)",
    line=dict(color="green", width=2),
    yaxis="y1"
))

# Right y-axis: Reserves
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Reserves"] / 1e12,
    name="Reserves (at Fed)",
    line=dict(color="gray", width=2),
    yaxis="y2"
))

# ---------------------------------------------------------
# 5. Layout with slider + range selector + dragmode
# ---------------------------------------------------------
fig.update_layout(
    title="US Reverse Repo and Reserves",
    dragmode="pan",   # ✅ Left mouse drag = pan/scroll chart
    xaxis=dict(
        title="Date",
        rangeslider=dict(visible=True),   # ✅ slider at bottom
        rangeselector=dict(               # ✅ buttons for quick zoom
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        type="date"
    ),
    yaxis=dict(
        title="Reverse Repo (Trillion USD)",
        side="left",
        showgrid=True,
        zeroline=False
    ),
    yaxis2=dict(
        title="Reserves (Trillion USD)",
        side="right",
        overlaying="y",
        showgrid=False,
        zeroline=False
    ),
    legend=dict(x=0.1, y=1.1, orientation="h"),
    template="plotly_white"
)

# ---------------------------------------------------------
# 6. Show chart with mouse wheel zoom enabled
# ---------------------------------------------------------
fig.show(config={"scrollZoom": True})

