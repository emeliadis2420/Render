import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

# === DATA LOADING & PROCESSING ===
df = pd.read_excel("Kodiak_excel_skjal_2015-30_04_25.xlsx")
df.dropna(how="all", inplace=True)
df["Vextir"] = pd.to_numeric(df["Vextir"], errors="coerce")
df = df.dropna(subset=["Vextir"])
df["time_exec"] = pd.to_datetime(df["time_exec"])
df["Maturity"] = pd.to_datetime(df["Maturity"])
df["yield"] = (df["yield"] * 100).round(5)
df["Issuer"] = df["Issuer"].astype(str)

def map_issuer_type(iss):
    s = iss.lower()
    if "bank" in s:
        return "banki"
    elif "íl-sjóður" in s:
        return "ibudasjodur"
    elif any(tok in s for tok in ("bær", "borg", "sveitarfélag", "hreppur")):
        return "sveitarfelag"
    elif "ríkissjóður íslands" in s:
        return "rikissjodur"
    else:
        return "annad"

df["IssuerType"] = df["Issuer"].apply(map_issuer_type)
df["trade_date"] = df["time_exec"].dt.date
df = df.sort_values(by=["trade_date", "time_exec"], ascending=[True, False])
df = df.groupby(["trade_date", "symbol"]).first().reset_index()
df["Category"] = "Non-Government"
df.loc[df["symbol"].str.startswith("RIKB"), "Category"] = "Gov Non-Indexed"
df.loc[df["symbol"].str.startswith("RIKS"), "Category"] = "Gov Indexed"
df.loc[(df["Tryggt"] > 0) & (~df["symbol"].str.startswith("RIK")), "Category"] = "Non-Gov Indexed"

global_dates = pd.date_range(start=df["trade_date"].min(), end=df["trade_date"].max(), freq="B")

def fill_missing_trades(group):
    group["trade_date"] = pd.to_datetime(group["trade_date"])
    group = group.set_index("trade_date").sort_index()
    original_index = group.index.copy()
    start_date = group.index.min()
    maturity_date = pd.to_datetime(group["Maturity"].iloc[0])
    end_date = min(global_dates.max(), maturity_date)
    bond_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    group = group.reindex(bond_dates)
    group = group.ffill()
    group["volume"] = group["volume"].fillna(0)
    group.loc[~group.index.isin(original_index), "volume"] = 0
    return group.reset_index().rename(columns={"index": "trade_date"})

df_filled = df.groupby("symbol", group_keys=False).apply(fill_missing_trades)

def compute_credit_spread_for_day(df_day, trade_date):
    df_day = df_day[df_day["Category"].notna()].copy()
    gov_non = df_day[df_day["Category"] == "Gov Non-Indexed"].drop_duplicates("duration").sort_values("duration")
    gov_idx = df_day[df_day["Category"] == "Gov Indexed"].drop_duplicates("duration").sort_values("duration")
    non_gov = df_day[df_day["Category"].str.startswith("Non-Gov")].copy()

    interp_non = interp1d(gov_non["duration"], gov_non["yield"],
                          bounds_error=False, fill_value=(gov_non["yield"].iloc[0], gov_non["yield"].iloc[-1])) if len(gov_non) > 1 else lambda x: np.nan
    interp_idx = interp1d(gov_idx["duration"], gov_idx["yield"],
                          bounds_error=False, fill_value=(gov_idx["yield"].iloc[0], gov_idx["yield"].iloc[-1])) if len(gov_idx) > 1 else lambda x: np.nan

    non_gov["Interpolated Gov Yield"] = np.where(
        non_gov["Category"] == "Non-Gov Indexed",
        non_gov["duration"].apply(interp_idx),
        non_gov["duration"].apply(interp_non)
    )
    non_gov["Credit Spread"] = non_gov["yield"] - non_gov["Interpolated Gov Yield"]
    non_gov["trade_date"] = trade_date
    return non_gov

processed = [compute_credit_spread_for_day(df_filled[df_filled["trade_date"] == day], day)
             for day in df_filled["trade_date"].unique()]
df_non_gov = pd.concat(processed)
df_final = pd.merge(df_filled, df_non_gov[["trade_date", "symbol", "Interpolated Gov Yield", "Credit Spread"]],
                    on=["trade_date", "symbol"], how="left")

df = df_final.copy()
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["date_only"] = df["trade_date"].dt.date
dates = sorted(df["date_only"].dropna().unique())
df["Issuer"] = df.get("Issuer", pd.Series(dtype=str)).astype(str)
df["IssuerType"] = df.get("IssuerType", pd.Series(dtype=str)).astype(str)

issuer_types = ["banki", "ibudasjodur", "sveitarfelag", "annad"]
flat_issuer_options = []
for itype in issuer_types:
    flat_issuer_options.append({
        "label": html.Span(itype.capitalize(), style={"fontWeight": "bold"}),
        "value": itype
    })
    for name in sorted(df[df["IssuerType"] == itype]["Issuer"].unique()):
        flat_issuer_options.append({
            "label": "\u00a0\u00a0" + name,
            "value": name
        })

# === DASH APP ===
app = dash.Dash(__name__)
app.title = "Icelandic Bonds"

app.layout = html.Div([
    html.H1("Icelandic Bond Market Dashboard"),

    dcc.Interval(
        id="interval-component",
        interval=1800000,  # 30 minutes = 30 * 60 * 1000 ms
        n_intervals=0
    ),

    html.Div([
        html.Label("Select Trade Date:"),
        dcc.DatePickerSingle(
            id="date-dropdown",
            date=str(dates[-1]),
            min_date_allowed=min(dates),
            max_date_allowed=max(dates),
            display_format="YYYY-MM-DD",
            style={"border": "none"},
        )
    ], style={"width": "300px", "margin": "20px"}),

    html.Div([
        html.Label("Filter Non-Gov Bonds by Issuer:"),
        dcc.Dropdown(
            id="issuer-dropdown",
            options=flat_issuer_options,
            multi=True,
            placeholder="(none = all non-gov)",
        )
    ], style={"width": "400px", "margin": "20px"}),

    html.Div([dcc.Graph(id="non-indexed-graph")]),
    html.Div([dcc.Graph(id="indexed-graph")]),

    html.Div([
        html.H2("Non-Government Bonds Credit Spread Data"),
        dash_table.DataTable(
            id="credit-spread-table",
            columns=[
                {"name": "Bond Name", "id": "symbol"},
                {"name": "Yield", "id": "yield"},
                {"name": "Credit Spread", "id": "Credit Spread"},
            ],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
        )
    ], style={"margin": "20px"})
])

@app.callback(
    [
        Output("non-indexed-graph", "figure"),
        Output("indexed-graph", "figure"),
        Output("credit-spread-table", "data"),
    ],
    [
        Input("date-dropdown", "date"),
        Input("issuer-dropdown", "value"),
        Input("interval-component", "n_intervals")
    ]
)
def update_graphs(selected_date, selected_issuers, _):
    sel_date = pd.to_datetime(selected_date).date()
    df_day = df[df["date_only"] == sel_date]

    gov_non = df_day[df_day["Category"] == "Gov Non-Indexed"].sort_values("duration")
    gov_idx = df_day[df_day["Category"] == "Gov Indexed"].sort_values("duration")
    nongov_non = df_day[(df_day["Category"] == "Non-Government") & (df_day["volume"] > 0)].copy()
    nongov_idx = df_day[(df_day["Category"] == "Non-Gov Indexed") & (df_day["volume"] > 0)].copy()

    if selected_issuers:
        to_show = set()
        for val in selected_issuers:
            if val in issuer_types:
                to_show.update(df[df["IssuerType"] == val]["Issuer"].unique())
            else:
                to_show.add(val)
        nongov_non = nongov_non[nongov_non["Issuer"].isin(to_show)]
        nongov_idx = nongov_idx[nongov_idx["Issuer"].isin(to_show)]

    fig_non = px.scatter(gov_non, x="duration", y="yield", title="Non-Indexed Yield Curve")
    fig_non.add_scatter(x=gov_non["duration"], y=gov_non["yield"], mode="lines", name="Gov Curve")
    fig_non.add_scatter(x=nongov_non["duration"], y=nongov_non["yield"],
                        mode="markers+text", text=nongov_non["symbol"],
                        textposition="top center", name="Non-Gov Bonds")

    fig_idx = px.scatter(gov_idx, x="duration", y="yield", title="Indexed Yield Curve")
    fig_idx.add_scatter(x=gov_idx["duration"], y=gov_idx["yield"], mode="lines", name="Gov Curve")
    fig_idx.add_scatter(x=nongov_idx["duration"], y=nongov_idx["yield"],
                        mode="markers+text", text=nongov_idx["symbol"],
                        textposition="top center", name="Non-Gov Bonds")

    tbl_data = pd.concat([nongov_non, nongov_idx])[["symbol", "yield", "Credit Spread"]].to_dict("records")
    return fig_non, fig_idx, tbl_data

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
