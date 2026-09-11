"""
India Startup Ecosystem - Interactive Dashboard
===============================================
Run with:   streamlit run dashboard.py

Every number here comes from the functions in startup_analysis.py, so the
dashboard and outputs/findings.md always agree.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import startup_analysis as sa

st.set_page_config(page_title="India Startup Ecosystem Analytics", page_icon="🚀", layout="wide")

BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#9aa4b2"
SEGMENT_COLORS = {"Established Leader": BLUE, "Maturing Hub": GREY,
                  "Rising Challenger": ORANGE, "Slowing / Early": "#c9ced6"}
YEARS = list(range(sa.FIRST_FULL_YEAR, sa.LAST_YEAR + 1))


@st.cache_data(show_spinner="Running the analysis pipeline...")
def load():
    return sa.run_all(save=False, verbose=False)


R = load()
panel = R["panel"]
nat = R["national"]
sy = R["state_year"]
iy = R["industry_year"]
growth = R["state_growth"]
segments = R["segments"]
ind_growth = R["industry_growth"]

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("🚀 Startup Ecosystem")
page = st.sidebar.radio("Go to", [
    "Overview", "State Explorer", "Sector Explorer", "Growth Map",
    "Hotspot Finder", "Forecast", "Data & Method",
])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Source: DPIIT Startup India recognitions by state x industry x year, 2016-2025 "
    "(data as of 31 May 2026). Counts are **recognitions**, not funding or survival."
)


def hbar_frame(series: pd.Series) -> pd.DataFrame:
    """Series -> tidy frame for a horizontal bar chart, biggest bar on top."""
    return pd.DataFrame({"label": series.index.astype(str), "value": series.values})[::-1]


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


# -----------------------------------------------------------------------------
# 1. Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.title("India's Startup Ecosystem, 2017-2025")
    st.write("Where startups are being created, which sectors are rising, and how fast the map is changing.")

    d, dip = R["decentralisation"], R["dip_2024"]
    kpi_row([
        ("New recognitions, 2025", f"{nat[2025]:,}", f"{(nat[2025] / nat[2024] - 1) * 100:+.1f}% vs 2024"),
        ("Growth since 2017", f"{nat[2025] / nat[2017]:.1f}x", None),
        ("Top-5 state share", f"{d['top5_end']}%", f"{d['top5_end'] - d['top5_start']:+.1f}pp since 2017"),
        ("States with 1,000+ in 2025", int((sy[2025] >= 1000).sum()), None),
    ])

    c1, c2 = st.columns([3, 2])
    with c1:
        df = nat[nat.index >= sa.FIRST_FULL_YEAR].reset_index()
        df.columns = ["year", "startups"]
        df["yoy"] = df["startups"].pct_change() * 100
        fig = px.bar(df, x="year", y="startups", text="startups",
                     color=df["year"].eq(2024).map({True: "Only decline (2024)", False: "Growth year"}),
                     color_discrete_map={"Only decline (2024)": ORANGE, "Growth year": BLUE},
                     title="New recognitions per year")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(legend_title_text="", height=420, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        conc = R["concentration"]
        fig = go.Figure()
        fig.add_scatter(x=conc["year"], y=conc["top5_share_pct"], name="Top-5 share (%)",
                        mode="lines+markers", line=dict(color=BLUE))
        fig.add_scatter(x=conc["year"], y=conc["states_for_80pct"], name="States for 80%",
                        mode="lines+markers", line=dict(color=ORANGE), yaxis="y2")
        fig.update_layout(title="Is it spreading out?", height=420,
                          yaxis=dict(title="Top-5 share (%)"),
                          yaxis2=dict(title="States needed for 80%", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What the data says")
    c = R["convergence"]
    st.markdown(f"""
- **Decentralising, slowly.** Top-5 share fell from {d['top5_start']}% to {d['top5_end']}% (Spearman ρ vs year = {d['spearman_rho']}).
- **The 2024 dip was broad-based**: {dip['states_declining']} of {dip['states_considered']} sizeable states fell, and ~{-dip['others_reclassification_effect']} of the {-dip['national_change']} drop was an 'Others' reclassification. 2025 rebounded {dip['rebound_2025_pct']}%.
- **Smaller ecosystems grow faster** (ρ = {c['spearman_rho']}, p = {c['p_value']:.3f}, {c['n_states']} states), so growth is always shown as CAGR *and* absolute numbers.
- **AI took off**: {int(iy.loc['AI', 2022]):,} recognitions in 2022 → {int(iy.loc['AI', 2025]):,} in 2025.
""")

    st.subheader("Where 2025's startups came from")
    tm = panel[panel["year"] == 2025].copy()
    top_states = sy[2025].sort_values(ascending=False).head(12).index
    tm["state"] = tm["state"].where(tm["state"].isin(top_states), "Other states")
    tm = tm.groupby(["state", "industry"], as_index=False)["startups"].sum()
    tm = tm[tm["startups"] > 0]
    fig = px.treemap(tm, path=["state", "industry"], values="startups", color="state",
                     color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=520, margin=dict(t=10, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 2. State Explorer
# -----------------------------------------------------------------------------
elif page == "State Explorer":
    st.title("State Explorer")
    states_by_size = list(sy[2025].sort_values(ascending=False).index)
    state = st.selectbox("Choose a state", states_by_size, index=0)
    row = growth.loc[state]
    rank_2025 = int(sy[2025].rank(ascending=False, method="min")[state])
    seg = segments["segment"].get(state, "Not classified (2019 base < 50)")
    clusters = R["clusters"]
    cluster = clusters["cluster_name"].get(state, "Not clustered (under 300 recognitions in 2023-25)")

    kpi_row([
        ("Recognitions 2025", f"{int(row['count_2025']):,}", f"rank #{rank_2025} of {len(sy)}"),
        ("Share of India", f"{row['share_2025_pct']:.1f}%", None),
        ("CAGR 2019-25", "n/a" if pd.isna(row["cagr_2019_2025_pct"]) else f"{row['cagr_2019_2025_pct']:.1f}%", None),
        ("Momentum (CAGR 23-25 minus 19-23)",
         "n/a" if pd.isna(row["momentum_pp"]) else f"{row['momentum_pp']:+.1f}pp", None),
    ])
    st.info(f"**Segment:** {seg}  ·  **Industry-mix cluster:** {cluster}")

    c1, c2 = st.columns(2)
    with c1:
        df = pd.DataFrame({"year": YEARS,
                           state: (sy.loc[state, YEARS] / max(sy.loc[state, 2019], 1) * 100).values,
                           "India": (nat[YEARS] / nat[2019] * 100).values})
        fig = px.line(df, x="year", y=[state, "India"], markers=True,
                      title="Growth indexed to 2019 = 100", color_discrete_sequence=[ORANGE, GREY])
        fig.update_layout(height=400, legend_title_text="", yaxis_title="Index")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        recent = panel[(panel["state"] == state) & panel["year"].between(*sa.RECENT_WINDOW)]
        top_ind = recent.groupby("industry")["startups"].sum().sort_values(ascending=False).head(10)
        fig = px.bar(hbar_frame(top_ind), x="value", y="label", orientation="h",
                     title="Top 10 sectors, 2023-25", color_discrete_sequence=[BLUE])
        fig.update_layout(height=400, showlegend=False, xaxis_title="Recognitions", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("What this state specialises in")
    st.caption("Location quotient = sector's share in this state ÷ its share nationally. "
               "Above 1 = over-represented. Only sectors with 30+ recognitions in 2023-25 shown.")
    lq = R["lq"][R["lq"]["state"] == state].sort_values("lq", ascending=False)
    if lq.empty:
        st.write("Not enough activity in any sector to compute reliable specialisation.")
    else:
        show = pd.concat([lq.head(8), lq.tail(5)]).drop_duplicates().sort_values("lq")
        show["type"] = np.where(show["lq"] >= 1, "Over-represented", "Under-represented")
        fig = px.bar(show, x="lq", y="industry", orientation="h", color="type",
                     color_discrete_map={"Over-represented": BLUE, "Under-represented": ORANGE},
                     hover_data=["count"])
        fig.add_vline(x=1, line_dash="dash", line_color="black")
        fig.update_layout(height=450, showlegend=False, xaxis_title="Location quotient", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Compare states")
    compare = st.multiselect("Pick up to 5 states", states_by_size, default=states_by_size[:4], max_selections=5)
    if compare:
        df = sy.loc[compare, YEARS].T.reset_index().melt(id_vars="year", var_name="state", value_name="startups")
        fig = px.line(df, x="year", y="startups", color="state", markers=True)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. Sector Explorer
# -----------------------------------------------------------------------------
elif page == "Sector Explorer":
    st.title("Sector Explorer")
    sectors = list(ind_growth.index)
    sector = st.selectbox("Choose a sector", sectors, index=sectors.index("AI") if "AI" in sectors else 0)
    row = ind_growth.loc[sector]
    if row["taxonomy_flag"]:
        st.warning(f"Data note: {row['taxonomy_flag']}.")

    kpi_row([
        ("Recognitions 2025", f"{int(row['count_2025']):,}", None),
        ("Share of all startups", f"{row['share_2025_pct']:.1f}%", f"{row['share_change_pp']:+.1f}pp since 2019"),
        ("CAGR 2019-25", "n/a" if pd.isna(row["cagr_2019_2025_pct"]) else f"{row['cagr_2019_2025_pct']:.1f}%", None),
        ("CAGR 2023-25", "n/a" if pd.isna(row["cagr_2023_2025_pct"]) else f"{row['cagr_2023_2025_pct']:.1f}%", None),
    ])

    c1, c2 = st.columns(2)
    with c1:
        df = pd.DataFrame({"year": YEARS, "startups": iy.loc[sector, YEARS].values,
                           "share_pct": (iy.loc[sector, YEARS] / nat[YEARS] * 100).values})
        fig = go.Figure()
        fig.add_bar(x=df["year"], y=df["startups"], name="Recognitions", marker_color=BLUE)
        fig.add_scatter(x=df["year"], y=df["share_pct"], name="Share of all (%)", yaxis="y2",
                        mode="lines+markers", line=dict(color=ORANGE))
        fig.update_layout(title=f"{sector}: volume and share", height=400,
                          yaxis2=dict(overlaying="y", side="right", title="Share (%)"),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        recent = panel[(panel["industry"] == sector) & panel["year"].between(*sa.RECENT_WINDOW)]
        by_state = recent.groupby("state")["startups"].sum().sort_values(ascending=False)
        share = by_state / by_state.sum() * 100 if by_state.sum() else by_state
        hhi = float(((share / 100) ** 2).sum() * 10_000) if by_state.sum() else float("nan")
        fig = px.bar(hbar_frame(by_state.head(10)), x="value", y="label", orientation="h",
                     title="Top 10 states, 2023-25", color_discrete_sequence=[BLUE])
        fig.update_layout(height=400, showlegend=False, xaxis_title="Recognitions", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        overall_hhi = R["ai_concentration"].set_index("industry").loc["ALL STARTUPS", "hhi"]
        st.caption(f"Geographic concentration (HHI): **{hhi:.0f}** for {sector} vs **{overall_hhi:.0f}** "
                   f"for all startups. Higher = more concentrated in a few states.")

    st.subheader("Rising and falling sectors, 2019 → 2025")
    clean_ind = ind_growth[ind_growth["taxonomy_flag"] == ""]
    shift = pd.concat([clean_ind["share_change_pp"].nlargest(10), clean_ind["share_change_pp"].nsmallest(10)])
    shift = shift.sort_values()
    sdf = hbar_frame(shift[::-1])
    sdf["direction"] = np.where(sdf["value"] > 0, "Gained share", "Lost share")
    fig = px.bar(sdf, x="value", y="label", orientation="h", color="direction",
                 color_discrete_map={"Gained share": BLUE, "Lost share": ORANGE})
    fig.update_layout(height=560, showlegend=False, xaxis_title="Change in share (percentage points)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Growth Map
# -----------------------------------------------------------------------------
elif page == "Growth Map":
    st.title("Growth Map: scale vs acceleration")
    st.write("Each dot is a state with at least 50 recognitions in 2019. Right = bigger, up = growth speeding up "
             "(CAGR 2023-25 minus CAGR 2019-23). Dashed lines are medians, so the labels are relative.")
    seg = segments.reset_index().rename(columns={"index": "state"})
    fig = px.scatter(seg, x="recent_3yr_total", y="momentum_pp", color="segment", text="state",
                     log_x=True, size="count_2025", size_max=40, color_discrete_map=SEGMENT_COLORS,
                     hover_data={"cagr_2019_2025_pct": ":.1f", "count_2025": ":,"})
    fig.add_vline(x=R["segment_medians"]["size_median"], line_dash="dash", line_color="grey")
    fig.add_hline(y=R["segment_medians"]["momentum_median"], line_dash="dash", line_color="grey")
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_layout(height=620, xaxis_title="Recognitions 2023-25 (log scale)",
                      yaxis_title="Momentum (pp)", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why growth needs two numbers")
    metric = st.radio("Rank states by", ["CAGR 2019-25 (%)", "Absolute growth 2019-25"], horizontal=True)
    col = "cagr_2019_2025_pct" if metric.startswith("CAGR") else "absolute_growth"
    g = growth.dropna(subset=[col]).sort_values(col, ascending=False).head(15)
    fig = px.bar(hbar_frame(g[col]), x="value", y="label", orientation="h", color_discrete_sequence=[BLUE])
    fig.update_layout(height=480, showlegend=False, xaxis_title=metric, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    big_gaps = growth[growth["eligible_for_cagr"]].sort_values("rank_gap", ascending=False).head(5)
    st.caption("Biggest disagreements between the two rankings: " + ", ".join(
        f"{s} (CAGR #{int(r['rank_cagr'])} vs absolute #{int(r['rank_absolute'])})" for s, r in big_gaps.iterrows()))

# -----------------------------------------------------------------------------
# 5. Hotspot Finder
# -----------------------------------------------------------------------------
elif page == "Hotspot Finder":
    st.title("Hotspot Finder: state × sector")
    st.write("Three separate signals for every state-sector pair with 30+ recognitions in 2023-25. "
             "Each is a percentile (0-100). Move the weights to see how the ranking changes.")
    comp = R["hotspot_components"]

    c1, c2, c3 = st.columns(3)
    w_scale = c1.slider("Scale weight", 0, 100, 33, help="How many startups the pair produced in 2023-25")
    w_mom = c2.slider("Momentum weight", 0, 100, 33, help="Growth rate 2023→2025")
    w_spec = c3.slider("Specialisation weight", 0, 100, 34, help="Location quotient vs national average")
    if w_scale + w_mom + w_spec == 0:
        st.error("At least one weight must be above zero.")
        st.stop()

    f1, f2, f3 = st.columns(3)
    pick_states = f1.multiselect("Filter states", sorted(comp["state"].unique()))
    pick_sectors = f2.multiselect("Filter sectors", sorted(comp["industry"].unique()))
    min_count = f3.number_input("Min recognitions 2023-25", min_value=30, max_value=2000, value=30, step=10)

    scored = sa.hotspot_score(comp, w_scale, w_mom, w_spec)
    scored["overall_rank"] = np.arange(1, len(scored) + 1)
    view = scored[scored["recent_3yr"] >= min_count]
    if pick_states:
        view = view[view["state"].isin(pick_states)]
    if pick_sectors:
        view = view[view["industry"].isin(pick_sectors)]

    top = view.head(20).copy()
    top["pair"] = top["state"] + " · " + top["industry"]
    long = top.melt(id_vars="pair", value_vars=["scale_pct", "momentum_pct", "specialisation_pct"],
                    var_name="signal", value_name="percentile")
    long["signal"] = long["signal"].str.replace("_pct", "").str.capitalize()
    fig = px.bar(long, x="percentile", y="pair", color="signal", orientation="h", barmode="group",
                 color_discrete_sequence=[BLUE, ORANGE, GREY], category_orders={"pair": list(top["pair"])})
    fig.update_layout(height=max(400, 32 * len(top)), yaxis_title="", legend_title_text="",
                      title="Top 20 by your weights - the three signals side by side")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        view[["overall_rank", "state", "industry", "recent_3yr", "growth_2023_2025_pct", "lq",
              "scale_pct", "momentum_pct", "specialisation_pct", "hotspot_score"]].head(50)
        .rename(columns={"recent_3yr": "count 2023-25", "growth_2023_2025_pct": "growth %/yr", "lq": "LQ"}),
        use_container_width=True, hide_index=True,
    )
    with st.expander("How stable is this ranking? (weight sensitivity test)"):
        st.dataframe(R["hotspot_sensitivity"], hide_index=True, use_container_width=True)
        st.caption("Top-20 overlap and rank correlation vs equal weights. The list shifts noticeably under "
                   "extreme weights, which is why the weights are yours to set rather than hidden.")

# -----------------------------------------------------------------------------
# 6. Forecast
# -----------------------------------------------------------------------------
elif page == "Forecast":
    st.title("Forecast: how many recognitions in 2026-27?")
    bt = R["forecast_backtest"]
    st.write("Four simple methods were trained on 2017-2022 and scored on 2023-2025 **without seeing those years**. "
             "The winner is refitted on all data; the shaded band is the spread of all four methods.")

    choice = st.selectbox("Method", list(bt["method"]), index=0,
                          format_func=lambda m: f"{m}  (holdout MAPE {bt.set_index('method').loc[m, 'mape_pct']}%)")
    fc = sa.forecast(nat, choice)
    hist = nat[nat.index >= sa.FIRST_FULL_YEAR]
    yrs = list(fc.keys())
    fig = go.Figure()
    fig.add_scatter(x=list(hist.index), y=list(hist.values), mode="lines+markers", name="Actual",
                    line=dict(color=BLUE))
    fig.add_scatter(x=yrs + yrs[::-1], y=[fc[y]["high"] for y in yrs] + [fc[y]["low"] for y in yrs][::-1],
                    fill="toself", fillcolor="rgba(232,131,58,0.2)", line=dict(width=0),
                    name="Range across methods")
    fig.add_scatter(x=[hist.index[-1]] + yrs, y=[hist.values[-1]] + [fc[y]["point"] for y in yrs],
                    mode="lines+markers", line=dict(color=ORANGE, dash="dash"), name=f"Forecast: {choice}")
    fig.update_layout(height=460, xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    kpi_row([(f"{y} forecast", f"{fc[y]['point']:,}", f"range {fc[y]['low']:,}-{fc[y]['high']:,}") for y in yrs])
    st.subheader("Holdout backtest")
    st.dataframe(bt, hide_index=True, use_container_width=True)
    st.warning("Nine annual points, and a series driven by government policy. Treat this as a rough planning "
               "range, not a prediction.")

# -----------------------------------------------------------------------------
# 7. Data & Method
# -----------------------------------------------------------------------------
elif page == "Data & Method":
    st.title("Data & Method")
    a = R["audit"]
    kpi_row([("Raw rows", f"{a['rows']:,}", None), ("States / UTs", a["states"], None),
             ("Industries", a["industries"], None), ("Total recognitions", f"{a['total_recognitions']:,}", None)])
    st.markdown("""
**Cleaning decisions**
- The raw file only lists state-industry-year cells with at least 1 startup. Missing cells are real zeros, so the
  panel is zero-filled (9,932 rows → 20,160). Without this, shares and growth rates come out wrong.
- 2016 is the scheme's launch year (502 recognitions vs 5,473 in 2017), so it's left out of growth maths.
- The 'Others' category collapses from 240 (2023) to 1 (2024): a reclassification, flagged and excluded from sector trends.
- State CAGR needs a 2019 base of 50+; small bases produce misleading percentages.

**Limitations**
- Recognitions measure registrations, not funding, revenue or survival.
- Sector labels are self-reported; state is where a startup registered, not necessarily where it operates.
""")
    st.subheader("SQL queries")
    q = st.selectbox("Query", list(sa.SQL_QUERIES))
    st.code(sa.SQL_QUERIES[q].strip(), language="sql")
    st.dataframe(R["sql"][q], hide_index=True, use_container_width=True)

    st.download_button("Download clean panel (CSV)", panel.to_csv(index=False).encode(),
                       file_name="startups_panel.csv", mime="text/csv")
