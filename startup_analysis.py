"""
India Startup Ecosystem Analytics
=================================
Where is India's startup ecosystem actually growing, which sectors are rising,
and which state x sector combinations look like genuine hotspots?

Data: DPIIT (Startup India) recognised startups by state x industry x year,
      2016-2025 (data as of 31 May 2026). 9,932 rows, 36 states/UTs, 56 industries.

This single file holds the whole analysis pipeline. Run it directly:

    python startup_analysis.py

and it will (1) audit and clean the raw CSV, (2) load a SQLite database and
run the analytical SQL, (3) run every analysis and statistical test,
(4) save charts to outputs/figures and tables to outputs/tables, and
(5) write a plain-English findings report to outputs/findings.md.

The Streamlit dashboard (dashboard.py) imports the functions below, so the
dashboard and the report can never disagree about a number.

Important limitation, stated up front: DPIIT data counts *recognitions*
(a startup registering with the government scheme). It says nothing about
funding, revenue, or survival. "Growth" here means growth in new
recognitions, not business success.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "dpiit_startups_state_industry_year_2016_2026.csv"
DB_FILE = ROOT / "data" / "processed" / "startups.db"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

# 2016 is the scheme's launch year and only has 502 recognitions (vs 5,473 in 2017),
# so it's treated as a ramp-up year and left out of every growth calculation.
FIRST_FULL_YEAR = 2017
LAST_YEAR = 2025
CAGR_START = 2019           # 2017-2018 bases are too small for many states
MIN_BASE_FOR_CAGR = 50      # a state needs >=50 recognitions in the base year for its CAGR to count
RECENT_WINDOW = (2023, 2025)  # "current" 3-year window used for scale / specialisation


# =============================================================================
# 1. LOAD + AUDIT + CLEAN
# =============================================================================
def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def audit(df: pd.DataFrame) -> dict:
    """Facts about the raw file that shape every later decision."""
    count_col = "startups_ recognized"
    yearly = df.groupby("year")[count_col].sum()
    others = df[df["industry"].str.strip() == "Others"].groupby("year")[count_col].sum()
    return {
        "rows": len(df),
        "years": (int(df["year"].min()), int(df["year"].max())),
        "states": df["state"].nunique(),
        "industries": df["industry"].nunique(),
        "total_recognitions": int(df[count_col].sum()),
        "min_count": int(df[count_col].min()),   # 1 -> zero-count combinations are simply missing
        "duplicate_keys": int(df.duplicated(["year", "state", "industry"]).sum()),
        "nulls_in_count": int(df[count_col].isna().sum()),
        "yearly_totals": yearly.to_dict(),
        "others_by_year": others.to_dict(),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tidy long panel with every (year, state, industry) combination present.
    The raw file only lists combinations with >=1 startup; missing combos are
    real zeros, and must be filled or growth rates / shares come out wrong.
    """
    df = df.rename(columns={"startups_ recognized": "startups"})
    df = df[["year", "state", "industry", "startups"]].copy()
    df["state"] = df["state"].str.strip()
    df["industry"] = df["industry"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["industry"] = df["industry"].replace({"Non- Renewable Energy": "Non-Renewable Energy"})
    df = df.groupby(["year", "state", "industry"], as_index=False)["startups"].sum()

    full_index = pd.MultiIndex.from_product(
        [sorted(df["year"].unique()), sorted(df["state"].unique()), sorted(df["industry"].unique())],
        names=["year", "state", "industry"],
    )
    panel = df.set_index(["year", "state", "industry"]).reindex(full_index, fill_value=0).reset_index()
    panel["startups"] = panel["startups"].astype(int)
    return panel


# =============================================================================
# 2. SQL LAYER
# =============================================================================
SQL_QUERIES = {
    # Year-over-year national growth with LAG()
    "national_yoy": """
        WITH yearly AS (
            SELECT year, SUM(startups) AS total
            FROM startups GROUP BY year
        )
        SELECT year, total,
               LAG(total) OVER (ORDER BY year) AS prev_total,
               ROUND(100.0 * (total - LAG(total) OVER (ORDER BY year))
                     / LAG(total) OVER (ORDER BY year), 1) AS yoy_growth_pct
        FROM yearly ORDER BY year;
    """,
    # State ranking each year with RANK() + share of the national total
    "state_rank_by_year": """
        WITH st AS (
            SELECT year, state, SUM(startups) AS total
            FROM startups GROUP BY year, state
        )
        SELECT year, state, total,
               RANK() OVER (PARTITION BY year ORDER BY total DESC) AS rank_in_year,
               ROUND(100.0 * total / SUM(total) OVER (PARTITION BY year), 2) AS share_pct
        FROM st WHERE year >= 2017
        ORDER BY year, rank_in_year;
    """,
    # Running (cumulative) total per state
    "state_cumulative": """
        WITH st AS (
            SELECT year, state, SUM(startups) AS total
            FROM startups GROUP BY year, state
        )
        SELECT year, state, total,
               SUM(total) OVER (PARTITION BY state ORDER BY year
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative
        FROM st ORDER BY state, year;
    """,
    # Industry share shift between two years (conditional aggregation)
    "industry_share_shift": """
        WITH ind AS (
            SELECT industry,
                   SUM(CASE WHEN year = 2019 THEN startups ELSE 0 END) AS y2019,
                   SUM(CASE WHEN year = 2025 THEN startups ELSE 0 END) AS y2025
            FROM startups GROUP BY industry
        ), tot AS (
            SELECT SUM(y2019) AS t2019, SUM(y2025) AS t2025 FROM ind
        )
        SELECT industry, y2019, y2025,
               ROUND(100.0 * y2019 / t2019, 2) AS share_2019_pct,
               ROUND(100.0 * y2025 / t2025, 2) AS share_2025_pct,
               ROUND(100.0 * y2025 / t2025 - 100.0 * y2019 / t2019, 2) AS share_change_pp
        FROM ind, tot
        ORDER BY share_change_pp DESC;
    """,
    # Each state's #1 industry in the recent window (ROW_NUMBER to pick the top row)
    "state_top_industry": """
        WITH recent AS (
            SELECT state, industry, SUM(startups) AS total
            FROM startups WHERE year BETWEEN 2023 AND 2025
            GROUP BY state, industry
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY state ORDER BY total DESC) AS rn,
                      SUM(total) OVER (PARTITION BY state) AS state_total
            FROM recent
        )
        SELECT state, industry AS top_industry, total,
               ROUND(100.0 * total / state_total, 1) AS pct_of_state
        FROM ranked WHERE rn = 1 AND state_total > 0
        ORDER BY state_total DESC;
    """,
}


def build_database(panel: pd.DataFrame, db_path: Path = DB_FILE) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    panel.to_sql("startups", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_year_state ON startups(year, state)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_industry ON startups(industry)")
    conn.commit()
    return conn


def run_sql(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {name: pd.read_sql(q, conn) for name, q in SQL_QUERIES.items()}


# =============================================================================
# 3. CORE AGGREGATES
# =============================================================================
def national_series(panel: pd.DataFrame) -> pd.Series:
    return panel.groupby("year")["startups"].sum()


def state_year(panel: pd.DataFrame) -> pd.DataFrame:
    """states x years matrix of counts."""
    return panel.pivot_table(index="state", columns="year", values="startups", aggfunc="sum")


def industry_year(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="industry", columns="year", values="startups", aggfunc="sum")


def cagr(start: float, end: float, years: int) -> float:
    if start <= 0 or years <= 0:
        return np.nan
    return ((end / start) ** (1 / years) - 1) * 100


# =============================================================================
# 4. ANALYSES
# =============================================================================
def concentration_over_time(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Is the ecosystem spreading out or concentrating?
    HHI (0-10,000) on state shares, top-5 share, and how many states it takes to reach 80%.
    """
    sy = state_year(panel)
    rows = []
    for yr in range(FIRST_FULL_YEAR, LAST_YEAR + 1):
        s = sy[yr].sort_values(ascending=False)
        share = s / s.sum()
        rows.append({
            "year": yr,
            "hhi": round((share ** 2).sum() * 10_000, 0),
            "top5_share_pct": round(share.head(5).sum() * 100, 1),
            "states_for_80pct": int((share.cumsum() < 0.80).sum() + 1),
            "active_states": int((s > 0).sum()),
        })
    return pd.DataFrame(rows)


def decentralisation_test(conc: pd.DataFrame) -> dict:
    """Spearman trend of HHI against year: negative + significant = spreading out."""
    rho, p = stats.spearmanr(conc["year"], conc["hhi"])
    return {"spearman_rho": round(rho, 2), "p_value": round(p, 4),
            "hhi_start": conc["hhi"].iloc[0], "hhi_end": conc["hhi"].iloc[-1],
            "top5_start": conc["top5_share_pct"].iloc[0], "top5_end": conc["top5_share_pct"].iloc[-1]}


def state_growth_table(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Growth measured three ways, because each one alone misleads:
      - CAGR% (rewards small bases)
      - absolute growth (rewards big states)
      - momentum = recent CAGR (2023-25) minus earlier CAGR (2019-23): is growth speeding up?
    """
    sy = state_year(panel)
    out = pd.DataFrame(index=sy.index)
    out["base_2019"] = sy[CAGR_START]
    out["count_2023"] = sy[2023]
    out["count_2025"] = sy[LAST_YEAR]
    out["recent_3yr_total"] = sy[list(range(RECENT_WINDOW[0], RECENT_WINDOW[1] + 1))].sum(axis=1)
    out["cagr_2019_2025_pct"] = [cagr(a, b, LAST_YEAR - CAGR_START) for a, b in zip(out["base_2019"], out["count_2025"])]
    out["cagr_2019_2023_pct"] = [cagr(a, b, 4) for a, b in zip(sy[2019], sy[2023])]
    out["cagr_2023_2025_pct"] = [cagr(a, b, 2) for a, b in zip(sy[2023], sy[2025])]
    out["momentum_pp"] = out["cagr_2023_2025_pct"] - out["cagr_2019_2023_pct"]
    out["absolute_growth"] = out["count_2025"] - out["base_2019"]
    out["share_2025_pct"] = out["count_2025"] / out["count_2025"].sum() * 100

    eligible = out["base_2019"] >= MIN_BASE_FOR_CAGR
    out.loc[~eligible, ["cagr_2019_2025_pct", "cagr_2019_2023_pct", "cagr_2023_2025_pct", "momentum_pp"]] = np.nan
    out["eligible_for_cagr"] = eligible
    out["rank_cagr"] = out["cagr_2019_2025_pct"].rank(ascending=False, method="min")
    out["rank_absolute"] = out["absolute_growth"].rank(ascending=False, method="min")
    out["rank_gap"] = (out["rank_cagr"] - out["rank_absolute"]).abs()
    return out.sort_values("count_2025", ascending=False).round(2)


def classify_states(growth: pd.DataFrame) -> pd.DataFrame:
    """
    Four-quadrant map on (scale, momentum). Uses medians of eligible states, so
    labels are relative within India, not absolute judgements.
      Established Leader : big + accelerating
      Maturing Hub       : big + slowing
      Rising Challenger  : smaller + accelerating
      Slowing / Early    : smaller + slowing
    """
    g = growth[growth["eligible_for_cagr"]].copy()
    size_med = g["recent_3yr_total"].median()
    mom_med = g["momentum_pp"].median()

    def label(r):
        big = r["recent_3yr_total"] >= size_med
        fast = r["momentum_pp"] >= mom_med
        if big and fast:
            return "Established Leader"
        if big:
            return "Maturing Hub"
        if fast:
            return "Rising Challenger"
        return "Slowing / Early"

    g["segment"] = g.apply(label, axis=1)
    g.attrs["size_median"] = size_med
    g.attrs["momentum_median"] = mom_med
    return g


def convergence_test(growth: pd.DataFrame) -> dict:
    """
    Beta-convergence: do smaller ecosystems grow faster than bigger ones?
    Spearman between log(2019 base) and 2019-25 CAGR, eligible states only.
    """
    g = growth[growth["eligible_for_cagr"]].dropna(subset=["cagr_2019_2025_pct"])
    rho, p = stats.spearmanr(np.log(g["base_2019"]), g["cagr_2019_2025_pct"])
    return {"n_states": len(g), "spearman_rho": round(rho, 2), "p_value": round(p, 4)}


def industry_growth_table(panel: pd.DataFrame) -> pd.DataFrame:
    iy = industry_year(panel)
    nat = iy.sum()
    out = pd.DataFrame(index=iy.index)
    out["count_2019"] = iy[2019]
    out["count_2023"] = iy[2023]
    out["count_2025"] = iy[2025]
    out["share_2019_pct"] = iy[2019] / nat[2019] * 100
    out["share_2025_pct"] = iy[2025] / nat[2025] * 100
    out["share_change_pp"] = out["share_2025_pct"] - out["share_2019_pct"]
    out["cagr_2019_2025_pct"] = [cagr(a, b, 6) if a >= 30 else np.nan for a, b in zip(iy[2019], iy[2025])]
    out["cagr_2023_2025_pct"] = [cagr(a, b, 2) if a >= 30 else np.nan for a, b in zip(iy[2023], iy[2025])]
    # Taxonomy artifacts: 'Others' collapses from 240 (2023) to 1 (2024) -- a reclassification,
    # not a real decline. Flag it so no one reads it as a dying sector.
    out["taxonomy_flag"] = ""
    out.loc[out.index == "Others", "taxonomy_flag"] = "reclassified in 2024 - not a real trend"
    return out.sort_values("count_2025", ascending=False).round(2)


def industry_concentration(panel: pd.DataFrame, industries: list[str], year_range=RECENT_WINDOW) -> pd.DataFrame:
    """Geographic HHI for specific industries vs the ecosystem as a whole."""
    recent = panel[panel["year"].between(*year_range)]
    rows = []
    total_share = recent.groupby("state")["startups"].sum()
    total_share = total_share / total_share.sum()
    rows.append({"industry": "ALL STARTUPS", "hhi": round((total_share ** 2).sum() * 10_000),
                 "top_state": total_share.idxmax(), "top_state_share_pct": round(total_share.max() * 100, 1)})
    for ind in industries:
        s = recent[recent["industry"] == ind].groupby("state")["startups"].sum()
        if s.sum() == 0:
            continue
        sh = s / s.sum()
        rows.append({"industry": ind, "hhi": round((sh ** 2).sum() * 10_000),
                     "top_state": sh.idxmax(), "top_state_share_pct": round(sh.max() * 100, 1)})
    return pd.DataFrame(rows)


def dip_2024_breakdown(panel: pd.DataFrame) -> dict:
    """The only year national recognitions fell. Broad-based, or driven by a few states / a taxonomy change?"""
    sy = state_year(panel)
    iy = industry_year(panel)
    change = (sy[2024] - sy[2023])
    active = sy[2023] >= MIN_BASE_FOR_CAGR
    ind_change = (iy[2024] - iy[2023]).sort_values()
    nat_change = int(change.sum())
    others_effect = int(ind_change.get("Others", 0))
    return {
        "national_change": nat_change,
        "national_change_pct": round(nat_change / sy[2023].sum() * 100, 1),
        "states_declining": int((change[active] < 0).sum()),
        "states_considered": int(active.sum()),
        "biggest_state_drops": change.sort_values().head(5).to_dict(),
        "biggest_industry_drops": ind_change.head(5).to_dict(),
        "others_reclassification_effect": others_effect,
        "change_excluding_others": nat_change - others_effect,
        "rebound_2025_pct": round((sy[2025].sum() / sy[2024].sum() - 1) * 100, 1),
    }


def location_quotients(panel: pd.DataFrame, year_range=RECENT_WINDOW, min_count: int = 30) -> pd.DataFrame:
    """
    Location quotient = (industry share inside the state) / (industry share nationally).
    LQ 2.0 = the state has twice the national concentration of that sector.
    Only state-industry cells with >= min_count startups are kept, so a state with
    3 drone startups doesn't show up as a 'drone capital'.
    """
    recent = panel[panel["year"].between(*year_range)]
    si = recent.groupby(["state", "industry"])["startups"].sum().unstack(fill_value=0)
    state_tot = si.sum(axis=1)
    nat_share = si.sum(axis=0) / si.values.sum()
    lq = si.div(state_tot, axis=0).div(nat_share, axis=1)
    long = lq.stack().rename("lq").reset_index()
    counts = si.stack().rename("count").reset_index()
    long = long.merge(counts, on=["state", "industry"])
    long = long[long["count"] >= min_count]
    return long.sort_values("lq", ascending=False).reset_index(drop=True)


def cluster_states(panel: pd.DataFrame, min_total: int = 300, k_range=range(3, 7), seed: int = 42):
    """
    Group states by *what kind* of startups they produce (industry mix), not by size.
    KMeans on each state's industry-share vector, k chosen by silhouette score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    recent = panel[panel["year"].between(*RECENT_WINDOW)]
    si = recent.groupby(["state", "industry"])["startups"].sum().unstack(fill_value=0)
    si = si[si.sum(axis=1) >= min_total]
    top_ind = si.sum().sort_values(ascending=False).head(15).index
    shares = si[top_ind].div(si.sum(axis=1), axis=0)
    X = StandardScaler().fit_transform(shares)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=20, random_state=seed).fit(X)

    result = shares.copy()
    result["cluster"] = km.labels_
    # Name each cluster by the industries it over-indexes on most vs the average state
    # Name each cluster by the industries it over-indexes on most vs the average state.
    # The cluster whose profile deviates least from the average gets called "broad mix"
    # instead, because naming it after tiny over-indexes would be misleading.
    profile = shares.groupby(km.labels_).mean() - shares.mean()
    deviation = profile.abs().mean(axis=1)
    names = {}
    for c in profile.index:
        if c == deviation.idxmin():
            names[c] = "Broad mix (close to the national average)"
        else:
            names[c] = " + ".join(profile.loc[c].sort_values(ascending=False).head(2).index) + "-heavy"
    result["cluster_name"] = result["cluster"].map(lambda c: f"C{c}: {names[c]}")
    return result, {k: round(v, 3) for k, v in scores.items()}, best_k


# =============================================================================
# 5. HOTSPOT INDEX (state x industry) - components kept separate on purpose
# =============================================================================
def hotspot_components(panel: pd.DataFrame, min_recent: int = 30) -> pd.DataFrame:
    """
    For every state x industry cell with enough activity, three separate signals:
      scale          : recent 3-yr count (2023-25), as a percentile
      momentum       : 2023->2025 CAGR of that cell, as a percentile
      specialisation : location quotient, as a percentile
    They are shown side by side; the composite is optional and its weights are
    user-controlled in the dashboard.
    """
    sy_ind = panel.pivot_table(index=["state", "industry"], columns="year", values="startups", aggfunc="sum")
    df = pd.DataFrame(index=sy_ind.index)
    df["recent_3yr"] = sy_ind[[2023, 2024, 2025]].sum(axis=1)
    df["count_2023"] = sy_ind[2023]
    df["count_2025"] = sy_ind[2025]
    df = df[(df["recent_3yr"] >= min_recent) & (df["count_2023"] >= 5)].copy()
    df["growth_2023_2025_pct"] = ((df["count_2025"] / df["count_2023"]) ** 0.5 - 1) * 100
    lq = location_quotients(panel, min_count=0).set_index(["state", "industry"])["lq"]
    df["lq"] = lq.reindex(df.index)
    df = df.reset_index()
    df["scale_pct"] = df["recent_3yr"].rank(pct=True) * 100
    df["momentum_pct"] = df["growth_2023_2025_pct"].rank(pct=True) * 100
    df["specialisation_pct"] = df["lq"].rank(pct=True) * 100
    return df.round(2)


def hotspot_score(comp: pd.DataFrame, w_scale=1 / 3, w_momentum=1 / 3, w_spec=1 / 3) -> pd.DataFrame:
    total = w_scale + w_momentum + w_spec
    out = comp.copy()
    out["hotspot_score"] = (w_scale * out["scale_pct"] + w_momentum * out["momentum_pct"]
                            + w_spec * out["specialisation_pct"]) / total
    return out.sort_values("hotspot_score", ascending=False).reset_index(drop=True)


def hotspot_weight_sensitivity(comp: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Does the top-20 list survive different weightings? If it doesn't,
    the ranking is mostly an artefact of the weights we picked.
    """
    base = hotspot_score(comp)
    base_top = set(zip(base.head(top_n)["state"], base.head(top_n)["industry"]))
    schemes = {
        "equal (1/3 each)": (1, 1, 1),
        "scale-heavy (60/20/20)": (3, 1, 1),
        "momentum-heavy (20/60/20)": (1, 3, 1),
        "specialisation-heavy (20/20/60)": (1, 1, 3),
        "no momentum (50/0/50)": (1, 0, 1),
    }
    rows = []
    for name, w in schemes.items():
        s = hotspot_score(comp, *w)
        top = set(zip(s.head(top_n)["state"], s.head(top_n)["industry"]))
        rho = stats.spearmanr(base.set_index(["state", "industry"])["hotspot_score"],
                              s.set_index(["state", "industry"]).loc[base.set_index(["state", "industry"]).index]["hotspot_score"])[0]
        rows.append({"scheme": name, f"overlap_with_equal_top{top_n}": len(top & base_top),
                     "spearman_vs_equal": round(rho, 3)})
    return pd.DataFrame(rows)


# =============================================================================
# 6. FORECASTING with an honest holdout backtest
# =============================================================================
def _fit_predict(train: pd.Series, horizon_years: list[int], method: str) -> np.ndarray:
    x = np.array(train.index, dtype=float)
    y = train.values.astype(float)
    h = np.array(horizon_years, dtype=float)
    if method == "naive (last value)":
        return np.repeat(y[-1], len(h))
    if method == "linear trend":
        b, a = np.polyfit(x, y, 1)
        return a + b * h
    if method == "exponential trend":
        b, a = np.polyfit(x, np.log(y), 1)
        return np.exp(a + b * h)
    if method == "avg growth (last 3 yrs)":
        g = (y[-1] / y[-4]) ** (1 / 3) if len(y) >= 4 else y[-1] / y[0]
        return y[-1] * g ** (h - x[-1])
    raise ValueError(method)


FORECAST_METHODS = ["naive (last value)", "linear trend", "exponential trend", "avg growth (last 3 yrs)"]


def backtest_forecasts(series: pd.Series, holdout_years=(2023, 2024, 2025)) -> pd.DataFrame:
    """Train on years before the holdout, predict the holdout, score by MAPE. No peeking."""
    series = series[series.index >= FIRST_FULL_YEAR]
    train = series[series.index < holdout_years[0]]
    actual = series.loc[list(holdout_years)].values
    rows = []
    for m in FORECAST_METHODS:
        pred = _fit_predict(train, list(holdout_years), m)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        rows.append({"method": m, "mape_pct": round(mape, 1),
                     **{f"pred_{y}": int(round(p)) for y, p in zip(holdout_years, pred)}})
    return pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)


def forecast(series: pd.Series, method: str, years=(2026, 2027)) -> dict:
    """Refit the chosen method on all full years, forecast ahead, and give a range from the
    spread of all four methods (a crude but honest 'model uncertainty' band)."""
    series = series[series.index >= FIRST_FULL_YEAR]
    point = _fit_predict(series, list(years), method)
    all_preds = np.array([_fit_predict(series, list(years), m) for m in FORECAST_METHODS])
    return {int(y): {"point": int(round(p)), "low": int(round(lo)), "high": int(round(hi))}
            for y, p, lo, hi in zip(years, point, all_preds.min(axis=0), all_preds.max(axis=0))}


# =============================================================================
# 7. CHARTS
# =============================================================================
def make_charts(results: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    BLUE, GREY, ORANGE = "#2a6fdb", "#9aa4b2", "#e8833a"
    paths = []

    # 1 National trend
    nat = results["national"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [GREY if y == 2016 else (ORANGE if y == 2024 else BLUE) for y in nat.index]
    ax.bar(nat.index, nat.values, color=colors)
    for x, v in zip(nat.index, nat.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("New DPIIT-recognised startups per year (grey = 2016 launch year, orange = only decline)")
    ax.set_ylabel("Recognitions")
    paths.append(FIG_DIR / "01_national_trend.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    # 2 Concentration
    conc = results["concentration"]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(conc["year"], conc["top5_share_pct"], marker="o", color=BLUE, label="Top-5 state share (%)")
    ax1.set_ylabel("Top-5 share (%)", color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(conc["year"], conc["states_for_80pct"], marker="s", color=ORANGE, label="States needed for 80%")
    ax2.set_ylabel("States needed to reach 80% of startups", color=ORANGE)
    ax2.grid(False)
    ax1.set_title("Is the ecosystem spreading out? Falling top-5 share + more states needed = yes")
    paths.append(FIG_DIR / "02_concentration.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    # 3 Top states stacked-ish lines
    sy = results["state_year"]
    top = sy[LAST_YEAR].sort_values(ascending=False).head(8).index
    fig, ax = plt.subplots(figsize=(9, 5))
    for st in top:
        ax.plot(range(FIRST_FULL_YEAR, LAST_YEAR + 1), sy.loc[st, FIRST_FULL_YEAR:LAST_YEAR], marker="o", ms=3, label=st)
    ax.set_title("Top 8 states (by 2025) - recognitions per year")
    ax.legend(fontsize=8, ncol=2)
    paths.append(FIG_DIR / "03_top_states.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    # 4 Segment map
    seg = results["segments"]
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"Established Leader": BLUE, "Maturing Hub": GREY, "Rising Challenger": ORANGE, "Slowing / Early": "#c9ced6"}
    for name, g in seg.groupby("segment"):
        ax.scatter(g["recent_3yr_total"], g["momentum_pp"], s=60, color=palette[name], label=name, edgecolor="white")
    for st, r in seg.iterrows():
        ax.annotate(st, (r["recent_3yr_total"], r["momentum_pp"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.axvline(results["segment_medians"]["size_median"], color="black", lw=0.6, ls="--")
    ax.axhline(results["segment_medians"]["momentum_median"], color="black", lw=0.6, ls="--")
    ax.set_xlabel("Scale: recognitions 2023-25 (log)")
    ax.set_ylabel("Momentum: CAGR 23-25 minus CAGR 19-23 (pp)")
    ax.set_title("State segments: scale vs acceleration")
    ax.legend(fontsize=8)
    paths.append(FIG_DIR / "04_state_segments.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    # 5 Industry share shift
    ind = results["industry_growth"]
    ind = ind[ind["taxonomy_flag"] == ""]
    shift = pd.concat([ind["share_change_pp"].nlargest(8), ind["share_change_pp"].nsmallest(8)]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(shift.index, shift.values, color=[ORANGE if v < 0 else BLUE for v in shift.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Sector share change, 2019 -> 2025 (percentage points of all recognitions)")
    paths.append(FIG_DIR / "05_industry_share_shift.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    # 6 AI and deep-tech trend
    iy = results["industry_year"]
    deep = ["AI", "Robotics", "Biotechnology", "Internet of Things", "Analytics", "Computer Vision"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for d in deep:
        if d in iy.index:
            ax.plot(range(2018, LAST_YEAR + 1), iy.loc[d, 2018:LAST_YEAR], marker="o", ms=3, label=d,
                    lw=2.5 if d == "AI" else 1.2)
    ax.set_title("Deep-tech sectors: AI breaks away after 2023")
    ax.legend(fontsize=8)
    paths.append(FIG_DIR / "06_deeptech.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    # 7 LQ heatmap for top states x top industries
    lq = location_quotients(results["panel"], min_count=0)
    top_states = sy[LAST_YEAR].sort_values(ascending=False).head(12).index
    top_inds = results["industry_growth"].head(14).index
    mat = lq.pivot(index="state", columns="industry", values="lq").reindex(index=top_states, columns=top_inds)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(top_inds))); ax.set_xticklabels(top_inds, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_states))); ax.set_yticklabels(top_states, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Location quotient (1 = national average)")
    ax.set_title("Specialisation: which states over-index in which sectors (2023-25)")
    ax.grid(False)
    paths.append(FIG_DIR / "07_specialisation_heatmap.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    # 8 Forecast
    fc = results["forecast"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hist = nat[nat.index >= FIRST_FULL_YEAR]
    ax.plot(hist.index, hist.values, marker="o", color=BLUE, label="Actual")
    yrs = list(fc.keys())
    ax.plot([hist.index[-1]] + yrs, [hist.values[-1]] + [fc[y]["point"] for y in yrs], marker="o", ls="--",
            color=ORANGE, label=f"Forecast ({results['best_method']})")
    ax.fill_between(yrs, [fc[y]["low"] for y in yrs], [fc[y]["high"] for y in yrs], color=ORANGE, alpha=0.2,
                    label="Range across 4 methods")
    ax.set_title("National recognitions: forecast for 2026-27 (method picked by 2023-25 holdout backtest)")
    ax.legend(fontsize=8)
    paths.append(FIG_DIR / "08_forecast.png"); fig.tight_layout(); fig.savefig(paths[-1]); plt.close(fig)

    return paths


# =============================================================================
# 8. PIPELINE
# =============================================================================
def run_all(save: bool = True, verbose: bool = True) -> dict:
    def log(*a):
        if verbose:
            print(*a)

    raw = load_raw()
    aud = audit(raw)
    panel = clean(raw)
    log(f"[1] Loaded {aud['rows']:,} raw rows -> {len(panel):,}-row zero-filled panel "
        f"({aud['states']} states x {aud['industries']} industries x {len(panel['year'].unique())} years)")

    conn = build_database(panel)
    sql = run_sql(conn)
    conn.close()
    log(f"[2] SQLite built at {DB_FILE.name}; ran {len(sql)} analytical queries")

    r = {"audit": aud, "panel": panel, "sql": sql}
    r["national"] = national_series(panel)
    r["state_year"] = state_year(panel)
    r["industry_year"] = industry_year(panel)
    r["concentration"] = concentration_over_time(panel)
    r["decentralisation"] = decentralisation_test(r["concentration"])
    r["state_growth"] = state_growth_table(panel)
    r["segments"] = classify_states(r["state_growth"])
    r["segment_medians"] = dict(r["segments"].attrs)   # kept separately; DataFrame.attrs can get lost in copies
    r["convergence"] = convergence_test(r["state_growth"])
    r["industry_growth"] = industry_growth_table(panel)
    r["ai_concentration"] = industry_concentration(panel, ["AI", "Finance Technology", "Agriculture",
                                                           "Biotechnology", "Food & Beverages"])
    r["dip_2024"] = dip_2024_breakdown(panel)
    r["lq"] = location_quotients(panel)
    r["clusters"], r["silhouette"], r["best_k"] = cluster_states(panel)
    r["hotspot_components"] = hotspot_components(panel)
    r["hotspots"] = hotspot_score(r["hotspot_components"])
    r["hotspot_sensitivity"] = hotspot_weight_sensitivity(r["hotspot_components"])
    r["forecast_backtest"] = backtest_forecasts(r["national"])
    r["best_method"] = r["forecast_backtest"].iloc[0]["method"]
    r["forecast"] = forecast(r["national"], r["best_method"])
    log("[3] All analyses complete")

    if save:
        TAB_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_csv(ROOT / "data" / "processed" / "startups_panel.csv", index=False)
        for name, df in sql.items():
            df.to_csv(TAB_DIR / f"sql_{name}.csv", index=False)
        r["concentration"].to_csv(TAB_DIR / "concentration_by_year.csv", index=False)
        r["state_growth"].to_csv(TAB_DIR / "state_growth.csv")
        r["segments"].to_csv(TAB_DIR / "state_segments.csv")
        r["industry_growth"].to_csv(TAB_DIR / "industry_growth.csv")
        r["ai_concentration"].to_csv(TAB_DIR / "sector_geographic_concentration.csv", index=False)
        r["lq"].to_csv(TAB_DIR / "location_quotients.csv", index=False)
        r["clusters"].to_csv(TAB_DIR / "state_clusters.csv")
        r["hotspots"].to_csv(TAB_DIR / "hotspots.csv", index=False)
        r["hotspot_sensitivity"].to_csv(TAB_DIR / "hotspot_weight_sensitivity.csv", index=False)
        r["forecast_backtest"].to_csv(TAB_DIR / "forecast_backtest.csv", index=False)
        figs = make_charts(r)
        log(f"[4] Saved {len(list(TAB_DIR.glob('*.csv')))} tables and {len(figs)} charts")
        (ROOT / "outputs" / "findings.md").write_text(findings_report(r), encoding="utf-8")
        log("[5] Findings written to outputs/findings.md")
    return r


def findings_report(r: dict) -> str:
    a, d, c, dip = r["audit"], r["decentralisation"], r["convergence"], r["dip_2024"]
    nat = r["national"]
    g = r["state_growth"]
    ind = r["industry_growth"]
    seg = r["segments"]
    hs = r["hotspots"].head(10)
    ai = r["ai_concentration"].set_index("industry")
    fb = r["forecast_backtest"]
    fc = r["forecast"]
    top_cagr = g.dropna(subset=["cagr_2019_2025_pct"]).sort_values("cagr_2019_2025_pct", ascending=False).head(5)
    top_abs = g.sort_values("absolute_growth", ascending=False).head(5)
    risers = ind[ind["taxonomy_flag"] == ""].sort_values("share_change_pp", ascending=False).head(5)
    fallers = ind[ind["taxonomy_flag"] == ""].sort_values("share_change_pp").head(5)

    top5_names = list(r["state_year"][LAST_YEAR].sort_values(ascending=False).head(5).index)
    checked = ai.drop(index="ALL STARTUPS")
    most_even = checked["hhi"].idxmin()

    def fmt_p(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    def seg_list(name):
        return ", ".join(seg[seg["segment"] == name].sort_values("recent_3yr_total", ascending=False).index[:6])

    lines = [
        "# India Startup Ecosystem - Findings",
        "",
        f"Data: DPIIT recognised startups, {a['years'][0]}-{a['years'][1]}, {a['rows']:,} rows, "
        f"{a['states']} states/UTs, {a['industries']} industries, {a['total_recognitions']:,} recognitions in total.",
        "Recognitions are not funding, revenue, or survival. Every 'growth' figure below means growth in new registrations.",
        "",
        f"## 1. The ecosystem grew about {nat[2025] / nat[2017]:.0f}x, with one dip",
        f"New recognitions went from {nat[2017]:,} (2017) to {nat[2025]:,} (2025). The only decline was 2024 "
        f"({dip['national_change']:+,}, {dip['national_change_pct']}%). {dip['states_declining']} of "
        f"{dip['states_considered']} sizeable states fell that year, so it was broad-based rather than one state's problem. "
        f"About {-dip['others_reclassification_effect']} of the drop came from the 'Others' category being reclassified "
        f"(a taxonomy change, not a real decline). 2025 then rebounded {dip['rebound_2025_pct']}%.",
        "",
        "## 2. It is spreading out geographically, slowly",
        f"Top-5 state share went from {d['top5_start']}% to {d['top5_end']}% and HHI from {d['hhi_start']:.0f} to "
        f"{d['hhi_end']:.0f} (Spearman rho vs year = {d['spearman_rho']}, {fmt_p(d['p_value'])}). "
        f"{', '.join(top5_names)} still dominate, but their grip is loosening.",
        "",
        "## 3. Smaller ecosystems grow faster (convergence)",
        f"Across {c['n_states']} states with a 2019 base of 50+, a bigger starting base goes with slower growth "
        f"(Spearman rho = {c['spearman_rho']}, {fmt_p(c['p_value'])}). That is partly real catch-up and partly just "
        f"small-base arithmetic, which is why growth is reported three ways:",
        "",
        "| Top 5 by CAGR 2019-25 | CAGR % | Top 5 by absolute growth | Added |",
        "|---|---|---|---|",
    ]
    for (s1, r1), (s2, r2) in zip(top_cagr.iterrows(), top_abs.iterrows()):
        lines.append(f"| {s1} | {r1['cagr_2019_2025_pct']:.1f} | {s2} | {int(r2['absolute_growth']):,} |")
    lines += [
        "",
        "## 4. State segments (scale vs acceleration)",
        f"- Established Leaders (big and accelerating): {seg_list('Established Leader')}",
        f"- Maturing Hubs (big, growth slowing): {seg_list('Maturing Hub')}",
        f"- Rising Challengers (smaller, accelerating): {seg_list('Rising Challenger')}",
        f"- Slowing / Early: {seg_list('Slowing / Early')}",
        "",
        "## 5. Sector shifts: AI is the story of 2024-25",
        f"AI went from {int(r['industry_year'].loc['AI', 2022]):,} recognitions (2022) to "
        f"{int(r['industry_year'].loc['AI', 2025]):,} (2025). Biggest share gainers 2019-25: "
        + ", ".join(f"{i} ({v:+.1f}pp)" for i, v in risers["share_change_pp"].items())
        + ". Biggest share losers: "
        + ", ".join(f"{i} ({v:+.1f}pp)" for i, v in fallers["share_change_pp"].items()) + ".",
        "",
        f"AI is also more geographically concentrated than startups overall: HHI {ai.loc['AI', 'hhi']:.0f} vs "
        f"{ai.loc['ALL STARTUPS', 'hhi']:.0f}, with {ai.loc['AI', 'top_state']} holding "
        f"{ai.loc['AI', 'top_state_share_pct']}% of AI recognitions (2023-25). Of the sectors checked, the most evenly "
        f"spread is {most_even} (HHI {ai.loc[most_even, 'hhi']:.0f}).",
        "",
        "## 6. State types (clustering on industry mix)",
        f"KMeans on each state's industry mix (states with 300+ recognitions in 2023-25) picked k = {r['best_k']} "
        f"by silhouette score ({r['silhouette']}). Silhouette scores this low mean the clusters overlap - treat "
        f"them as rough groupings, not hard categories.",
    ]
    for name, grp in r["clusters"].groupby("cluster_name"):
        lines.append(f"- {name}: {', '.join(grp.index)}")
    lines += [
        "",
        "## 7. Top 10 state x sector hotspots (equal weights)",
        "| # | State | Sector | 2023-25 count | Growth 23-25 %/yr | LQ | Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, h in hs.iterrows():
        lines.append(f"| {i + 1} | {h['state']} | {h['industry']} | {int(h['recent_3yr'])} | "
                     f"{h['growth_2023_2025_pct']:.0f} | {h['lq']:.2f} | {h['hotspot_score']:.0f} |")
    sens = r["hotspot_sensitivity"]
    lines += [
        "",
        "How much does this list depend on the weights? Overlap with the equal-weight top 20:",
        "",
    ] + [f"- {row['scheme']}: {row.iloc[1]}/20 shared, Spearman {row['spearman_vs_equal']}" for _, row in sens.iterrows()] + [
        "",
        "The top of the list is reasonably stable when weights shift moderately but changes a lot when momentum "
        "is dropped entirely, so the weights should be visible to whoever uses the ranking (the dashboard lets you move them).",
        "",
        "## 8. Forecast",
        "Each method was trained on 2017-2022 and scored on 2023-2025 without seeing those years:",
        "",
        "| Method | MAPE on 2023-25 |",
        "|---|---|",
    ] + [f"| {row['method']} | {row['mape_pct']}% |" for _, row in fb.iterrows()] + [
        "",
        f"Best: {r['best_method']}. Refit on 2017-2025, it gives {fc[2026]['point']:,} for 2026 "
        f"(range across methods {fc[2026]['low']:,}-{fc[2026]['high']:,}) and {fc[2027]['point']:,} for 2027. "
        + (f"Note the point forecast sits below 2025's actual {nat[2025]:,}: a straight line fitted through 2017-2025 "
           "underweights the 2025 jump, so the range is more useful than the point. " if fc[2026]['point'] < nat[2025] else "")
        + "With only 9 annual points and a policy-driven series, this is a rough planning number, not a prediction to rely on.",
        "",
        "## Limitations",
        "- Recognitions measure registration activity, not quality, funding or survival.",
        "- Sector labels are self-reported and the taxonomy changes (the 'Others' category collapse in 2024).",
        "- State counts reflect where a startup registered, which may differ from where it operates.",
        "- 2016 is a partial launch year and is excluded from growth maths.",
        "- Small-base states produce extreme CAGRs; a minimum base of 50 is applied, and absolute growth is shown alongside.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    results = run_all()
    print()
    print(findings_report(results))
