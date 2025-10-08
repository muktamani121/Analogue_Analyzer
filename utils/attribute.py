# utils/attribute.py
import re
from typing import List, Tuple
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.styling import load_css
from utils.styling import apply_button_color, close_button_color  # optional color helpers

# ---- Sub-criteria lists ----
MARKET_SUBS = ["Therapy area dynamics", "Nature of treatment"]
COMP_SUBS   = ["Level of competition", "Order of entry"]
PROD_SUBS   = ["Efficacy", "Safety/tolerability", "Route of administration", "Mechanism of action"]

# ---------------- Helpers for periods / labels ----------------
_rx_month = re.compile(r"^\d{4}[-]?\d{2}$")
_rx_year  = re.compile(r"^\d{4}$")

def parse_month(val) -> pd.Period:
    """Parse YYYY or YYYY-MM (or datetimes/strings) into a monthly Period ('YYYY-MM')."""
    if pd.isna(val):
        return pd.NaT
    s = str(val)
    if _rx_month.match(s):
        fmt = s if "-" in s else f"{s[:4]}-{s[4:]}"
        return pd.Period(fmt, freq="M")
    if _rx_year.match(s):
        return pd.Period(s, freq="A").asfreq("M", "end")
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    return pd.Period(dt, freq="M")

def _entity_label_from(row: pd.Series, attrs: List[str]) -> str:
    parts = []
    for a in attrs:
        v = row.get(a, None)
        parts.append("" if pd.isna(v) else str(v))
    return " | ".join(parts)

# ==============================================================
# 🔹 BEST-FIT HELPERS (kept for reference / optional use)
# ==============================================================

def _best_fit_curve_ms(score_df: pd.DataFrame, entities: List[str]) -> pd.Series:
    """Market-share best-fit per sub-criterion row (returns percent)."""
    df = score_df.copy()
    df = df[["Category", "Subcriterion"] + entities].copy()
    for e in entities:
        df[e] = pd.to_numeric(df[e], errors="coerce").fillna(0.0)
    entity_totals = df[entities].sum()
    grand_total = float(entity_totals.sum()) if float(entity_totals.sum()) != 0 else 1.0
    row_sum = df[entities].sum(axis=1).replace(0, 1.0)
    for e in entities:
        df[f"{e}_RowShare"] = df[e] / row_sum
    for e in entities:
        weight = float(entity_totals[e]) / grand_total
        df[f"{e}_Attr"] = df[f"{e}_RowShare"] * weight
    best_fit = df[[f"{e}_Attr" for e in entities]].sum(axis=1)
    return best_fit * 100.0

def _best_fit_curve_abs(score_df: pd.DataFrame, entities: List[str]) -> pd.Series:
    """Absolute best-fit per sub-criterion row (returns percent; divide by 100 for raw)."""
    df = score_df.copy()
    df = df[["Category", "Subcriterion"] + entities].copy()
    for e in entities:
        df[e] = pd.to_numeric(df[e], errors="coerce").fillna(0.0)
    entity_totals = df[entities].sum()
    grand_total = float(entity_totals.sum()) if float(entity_totals.sum()) != 0 else 1.0
    for e in entities:
        denom = float(entity_totals[e]) if entity_totals[e] > 0 else 1.0
        df[f"{e}_ColShare"] = df[e] / denom
        df[f"{e}_Attr"] = df[f"{e}_ColShare"] * (float(entity_totals[e]) / grand_total)
    best_fit = df[[f"{e}_Attr" for e in entities]].sum(axis=1)
    return best_fit * 100.0

# ==============================================================
# 🔎 QC — Weightage Calculation (Excel logic)
# ==============================================================

def _qc_weightage_calc_detail(
    score_df: pd.DataFrame,
    entities: List[str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Excel-style weight building:

      Row contribution (per entity)   = Score * (Weight_% / 100)
      WeightedSum (per entity)        = sum(Row contribution)
      WeightShare (per entity, sum=1) = WeightedSum / sum(WeightedSum)

    Returns:
      row_qc_df: per-row details with each "Weighted <entity>" column
      weights  : Series of shares (sum=1.0) indexed by entity
      summary  : table with WeightedSum and WeightShare_% per entity
    """
    if score_df is None or not entities:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    # Drop any Σ rows robustly if they exist
    df = score_df.copy()
    if "Subcriterion" in df.columns:
        df = df[~df["Subcriterion"].astype(str).str.contains("Σ Total", na=False)].copy()

    cols = ["Category", "Subcriterion", "Weight_%"] + entities
    df = df[cols].copy()

    # numerics
    df["Weight_%"] = pd.to_numeric(df["Weight_%"], errors="coerce").fillna(0.0)
    for e in entities:
        df[e] = pd.to_numeric(df[e], errors="coerce").fillna(0.0)

    # per-row weighted values
    row_qc = df[["Category", "Subcriterion", "Weight_%"]].copy()
    for e in entities:
        row_qc[f"Weighted {e}"] = df[e] * (df["Weight_%"] / 100.0)

    # totals & shares
    wsum = {e: float(row_qc[f"Weighted {e}"].sum()) for e in entities}
    s = pd.Series(wsum, dtype=float)
    s_total = float(s.sum())
    if s_total <= 0:
        weights = pd.Series({e: (1.0 / len(entities) if entities else 0.0) for e in entities}, dtype=float)
    else:
        weights = s / s_total

    summary = pd.DataFrame({
        "Entity": entities,
        "WeightedSum": [wsum[e] for e in entities],
        "WeightShare_%": [weights[e] * 100.0 for e in entities],
    })

    return row_qc, weights, summary

# ---------------- Matrix helpers ----------------
def _coerce_numeric_matrix(df: pd.DataFrame, entity_cols: List[str]) -> pd.DataFrame:
    """Coerce Weight_% to numeric; entity scores to 0..10."""
    df = df.copy()
    if "Weight_%" in df.columns:
        df["Weight_%"] = pd.to_numeric(df["Weight_%"], errors="coerce").fillna(0.0)
    for c in entity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0, upper=10)
    return df

# ========= Attribute Scoring Helpers (retained; not shown in UI) =========
def _attribute_base_uptake(
    score_df: pd.DataFrame, entity_cols: List[str], method: str = "market_share"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kept for potential internal use."""
    df = score_df.copy()
    if "Subcriterion" in df.columns:
        df = df[~df["Subcriterion"].astype(str).str.contains("Σ Total", na=False)].copy()

    cols_needed = ["Category", "Subcriterion"] + entity_cols
    df = df[cols_needed].copy()

    for c in entity_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    entity_totals = df[entity_cols].sum(axis=0)
    grand_total = float(entity_totals.sum())

    if grand_total <= 0:
        rows_df = df[["Category", "Subcriterion"]].assign(**{"Base_Uptake": 0.0, "Base_Uptake_%": 0.0})
        entities_df = pd.DataFrame({
            "Entity": entity_cols,
            "EntityTotal": 0.0,
            "EntityWeight": 0.0,
            "Total_AttrCalc": 0.0,
            "Total_AttrCalc_%": 0.0
        })
        return rows_df, entities_df

    entity_weight = (entity_totals / grand_total).astype(float)

    work = df.copy()
    attrcalc_cols = []
    if method == "market_share":
        for e in entity_cols:
            denom = float(entity_totals[e]) if entity_totals[e] > 0 else 1.0
            work[f"{e}__MarketShare"] = work[e] / denom
            work[f"{e}__AttrCalc"]    = work[f"{e}__MarketShare"] * float(entity_weight[e])
            attrcalc_cols.append(f"{e}__AttrCalc")

        work["Base_Uptake"]   = work[attrcalc_cols].sum(axis=1)
        work["Base_Uptake_%"] = work["Base_Uptake"] * 100.0

        ent_totals_calc = pd.Series({e: float(work[f"{e}__AttrCalc"].sum()) for e in entity_cols})
        ent_total_sum = float(ent_totals_calc.sum())
        ent_pct = (ent_totals_calc / ent_total_sum * 100.0) if ent_total_sum > 0 else ent_totals_calc * 0.0

    else:  # absolute
        for e in entity_cols:
            work[f"{e}__AttrCalc"] = work[e] * float(entity_weight[e])
            attrcalc_cols.append(f"{e}__AttrCalc")

        work["Base_Uptake"] = work[attrcalc_cols].sum(axis=1)
        total_bu = float(work["Base_Uptake"].sum())
        work["Base_Uptake_%"] = (work["Base_Uptake"] / total_bu * 100.0) if total_bu > 0 else 0.0

        ent_totals_calc = pd.Series({e: float(work[f"{e}__AttrCalc"].sum()) for e in entity_cols})
        ent_total_sum = float(ent_totals_calc.sum())
        ent_pct = (ent_totals_calc / ent_total_sum * 100.0) if ent_total_sum > 0 else ent_totals_calc * 0.0

    rows_df = work[["Category", "Subcriterion", "Base_Uptake", "Base_Uptake_%"]].copy()
    entities_df = pd.DataFrame({
        "Entity": entity_cols,
        "EntityTotal": entity_totals.astype(float).values,
        "EntityWeight": entity_weight.values,
        "Total_AttrCalc": ent_totals_calc.values,
        "Total_AttrCalc_%": ent_pct.values
    })
    return rows_df, entities_df

# ---------------- Uptake curve (Excel best-fit + full QC) ----------------
def _render_uptake_curve_relative_basis(
    df_full: pd.DataFrame,
    ttp_attrs: List[str],
    entities_to_plot: List[str],
    selected_measure: str
):
    """
    Relative uptake curve renderer with Excel-style Best-fit:
      - Entity shares from Score Matrix: sum(Score * Weight_%/100) -> normalize
      - BestFit(k) = sum_e share_e * Value_e(k)
      - For plotting: use % when basis is "% Market share", otherwise raw/normalized value.
      - Gaps if any entity missing at a period (strict Excel behaviour).
      - QC expander outputs four separate dataframes.
    """
    # ---- UI controls ----
    basis_choice = st.radio(
        "Value basis",
        ("Absolute value", "% Market share"),
        index=0, horizontal=True, key="attr_basis_choice"
    )

    abs_mode = None
    if basis_choice == "Absolute value":
        abs_mode = st.radio(
            "Absolute mode",
            ("Absolute (raw units)", "Normalized (0–1, Min–Max)"),
            index=0, horizontal=True, key="attr_abs_mode"
        )

    granularity = st.radio(
        "Granularity for Uptake Curve",
        ("Yearly", "Monthly"),
        index=0, horizontal=True, key="attr_uptake_granularity"
    )
    prefix = "M" if granularity == "Monthly" else "Y"

    # ---- Entities (composite key) ----
    if not ttp_attrs or not all(a in df_full.columns for a in ttp_attrs):
        st.error("Missing the attributes used to form entities (from Time to Peak page).")
        return

    data = df_full.copy()
    data["__EntityKey"] = data[ttp_attrs].astype(str).agg(" | ".join, axis=1)
    data = data[data["__EntityKey"].isin(entities_to_plot)].copy()

    # ---- Time parsing ----
    if "Time_Horizon" not in data.columns:
        st.error("Column `Time_Horizon` is required to build uptake curves.")
        return
    th = pd.to_datetime(data["Time_Horizon"], errors="coerce")
    data = data.assign(__TH=th).dropna(subset=["__TH"])
    if data.empty:
        st.info("No data after parsing Time_Horizon.")
        return

    # ---- Measure ----
    data[selected_measure] = pd.to_numeric(data[selected_measure], errors="coerce").fillna(0.0)

    # ---- Grain ----
    if granularity == "Monthly":
        data["__P"] = data["__TH"].dt.to_period("M")
        rel_calc = lambda p, l: (p.year - l.year) * 12 + (p.month - getattr(l, "month", 12)) + 1
        x_title = "Months since launch (M1, M2, …)"
    else:
        data["__P"] = data["__TH"].dt.to_period("A")
        rel_calc = lambda p, l: (p.year - l.year) + 1
        x_title = "Years since launch (Y1, Y2, …)"

    # ---- Aggregate value per (entity, period) ----
    grouped = (
        data.groupby(["__EntityKey", "__P"], dropna=False)[selected_measure]
            .sum()
            .reset_index()
            .rename(columns={selected_measure: "Value"})
    )
    if grouped.empty:
        st.info("No aggregated data to plot.")
        return

    # ---- Basis transform ----
    if basis_choice == "% Market share":
        grouped["Total_in_Period"] = grouped.groupby("__P", dropna=False)["Value"].transform("sum")
        grouped["BasisValue"] = (grouped["Value"] /
                                 grouped["Total_in_Period"].where(grouped["Total_in_Period"] > 0, 1)) * 100.0
        yaxis_title = "Market Share (%)"
    else:
        grouped["BasisValue"] = grouped["Value"]
        yaxis_title = None  # resolved later

    # ---- Build relative timeline per entity, truncate at first peak (keeps TTP behaviour) ----
    rel_rows = []
    for ent, g in grouped.groupby("__EntityKey", dropna=False):
        g = g.sort_values("__P")
        nz = g[g["BasisValue"] > 0]
        if nz.empty:
            continue
        launch_p = nz["__P"].iloc[0]
        g = g.assign(RelIdx=g["__P"].apply(lambda p: rel_calc(p, launch_p)))
        g = g[g["RelIdx"] >= 1].copy()
        if g.empty:
            continue
        vmax = float(g["BasisValue"].max())
        if vmax <= 0:
            continue
        peak_rel = int(g.loc[g["BasisValue"] == vmax, "RelIdx"].min())
        g = g[g["RelIdx"] <= peak_rel].copy()
        if g.empty:
            continue

        if basis_choice == "% Market share":
            g["PlotY"] = g["BasisValue"]
        else:
            if abs_mode == "Normalized (0–1, Min–Max)":
                seg_min = float(g["BasisValue"].min())
                seg_max = float(g["BasisValue"].max())
                denom = (seg_max - seg_min) if (seg_max - seg_min) > 0 else 1.0
                g["PlotY"] = (g["BasisValue"] - seg_min) / denom
                yaxis_title = "Normalized Uptake (0–1)"
            else:
                g["PlotY"] = g["BasisValue"]
                yaxis_title = f"Uptake ({selected_measure})"

        g["Entity"] = str(ent)
        rel_rows.append(g[["Entity", "__P", "RelIdx", "BasisValue", "PlotY"]])

    if not rel_rows:
        st.info("No entities with non-zero values to plot.")
        return

    rel_df = pd.concat(rel_rows, ignore_index=True)
    rel_df["RelIdx"] = rel_df["RelIdx"].astype(int)
    rel_df = rel_df[rel_df["RelIdx"] >= 1].copy()

    # ---- Data after applying filters (QC table 1) ----
    piv_val = rel_df.pivot_table(index="RelIdx", columns="Entity", values="PlotY", aggfunc="first")
    data_after_df = piv_val.copy()
    data_after_df.index.name = "Relative_Index"
    data_after_df = data_after_df.reset_index()
    data_after_df["Relative_Label"] = data_after_df["Relative_Index"].apply(lambda i: f"{prefix}{int(i)}")
    data_after_df = data_after_df[["Relative_Index", "Relative_Label"] + [c for c in piv_val.columns]]

    # ---- Weightage Calculation (QC table 2) ----
    score_df = st.session_state.get("attr_matrix_raw")
    row_qc_df, weights, summary_df = _qc_weightage_calc_detail(score_df, list(piv_val.columns))

    # ---- Attributed Data (QC table 3) — fixed weights, no renormalization (Excel strict) ----
    attrib_piv = pd.DataFrame(index=piv_val.index)
    for e in piv_val.columns:
        attrib_piv[f"Attributed {e}"] = piv_val[e] * float(weights.get(e, 0.0))

    # ---- Best Fit graph table (QC table 4) ----
    has_all = ~piv_val.isna().any(axis=1)  # blank if any entity missing at a period
    bestfit_curve = attrib_piv.sum(axis=1).where(has_all, pd.NA)

    bestfit_table = data_after_df.copy()
    bestfit_table["Best Fit Uptake Curve"] = bestfit_curve.values
    cols = ["Relative_Index", "Relative_Label", "Best Fit Uptake Curve"] + [c for c in piv_val.columns]
    bestfit_table = bestfit_table[cols]

    # ---- Plot: all entities + Best-fit line ----
    fig = go.Figure()
    for ent, g in rel_df.groupby("Entity", dropna=False):
        fig.add_trace(go.Scatter(
            x=g["RelIdx"], y=g["PlotY"], mode="lines+markers", name=str(ent),
            hovertemplate=f"{x_title.split('(')[0].strip()}: " + "%{x}<br>%{meta}: %{y:.3f}<extra></extra>",
            meta=yaxis_title or ""
        ))

    fig.add_trace(go.Scatter(
        x=bestfit_curve.index.astype(int),
        y=bestfit_curve.astype(float),  # NaNs produce gaps, matching Excel IF(COUNT(...))
        mode="lines",
        name="Best-fit uptake curve",
        line=dict(dash="dash"),
        hovertemplate=f"{x_title.split('(')[0].strip()}: " + "%{x}<br>Best-fit: %{y:.3f}<extra></extra>"
    ))

    # ---- Axes & layout ----
    ymax = float(pd.concat([rel_df["PlotY"], bestfit_curve.rename("BestFit")]).max())
    if basis_choice == "% Market share":
        y_range = [0, max(100.0, ymax * 1.05 if ymax > 0 else 1.0)]
    else:
        y_range = [0, 1.05] if abs_mode == "Normalized (0–1, Min–Max)" else [0, ymax * 1.1 if ymax > 0 else 1.0]

    max_x = int(rel_df["RelIdx"].max())
    if granularity == "Monthly":
        tickvals = [1, 6] + list(range(12, max_x + 1, 6)) if max_x > 60 else list(range(1, max_x + 1))
        ticktext = [f"M{i}" for i in tickvals]
    else:
        tickvals = list(range(1, max_x + 1))
        ticktext = [f"Y{i}" for i in tickvals]

    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[1, max_x])
    fig.update_layout(
        title=("Uptake — % Market share (truncated at peak)"
               if basis_choice == "% Market share"
               else ("Uptake — Absolute (raw, truncated at peak)"
                     if abs_mode != "Normalized (0–1, Min–Max)"
                     else "Uptake — Absolute (Normalized 0–1, truncated at peak)")),
        xaxis_title=x_title,
        yaxis_title=(yaxis_title or ""),
        yaxis=dict(range=y_range),
        height=520, hovermode="x unified", margin=dict(t=60, b=90)
    )
    st.plotly_chart(fig, use_container_width=True)

    #---- QC expander: four separate dataframes with downloads ----
    with st.expander("🧪 QC — step-by-step tables", expanded=False):
        # 1) Data after filters
        st.markdown("**1) Data after applying filters (values used for plotting)**")
        st.dataframe(data_after_df, use_container_width=True)
        st.download_button(
            "📥 Download (CSV) — Data after filters",
            data=data_after_df.to_csv(index=False).encode("utf-8"),
            file_name="qc_data_after_filters.csv",
            mime="text/csv"
        )

        # 2) Weightage Calculation (rows + summary)
        st.markdown("---\n**2) Weightage Calculation (row weights × scores)**")
        st.dataframe(row_qc_df, use_container_width=True)
        st.download_button(
            "📥 Download (CSV) — Weightage Calculation (rows)",
            data=row_qc_df.to_csv(index=False).encode("utf-8"),
            file_name="qc_weightage_calculation_rows.csv",
            mime="text/csv"
        )
        st.markdown("**Weight shares (summary)**")
        st.dataframe(summary_df, use_container_width=True)
        st.download_button(
            "📥 Download (CSV) — Weight shares summary",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="qc_weightage_summary.csv",
            mime="text/csv"
        )

        # 3) Attributed Data
        st.markdown("---\n**3) Attributed Data (per period: weight × value)**")
        attrib_out = attrib_piv.copy()
        attrib_out.index.name = "Relative_Index"
        attrib_out = attrib_out.reset_index()
        attrib_out["Relative_Label"] = attrib_out["Relative_Index"].apply(lambda i: f"{prefix}{int(i)}")
        attrib_out = attrib_out[["Relative_Index", "Relative_Label"] + [c for c in attrib_piv.columns]]
        st.dataframe(attrib_out, use_container_width=True)
        st.download_button(
            "📥 Download (CSV) — Attributed Data",
            data=attrib_out.to_csv(index=False).encode("utf-8"),
            file_name="qc_attributed_data.csv",
            mime="text/csv"
        )

        # 4) Best Fit graph table
        st.markdown("---\n**4) Best Fit (graph table)**")
        st.dataframe(bestfit_table, use_container_width=True)
        st.download_button(
            "📥 Download (CSV) — Best Fit Table",
            data=bestfit_table.to_csv(index=False).encode("utf-8"),
            file_name="qc_bestfit_graph_table.csv",
            mime="text/csv"
        )

    # ---- Download plotted entity series (unchanged) ----
    out = rel_df.sort_values(["Entity", "RelIdx"]).rename(columns={
        "RelIdx": "Relative_Index",
        "PlotY": "Y_Value"
    })
    with st.expander("📂 Download Uptake Data", expanded=False):
        st.dataframe(out.head(30), use_container_width=True)
        tag = "share" if basis_choice == "% Market share" else ("abs_norm" if abs_mode == "Normalized (0–1, Min–Max)" else "abs_raw")
        st.download_button(
            "📥 Download CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"uptake_relative_{tag}_{granularity.lower()}.csv",
            mime="text/csv"
        )

# ========================= Main Page =========================
def scoring_page():
    load_css()
    st.markdown("## 🧮 Composite Scoring")

    # ====== 0) Pull filtered dataset & TTP entities ======
    if st.session_state.get("filtered_data") is None:
        st.error("No filtered dataset found. Please complete the filter step first.")
        return
    df_full = st.session_state.filtered_data

    ttp_entities = list(st.session_state.get("ttp_entities", []))
    ttp_attrs    = list(st.session_state.get("ttp_selected_attrs", []))

    if not ttp_entities or not ttp_attrs:
        st.error("No entities found from the Time to Peak page. Please run that page first.")
        return

    # ====== 1) Sub-criteria ======
    st.subheader("Select Sub-criteria per Category")
    st.caption("Default weight = **12.5%** per selected sub-criterion. You can edit weights and entity scores below.")

    cA, cB, cC = st.columns(3)
    with cA:
        sel_market = st.multiselect("Market Criteria", MARKET_SUBS, default=MARKET_SUBS, key="sel_market_attr")
    with cB:
        sel_comp   = st.multiselect("Competitive Landscape", COMP_SUBS, default=COMP_SUBS, key="sel_comp_attr")
    with cC:
        sel_prod   = st.multiselect("Product Criteria", PROD_SUBS, default=PROD_SUBS, key="sel_prod_attr")

    if not sel_market and not sel_comp and not sel_prod:
        st.info("Please select at least one sub-criterion above.")
        return

    # ====== 2) Entity picker (columns in matrix) ======
    st.subheader("Entities Included in Score Matrix")
    default_entity_cols = ttp_entities[:]
    chosen_entity_cols = st.multiselect(
        "Select entities (columns). All are selected by default.",
        options=ttp_entities,
        default=default_entity_cols,
        key="attr_entities_in_matrix"
    )
    if not chosen_entity_cols:
        st.info("Select at least one entity to proceed.")
        return

    # ====== 3) Build the matrix ======
    rows = []
    def _add_rows(cat, subs):
        for s in subs:
            row = {"Category": cat, "Subcriterion": s, "Weight_%": 12.5}
            for ent in chosen_entity_cols:
                row[ent] = 5.0  # default score; editable by user
            rows.append(row)

    _add_rows("Market",      sel_market)
    _add_rows("Competitive", sel_comp)
    _add_rows("Product",     sel_prod)

    base = pd.DataFrame(rows)

    # Persist user edits across interactions with a signature
    sig = ("|".join(sel_market), "|".join(sel_comp), "|".join(sel_prod), "|".join(chosen_entity_cols))
    if st.session_state.get("attr_matrix_signature") != sig or "attr_matrix_raw" not in st.session_state:
        st.session_state["attr_matrix_signature"] = sig
        st.session_state["attr_matrix_raw"] = base[["Category","Subcriterion","Weight_%"] + chosen_entity_cols].copy()

    # 🚫 Ensure no legacy Σ rows exist in state
    st.session_state["attr_matrix_raw"] = st.session_state["attr_matrix_raw"][
        ~st.session_state["attr_matrix_raw"]["Subcriterion"].astype(str).str.contains("Σ Total", na=False)
    ].copy()

    # Coerce (no Σ row in UI)
    working = _coerce_numeric_matrix(st.session_state["attr_matrix_raw"], chosen_entity_cols)
    show_df = working.copy()

    # ====== 4) Editor (weights + entity columns editable) ======
    st.subheader("Score Matrix")
    st.caption(" Score scale range from  1 to 10 .we are scoring all Analogue on a scale of 10 with default  set to  5.")
    edited = st.data_editor(
        show_df[["Category","Subcriterion","Weight_%"] + chosen_entity_cols],
        use_container_width=True,
        num_rows="fixed",
        key="score_matrix_editor_attr",
        disabled=["Category","Subcriterion"]
    )

    # Persist exactly what's in the grid
    st.session_state["attr_matrix_raw"] = _coerce_numeric_matrix(edited.copy(), chosen_entity_cols)

    # Current sum + normalize
    w_sum = float(st.session_state["attr_matrix_raw"]["Weight_%"].sum())
    c1, c2, _ = st.columns([2,3,5])
    with c1:
        st.write(f"**Current Weight Sum:** {w_sum:.2f}%")
    with c2:
        if st.button("🔁 Normalize Weight_% to 100", key="normalize_weights_attr", use_container_width=True):
            if w_sum > 0:
                tmp = st.session_state["attr_matrix_raw"].copy()
                tmp["Weight_%"] = tmp["Weight_%"] * (100.0 / w_sum)
                st.session_state["attr_matrix_raw"] = tmp
                st.rerun()
            else:
                st.warning("All weights are zero — nothing to normalize.")

    st.download_button(
        "📥 Download Edited Matrix (CSV)",
        data=st.session_state["attr_matrix_raw"].to_csv(index=False).encode("utf-8"),
        file_name="attribute_matrix.csv",
        mime="text/csv"
    )

    # ====== 5) Uptake Curves (relative, with Excel Best-fit & QC) ======
    st.markdown("---")
    st.subheader("📌Uptake Curves")

    # Prefer new display measure names; exclude 'Launch Year' (string column)
    numeric_cols = [c for c in df_full.select_dtypes("number").columns if c not in ["Launch Year"]]
    # Prioritize your new display metrics first
    pref_measures = ["Unit","Cash","Gram","Patient Day","DOT","Yen Bn","Bulk","Cash Div","DOT Div","Gram Div"]
    preferred       = [c for c in pref_measures if c in numeric_cols]
    measure_choices = preferred + [c for c in numeric_cols if c not in preferred]
    if not measure_choices:
        st.error("No numeric measures found to plot uptake.")
        return
    selected_measure = st.selectbox("Sales / Measure column", measure_choices, index=0, key="attr_uptake_measure")

    _render_uptake_curve_relative_basis(
        df_full=df_full,
        ttp_attrs=ttp_attrs,
        entities_to_plot=chosen_entity_cols,
        selected_measure=selected_measure
    )

    # ====== 6) Navigation ======
    st.markdown("---")
    c_left, _, c_right = st.columns([3,4,3])
    with c_left:
        try: apply_button_color("red")
        except Exception: pass
        if st.button("⬅️ Back to Analogue Templates", key="score_back_to_analog", use_container_width=True, type="primary"):
            st.session_state["page"] = "analog"
            if hasattr(st, "rerun"): st.rerun()
            else: st.experimental_rerun()
        try: close_button_color()
        except Exception: pass
    with c_right:
        try: apply_button_color("green")
        except Exception: pass
        if st.button("⬅️⬅️ Back to Filter Selection", key="score_back_to_filter", use_container_width=True, type="primary"):
            st.session_state["restore_filters"] = True
            st.session_state["page"] = "filter"
            if hasattr(st, "rerun"): st.rerun()
            else: st.experimental_rerun()
        try: close_button_color()
        except Exception: pass
