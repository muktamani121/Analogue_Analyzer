# utils/time_to_peak.py
import re
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------- Constants -----------------------------
# Fallback lists (only used if the incoming df doesn't carry the expected columns)
ATTRIBUTES_JPM = [
    "HCO Type","ATC2","ATC3","Company",
    "Formulation 0","Formulation 1","Formulation 2","Formulation 3",
    "Gx/Non-Gx","Brand Name","Market","Molecule Name","Package Description",
    "Launch Year","Launch Date",
]
MEASURES_JPM = [
    "Bulk","Yen Bn","Cash Div","Cash","DOT Div","DOT","Gram Div","Gram","Patient Day","Unit",
]

# Accept any of these as the ATC column (first one found will be used)
ATC_COLS = ["ATC3", "ATC3_Code_Name", "atc3_name_en"]  # includes legacy fallback

_rx_month = re.compile(r"^\d{4}[-]?\d{2}$")
_rx_year  = re.compile(r"^\d{4}$")


# ----------------------------- Helpers ------------------------------
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


def _first_present(cols: List[str], df: pd.DataFrame) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _month_diff(a: pd.Period, b: pd.Period) -> int:
    """Months from a -> b (>=0)."""
    if a is pd.NaT or b is pd.NaT:
        return 0
    return max((b.year - a.year) * 12 + (b.month - a.month), 0)


# ------------------- Uptake Curve (relative, shared basis) -------------------
def _render_uptake_curve_relative_basis(
    df: pd.DataFrame,
    measure: str,
    selected_attrs: List[str],
    basis_choice: str
):
    """
    Plot uptake on a relative timeline with either:
      - '% Market share' basis, OR
      - 'Absolute value' basis with an extra radio:
            * Absolute (raw units)
            * Normalized (0–1) via min–max scaling
    Additionally, each entity's curve is truncated at its first peak.
    """
    st.markdown("### 📌 Uptake Curve")

    granularity = st.radio(
        "Granularity for Uptake Curve",
        ("Yearly", "Monthly"),
        index=0, horizontal=True, key="uptake_granularity"
    )

    abs_mode = None
    if basis_choice == "Absolute value":
        abs_mode = st.radio(
            "Absolute mode",
            ("Absolute (raw units)", "Normalized (0–1, Min–Max)"),
            index=0, horizontal=True, key="uptake_abs_mode"
        )

    data = df.copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce").fillna(0.0)

    th = pd.to_datetime(data["Time_Horizon"], errors="coerce")
    data = data.assign(__TH=th).dropna(subset=["__TH"])
    if data.empty:
        st.info("No data after parsing Time_Horizon.")
        return

    data["Entity"] = data.apply(lambda r: _entity_label_from(r, selected_attrs), axis=1)

    if granularity == "Monthly":
        data["__P"] = data["__TH"].dt.to_period("M")
        x_title = "Months since launch (M1, M2, …)"
        prefix = "M"
        rel_calc = lambda p, l: (p.year - l.year) * 12 + (p.month - getattr(l, "month", 12)) + 1
    else:
        data["__P"] = data["__TH"].dt.to_period("A")
        x_title = "Years since launch (Y1, Y2, …)"
        prefix = "Y"
        rel_calc = lambda p, l: (p.year - l.year) + 1

    grouped = (
        data.groupby(["Entity", "__P"], dropna=False)[measure]
            .sum()
            .reset_index()
            .rename(columns={measure: "Value"})
    )
    if grouped.empty:
        st.info("No aggregated data to plot.")
        return

    if basis_choice == "% Market share":
        grouped["Total_in_Period"] = grouped.groupby("__P", dropna=False)["Value"].transform("sum")
        grouped["BasisValue"] = (grouped["Value"] / grouped["Total_in_Period"].where(grouped["Total_in_Period"] > 0, 1)) * 100.0
        y_title = "Market Share (%)"
        y_range = (0, None)
    else:
        grouped["BasisValue"] = grouped["Value"]

    rel_rows = []
    for ent, g in grouped.groupby("Entity", dropna=False):
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
            y_title_use = "Market Share (%)"
            y_range = (0, max(100, float(g["PlotY"].max()) * 1.05))
        else:
            if abs_mode == "Normalized (0–1, Min–Max)":
                seg_min = float(g["BasisValue"].min())
                seg_max = float(g["BasisValue"].max())
                denom = (seg_max - seg_min) if (seg_max - seg_min) > 0 else 1.0
                g["PlotY"] = (g["BasisValue"] - seg_min) / denom
                y_title_use = "Normalized Uptake (0–1)"
                y_range = (0, 1.05)
            else:
                g["PlotY"] = g["BasisValue"]
                y_title_use = f"Uptake ({measure})"
                y_range = (0, None)

        g["RelLabel"] = g["RelIdx"].apply(lambda i: f"{prefix}{int(i)}")
        g["Entity"] = str(ent)
        rel_rows.append(g[["Entity", "RelIdx", "RelLabel", "PlotY"]])

    if not rel_rows:
        st.info("No entities with non-zero values to plot.")
        return

    rel_df = pd.concat(rel_rows, ignore_index=True)
    rel_df["RelIdx"] = rel_df["RelIdx"].astype(int)
    rel_df = rel_df[rel_df["RelIdx"] >= 1].copy()

    fig = go.Figure()
    for ent, g in rel_df.groupby("Entity", dropna=False):
        fig.add_trace(go.Scatter(
            x=g["RelIdx"], y=g["PlotY"], mode="lines+markers", name=str(ent),
            hovertemplate=f"{x_title.split('(')[0].strip()}: {prefix}" + "%{x}<br>%{meta}: %{y:.3f}<extra></extra>",
            meta=y_title_use
        ))

    if y_range[1] is None:
        ymax = float(rel_df["PlotY"].max()) if not rel_df.empty else 0.0
        yrng = [0, ymax * 1.05 if ymax > 0 else 1]
    else:
        yrng = [y_range[0], y_range[1]]

    max_x = int(rel_df["RelIdx"].max())
    if granularity == "Monthly":
        tickvals = [1] + list(range(6, max_x + 1, 6))
        ticktext = [f"M{i}" for i in tickvals]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[1, max_x])
    else:
        tickvals = list(range(1, max_x + 1))
        ticktext = [f"Y{i}" for i in tickvals]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, range=[1, max_x])

    fig.update_layout(
        title=("Uptake Curve — % Market share (truncated at peak)"
               if basis_choice == "% Market share"
               else ("Uptake Curve — Absolute (raw, truncated at peak)"
                     if (abs_mode != "Normalized (0–1, Min–Max)")
                     else "Uptake Curve — Absolute (Normalized 0–1, truncated at peak)")),
        xaxis_title=x_title,
        yaxis_title=y_title_use,
        yaxis=dict(range=yrng),
        height=520, hovermode="x unified", margin=dict(t=60, b=90)
    )

    st.plotly_chart(fig, use_container_width=True)

    out = rel_df.sort_values(["Entity", "RelIdx"]).rename(columns={
        "RelIdx": "Relative_Index",
        "RelLabel": "Relative_Label",
        "PlotY": "Y_Value"
    })
    fname = (
        f"uptake_relative_share_{granularity.lower()}.csv"
        if basis_choice == "% Market share"
        else f"uptake_relative_abs_{granularity.lower()}.csv"
    )
    with st.expander("📂 Download Uptake Data", expanded=False):
        st.dataframe(out.head(30))
        st.download_button(
            "📥 Download CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=fname,
            mime="text/csv"
        )


# ------------------------- Time to Peak (main) -------------------------
def render_time_to_peak(df: pd.DataFrame):
    st.markdown("### ⏱️ Time to Peak Analysis")

    filter_attrs = list(st.session_state.get("attr_selection_final", [])) or \
                   [c for c in ATTRIBUTES_JPM if c in df.columns]
    # don't allow Time_Horizon / Launch Date as entity attrs
    available_attrs    = [a for a in filter_attrs if a in df.columns and a not in ("Time_Horizon", "Launch Date")]
    available_measures = [c for c in MEASURES_JPM if c in df.columns]

    if not available_attrs or not available_measures:
        st.error("Required attributes or measures not found in the filtered data.")
        return

    selected_attrs = st.multiselect(
        "Select Attribute(s) (Entities)",
        options=available_attrs,
        default=available_attrs
    )
    if not selected_attrs:
        st.error("Please select at least one attribute to define entities.")
        return

    selected_measure = st.selectbox("Select Measure (Metric)", available_measures)

    basis_choice = st.radio(
        "Value basis (applies to Time-to-Peak & Uptake)",
        ("Absolute value", "% Market share"),
        index=0, horizontal=True, key="basis_choice_global"
    )

    if "Time_Horizon" not in df.columns:
        st.error("Column `Time_Horizon` is required in the input.")
        return

    work = df.copy()
    work["_P_M"] = work["Time_Horizon"].apply(parse_month)
    work[selected_measure] = pd.to_numeric(work[selected_measure], errors="coerce").fillna(0.0)

    if "Launch Date" in work.columns:
        work["_L_M"] = work["Launch Date"].apply(parse_month)
    else:
        work["_L_M"] = pd.NaT

    work["Entity"] = work.apply(lambda r: _entity_label_from(r, selected_attrs), axis=1)

    ent_launch_map = {}
    for ent, g0 in work.groupby("Entity", dropna=False):
        lm = g0["_L_M"].dropna()
        ent_launch_map[ent] = (lm.min() if not lm.empty else g0["_P_M"].min())

    grp_cols = list(dict.fromkeys([*selected_attrs, "_P_M"]))
    agg_m = (
        work.groupby(grp_cols, dropna=False)[selected_measure]
            .sum()
            .reset_index()
            .rename(columns={selected_measure: "Value"})
    )
    agg_m["Entity"] = agg_m.apply(lambda r: _entity_label_from(r, selected_attrs), axis=1)

    if basis_choice == "% Market share":
        agg_m["Total_in_Month"] = agg_m.groupby("_P_M", dropna=False)["Value"].transform("sum")
        agg_m["BasisValue"] = (agg_m["Value"] / agg_m["Total_in_Month"].where(agg_m["Total_in_Month"] > 0, 1)) * 100.0
        ttp_value_label = "Peak value (% Share)"
    else:
        agg_m["BasisValue"] = agg_m["Value"]
        ttp_value_label = "Peak value"

    results = []
    for ent, g in agg_m.groupby("Entity", dropna=False):
        g = g.dropna(subset=["_P_M"])
        if g.empty:
            continue

        monthly = (
            g.groupby("_P_M", dropna=False)["BasisValue"]
             .sum()
             .sort_index()
        )

        vmax = float(monthly.max())
        if vmax <= 0:
            continue

        peak_pm = monthly.index[(monthly.values == vmax)].min()
        launch_pm = ent_launch_map.get(ent, None) or monthly.index.min()

        months_to_peak = _month_diff(launch_pm, peak_pm)
        years_to_peak  = round(months_to_peak / 12.0, 6)

        results.append({
            "Entity": ent,
            "Launch": f"{launch_pm.year}-{launch_pm.month:02d}",
            "Peak":   f"{peak_pm.year}-{peak_pm.month:02d}",
            "Month to peak": int(months_to_peak),
            "Year to peak":  years_to_peak,
            ttp_value_label: vmax,
        })

    if not results:
        st.info("No valid entity with data.")
        return

    ttp_df = pd.DataFrame(results).sort_values("Entity").reset_index(drop=True)

    st.session_state["ttp_df"] = ttp_df.copy()
    st.session_state["ttp_entities"] = ttp_df["Entity"].dropna().astype(str).unique().tolist()
    st.session_state["ttp_selected_attrs"] = list(selected_attrs)
    st.session_state["ttp_basis_choice"] = basis_choice
    st.session_state["ttp_measure"] = selected_measure

    with st.expander("📊 Time to Peak Table", expanded=True):
        base_cols = ["Entity", "Launch", "Peak", "Month to peak", "Year to peak", ttp_value_label]
        st.dataframe(ttp_df[base_cols], use_container_width=True)
        st.download_button(
            "📥 Download TTP CSV",
            data=ttp_df[base_cols].to_csv(index=False).encode("utf-8"),
            file_name="time_to_peak.csv",
            mime="text/csv"
        )

    with st.expander("📌 Uptake Curve (Market Dynamics over Relative Time)", expanded=False):
        _render_uptake_curve_relative_basis(df, selected_measure, selected_attrs, basis_choice)
