# utils/filters.py
from typing import List, Tuple, Optional, Dict, Union
import re
import pandas as pd
import streamlit as st
from utils.duck_backend import get_con, get_full_filtered_df  # cleaned: single import line

# ----------------------- Constants -----------------------
# NOTE: Launch Date is intentionally NOT included here so it won't appear
# as a selectable attribute in Step 1/2. We'll auto-include it in the preview.
ATTRIBUTES_JPM = [
    "HCO Type", "ATC2", "ATC3", "Company",
    "Formulation 0", "Formulation 1", "Formulation 2", "Formulation 3",
    "Gx/Non-Gx", "Brand Name", "Market", "Molecule Name", "Package Description",
    "Launch Year"
]

MEASURES_JPM = [
    "Bulk","Yen Bn","Cash Div","Cash","DOT Div","DOT","Gram Div","Gram","Patient Day","Unit"
]

# ----------------------- SQL helpers -----------------------
def _qid(col: str) -> str:
    """Quote identifier if it contains non-alnum chars (spaces, caps are fine in DuckDB)."""
    return f'"{col}"' if not col.replace("_", "").isalnum() else col

def _in(col: str, vals: List[str]) -> str:
    """SQL IN clause with safe single-quote escaping + quoted identifier."""
    if not vals:
        return ""
    escaped: List[str] = []
    for v in vals:
        s = str(v).replace("'", "''")
        escaped.append(f"'{s}'")
    return f"{_qid(col)} IN ({','.join(escaped)})"

def _where(filters: Dict[str, List[str]], horizons: Optional[List[str]]) -> str:
    parts = []
    for c, v in (filters or {}).items():
        if v:
            parts.append(_in(c, v))
    if horizons:
        parts.append(_in("Time_Horizon", horizons))
    return (" WHERE " + " AND ".join(parts)) if parts else ""

@st.cache_data(show_spinner=False)
def _distinct(view: str, where_sql: str, col: str) -> List[str]:
    con = get_con()
    sql = f"SELECT DISTINCT {_qid(col)} FROM {view} {where_sql} ORDER BY 1"
    return [r[0] for r in con.execute(sql).fetchall()]

@st.cache_data(show_spinner=False)
def _query(view: str, where_sql: str, cols: List[str], limit: int = 1000) -> pd.DataFrame:
    con = get_con()
    sel = ", ".join(_qid(c) for c in cols)
    sql = f"SELECT {sel} FROM {view} {where_sql} LIMIT {limit}"
    return con.execute(sql).df()

# ----------------------- Horizon helpers -----------------------
_rx_month = re.compile(r"^\d{4}-\d{2}$")
_rx_year  = re.compile(r"^\d{4}$")

def _h_key(s: str) -> int:
    if s is None:
        return -10**9
    ss = str(s)
    if _rx_month.match(ss):
        y, m = ss.split("-")
        return int(y) * 12 + int(m)
    if _rx_year.match(ss):
        return int(ss) * 12
    return hash(ss) & 0x7fffffff

def _sort_horizons(vals: List[str]) -> List[str]:
    return sorted(vals, key=_h_key)

def _range_inclusive(all_vals_sorted: List[str], start_val: str, end_val: str) -> List[str]:
    if not all_vals_sorted:
        return []
    ks, ke = _h_key(start_val), _h_key(end_val)
    if ks > ke:
        start_val, end_val = end_val, start_val
    started = False
    out: List[str] = []
    for v in all_vals_sorted:
        if not started and str(v) == str(start_val):
            started = True
        if started:
            out.append(v)
        if str(v) == str(end_val):
            break
    return out

# ----------------------- Safe widget defaults -----------------------
def _sanitize_default_for_widget(key: str, allowed: List[str]) -> List[str]:
    """
    Ensure st.session_state[key] only contains values present in 'allowed'.
    Returns the list of invalid values that were removed.
    Must be called BEFORE creating the widget bound to 'key'.
    """
    prev = st.session_state.get(key, [])
    if prev is None:
        prev = []
    if not isinstance(prev, list):
        prev = [prev]
    prev = [str(v) for v in prev]

    allowed_set = set(map(str, allowed))
    valid = [v for v in prev if v in allowed_set]
    invalid = [v for v in prev if v not in allowed_set]

    if invalid:
        st.session_state[key] = valid
    return invalid

# ----------------------- Main UI -----------------------
def render_filters_fast(
    view_or_df: Union[str, pd.DataFrame],
    source: str
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[List[str]]]:

    con = get_con()

    # Accept a DuckDB view/table name OR a DataFrame (register it).
    if isinstance(view_or_df, pd.DataFrame):
        tmp_view = "__filters_tmp_view"
        con.register(tmp_view, view_or_df)
        view = tmp_view
        cols = list(view_or_df.columns)
    else:
        view = view_or_df
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({view})").fetchall()]

    # Determine attributes/measures lists
    if source == "JPM":
        ALL_ATTRS, MEASURES = ATTRIBUTES_JPM, MEASURES_JPM
    else:
        schema = con.execute(f"PRAGMA table_info({view})").fetchall()
        num_types = {"DOUBLE", "BIGINT", "INTEGER", "DECIMAL", "REAL", "FLOAT"}
        numeric_cols = [name for (_cid, name, ctype, *_rest) in schema if str(ctype).upper() in num_types]
        MEASURES = numeric_cols
        ALL_ATTRS = [c for c in cols if c not in numeric_cols and c != "Time_Horizon"]

    # Session flags + canonical selection bucket
    st.session_state.setdefault("attrs_confirmed", False)
    st.session_state.setdefault("apply_filters", False)
    st.session_state.setdefault("attr_selection_final", [])  # canonical Step-1 selection

    # ---- Step 1: Attributes (confirm here only) ----
    with st.expander("****Step 1: Select Attributes****", expanded=True):
        _ = _sanitize_default_for_widget("attr_selection", ALL_ATTRS)

        default_attrs = [a for a in st.session_state.get("attr_selection_final", []) if a in ALL_ATTRS]

        col_80, col_20 = st.columns([8, 2])
        with col_80:
            attrs = st.multiselect(
                "Choose attributes to filter",
                ALL_ATTRS,
                default=default_attrs,
                key="attr_selection"
            )
        with col_20:
            if st.button("Confirm", key="confirm_attrs", use_container_width=True, type="primary"):
                if attrs:
                    attrs_clean = list(dict.fromkeys(attrs))
                    # Ensure Molecule Name and ATC3 present AT THE END if missing
                    for tail in ("Molecule Name", "ATC3"):
                        if tail in ALL_ATTRS and tail not in attrs_clean:
                            attrs_clean.append(tail)

                    st.session_state["attr_selection_final"] = attrs_clean
                    st.session_state.attrs_confirmed = True
                    st.session_state.apply_filters = False

                    keep_hier = {f"hier_{a}" for a in attrs_clean}
                    for k in list(st.session_state.keys()):
                        if k.startswith("hier_") and k not in keep_hier:
                            del st.session_state[k]

                    st.rerun()
                else:
                    st.warning("Pick at least one attribute to proceed.")

    if not st.session_state.attrs_confirmed:
        return None, None, None

    # ---- Step 2: Hierarchical Filters ----
    user_attrs = st.session_state.get("attr_selection_final", []) or []
    hier_attrs = list(user_attrs)

    with st.expander("****Step 2: Hierarchical Attribute Filters****", expanded=True):
        attr_filters: Dict[str, List[str]] = {}
        prefix_filters: Dict[str, List[str]] = {}

        c1, c2 = st.columns(2)
        half = (len(hier_attrs) + 1) // 2

        def _mk_widget(a: str, container):
            where_prefix = _where(prefix_filters, None)
            options = _distinct(view, where_prefix, a)
            options = [o for o in options if o is not None]
            options_sorted = sorted(map(str, options))
            key = f"hier_{a}"

            allowed_for_widget = ["ALL"] + options_sorted
            invalid = _sanitize_default_for_widget(key, allowed_for_widget)

            with container:
                current = st.multiselect(
                    a,
                    allowed_for_widget,
                    key=key
                )
                if invalid:
                    sample = ", ".join(invalid[:3])
                    suffix = "…" if len(invalid) > 3 else ""
                    st.caption(f"ℹ️ Removed {len(invalid)} unavailable selection(s): {sample}{suffix}")

            if "ALL" in current or not current:
                allowed = options_sorted if "ALL" in current else []
            else:
                allowed = [v for v in current if v in options_sorted]

            attr_filters[a] = allowed
            prefix_filters[a] = allowed

        for a in hier_attrs[:half]:
            _mk_widget(a, c1)
        for a in hier_attrs[half:]:
            _mk_widget(a, c2)

    # ---- Step 3: Measures ----
    with st.expander("****Step 3: Select Measures****", expanded=True):
        invalid_meas = _sanitize_default_for_widget("measures", MEASURES)
        measures: List[str] = st.multiselect("Measures", MEASURES, key="measures")
        if invalid_meas:
            sample = ", ".join(invalid_meas[:3])
            suffix = "…" if len(invalid_meas) > 3 else ""
            st.caption(f"ℹ️ Removed {len(invalid_meas)} unavailable measure(s): {sample}{suffix}")
        if not measures:
            st.warning("Please select at least one measure to proceed.")
    if not measures:
        return None, None, None

    # ---- Step 4: Time Horizon ----
    with st.expander("****Step 4: Select Time Horizon Range****", expanded=True):
        where_attr_only = _where(attr_filters, None)
        horizon_opts = _distinct(view, where_attr_only, "Time_Horizon")
        horizon_opts = [h for h in horizon_opts if h is not None]
        horizon_opts = _sort_horizons(list(map(str, horizon_opts)))

        if not horizon_opts:
            st.error("Column `Time_Horizon` not found or no values under current filters.")
            return None, None, None

        def _default_from_to():
            if "h_from" in st.session_state and "h_to" in st.session_state:
                f, t = st.session_state["h_from"], st.session_state["h_to"]
                if f in horizon_opts and t in horizon_opts:
                    return f, t
            return horizon_opts[0], horizon_opts[-1]

        def_from, def_to = _default_from_to()
        c_from, c_to = st.columns(2)
        with c_from:
            h_from = st.selectbox("From", horizon_opts,
                                  index=horizon_opts.index(def_from) if def_from in horizon_opts else 0,
                                  key="h_from")
        with c_to:
            h_to   = st.selectbox("To",   horizon_opts,
                                  index=horizon_opts.index(def_to) if def_to in horizon_opts else len(horizon_opts)-1,
                                  key="h_to")

        horizons_selected = _range_inclusive(horizon_opts, h_from, h_to)
        if not horizons_selected:
            st.warning("Please choose a valid horizon range.")
            return None, None, None

        st.session_state["horizon"] = horizons_selected

    # ---- Apply / Reset ----
    col_apply, col_reset, _ = st.columns([2, 2, 6])
    with col_apply:
        if st.button("✅ Apply Filters", key="apply_filters_btn"):
            st.session_state.apply_filters = True
            st.rerun()
    with col_reset:
        if st.button("🔄 Reset Filters", key="reset_filters_btn"):
            for key in list(st.session_state.keys()):
                if key.startswith("hier_") or key in [
                    "attrs_confirmed", "apply_filters",
                    # "attr_selection_final",  # keep if you want persistent selection
                    "measures", "horizon", "h_from", "h_to"
                ]:
                    del st.session_state[key]
            st.rerun()

    if not st.session_state.apply_filters:
        return None, None, None

    # ---- Final query (server-side) ----
    chosen_filters  = attr_filters
    chosen_measures = st.session_state.get("measures", [])
    horizons        = st.session_state.get("horizon", [])

    where_sql = _where(chosen_filters, horizons)

    # Always include Launch Date in preview if available
    base_cols = list(chosen_filters.keys())
    preview_cols = base_cols + ["Time_Horizon"] + chosen_measures
    if "Launch Date" in cols and "Launch Date" not in preview_cols:
        preview_cols.insert(0, "Launch Date")

    # De-dupe while preserving order
    seen = set()
    show_cols: List[str] = []
    for c in preview_cols:
        if c and c not in seen:
            show_cols.append(c)
            seen.add(c)

    # Preview
    df_preview = _query(view, where_sql, show_cols, limit=1000)
    st.caption("Preview limited to 1,000 rows for speed.")

    # ---- Full export (no row limit) for QC ----
    with st.expander("📥 Download full filtered data (no row limit)", expanded=False):
        st.write("This export uses the same filters and columns as the preview, without the 1,000-row cap.")
        full_df = get_full_filtered_df(view, where_sql, show_cols)
        st.download_button(
            "Download CSV (all rows)",
            data=full_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_full.csv",
            mime="text/csv"
        )
        try:
            buf = full_df.to_parquet(index=False)
            st.download_button(
                "Download Parquet (all rows)",
                data=buf,
                file_name="filtered_full.parquet",
                mime="application/octet-stream",
            )
        except Exception:
            pass

    # Persist for downstream pages
    st.session_state["selected_measures"] = chosen_measures
    st.session_state["selected_horizons"] = horizons
    st.session_state["last_where_sql"]    = where_sql
    st.session_state["last_show_cols"]    = show_cols
    st.session_state["duck_view"]         = view

    return full_df, chosen_measures, horizons

# ---------------- Friendly wrapper ----------------
def render_filters(view_or_df: Union[str, pd.DataFrame], source: str):
    return render_filters_fast(view_or_df, source)
