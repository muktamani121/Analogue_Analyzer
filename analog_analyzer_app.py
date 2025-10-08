# analogue_app_main.py
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from urllib.parse import urlparse

# ✅ This must be the FIRST and ONLY Streamlit call
st.set_page_config(page_title="Home", layout="wide", page_icon="🏠")

# ✅ All other imports AFTER
from utils.jpm_loader import load_jpm_duck  # CHANGED
from utils.filters import render_filters_fast  # CHANGED
#from utils.filters import render_filters 
from utils.time_to_peak import render_time_to_peak
from utils.gx_erosion import render_gx_erosion_timeline
from utils.iqvia_smart import render_iqvia_smart
from utils.purple_loader import render_purple_loader
from utils.first_launches import render_first_launches_atc3
from utils.formulation_advantage import render_formulation_advantage
from utils.styling import load_css, render_banner
import utils.styling as styling
from utils.styling import load_css, apply_button_color, close_button_color
from utils.attribute import scoring_page

# ---------------- Session Defaults -----------------------
DEFAULTS = {
    "page": "landing",
    "data_source": None,
    "filtered_data": None,
    "selected_measures": [],
    "selected_time_horizon": None,
    # snapshot + restore flag
    "filters_config": None,     # dict snapshot of filter UI selections
    "restore_filters": False,   # tell filter page to restore from snapshot once
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ---------------- Navigation Helpers ---------------------

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _go(page: str):
    st.session_state.page = page
    _safe_rerun()

# ---------------- Snapshot Helper ------------------------

def _snapshot_filters():
    """
    Capture current filter UI selections from st.session_state.
    Keys used in fast filters: attr_selection, hier_<attr>, measures, horizon
    and confirmation flags: attrs_confirmed, hier_confirmed, meas_confirmed, horizon_confirmed
    """
    cfg = {}
    cfg["attrs"] = st.session_state.get("attr_selection", []) or []
    cfg["measures"] = st.session_state.get("measures", []) or []
    cfg["horizon"] = st.session_state.get("horizon", []) or []
    hier = {}
    for a in cfg["attrs"]:
        hier[a] = st.session_state.get(f"hier_{a}", ["ALL"]) or ["ALL"]
    cfg["hier"] = hier

    cfg["attrs_confirmed"] = st.session_state.get("attrs_confirmed", True)
    cfg["hier_confirmed"] = st.session_state.get("hier_confirmed", True)
    cfg["meas_confirmed"] = st.session_state.get("meas_confirmed", True)
    cfg["horizon_confirmed"] = st.session_state.get("horizon_confirmed", True)
    return cfg


def _restore_filters_into_session(cfg: dict):
    """Pre-seed Streamlit widget state from a snapshot before calling filters UI."""
    if not cfg:
        return
    st.session_state["attr_selection"] = cfg.get("attrs", [])
    st.session_state["measures"] = cfg.get("measures", [])
    st.session_state["horizon"] = cfg.get("horizon", [])

    for a, vals in (cfg.get("hier") or {}).items():
        st.session_state[f"hier_{a}"] = vals

    st.session_state["attrs_confirmed"] = cfg.get("attrs_confirmed", True)
    st.session_state["hier_confirmed"] = cfg.get("hier_confirmed", True)
    st.session_state["meas_confirmed"] = cfg.get("meas_confirmed", True)
    st.session_state["horizon_confirmed"] = cfg.get("horizon_confirmed", True)

    st.session_state["prev_attrs"] = st.session_state.get("attr_selection", [])

# --------------- Landing Page ---------------------------

def landing_page():
    load_css()
    render_banner()
    st.title('Welcome to Analogue Analyzer')
    st.subheader("Seamless Data-Driven Analogue Generator")
    st.markdown('''
        <p style="color:grey">The Analogue Analyzer is a decision-support application that empowers commercial, medical and forecasting teams to learn from the past in order to shape the future. Pharmaceutical brands do not launch in a vacuum; every new therapy follows a pathway already travelled by earlier molecules, in similar indications, delivery forms or market circumstances. By surfacing those historical precedents — or analogues — we can benchmark uptake curves, pricing resilience, channel mix and competitive response, then apply those learnings to stress-test assumptions and improve the accuracy of our plans.</p>
    ''', unsafe_allow_html=True)

    sources = [
        {"key": "JPM", "color_class": "green",  "icon": "🇯🇵", "name": "JPM", "btn_label": "JPM", "desc": "JPM comprehensive monthly tracker of the Japanese pharmaceutical market, providing sales, volume, and product-level insights."},
        {"key": "Purple","color_class": "purple","icon": "🌐","name": "Purple","btn_label": "Purple","desc": "Asia-Pacific-focused dataset capturing prescription trends, digital engagement, and HCP-channel interactions."},
        {"key": "IQVIA","color_class": "blue","icon": "🌐","name": "IQVIA SMART","btn_label": "IQVIA SMART","desc": "Global dataset combining longitudinal patient-level treatment patterns with promotional intelligence."},
    ]
    cols = st.columns(3)
    for col, src in zip(cols, sources):
        with col:
            st.markdown(f"""
                <div class='data-box data-box-{src['color_class']}'>
                  <h2>{src['icon']} {src['name']}</h2>
                  <p>{src['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {src['btn_label']}", key=f"btn_{src['key']}"):
                if src['key'] == 'IQVIA':
                    # Delegate to IQVIA module to show its own warning/placeholder
                    render_iqvia_smart(None)
                elif src['key'] == 'Purple':
                    # Delegate to Purple module to show its own warning/placeholder
                    render_purple_loader(None)
                else:
                    st.session_state.data_source = src['key']
                    _go('filter')
                    _safe_rerun()

# --------------- Filter Page ----------------------------

def filter_page():
    load_css()
    st.markdown(f"## 🎯 Parameter Selection")

    if st.session_state.data_source == 'JPM':
        view = load_jpm_duck()  # prepares DuckDB view over Parquet
        st.session_state["duck_view"] = view
        df = None  # not used in fast path
    elif st.session_state.data_source == 'Purple':
        df = load_purple_data()  # existing loader
    else:
        st.error("Unknown data source")
        return

    # Restore previous selections (if coming back from Analogue)
    if st.session_state.get("restore_filters") and st.session_state.get("filters_config"):
        _restore_filters_into_session(st.session_state["filters_config"])
        st.session_state["restore_filters"] = False

    # Drive filters
    if st.session_state.data_source == 'JPM':
        filtered_df, measures, horizons = render_filters_fast(
            st.session_state["duck_view"], st.session_state.data_source
        )
    else:
        from utils.filters import render_filters  # your original function for Purple
        filtered_df, measures, horizons = render_filters(df, st.session_state.data_source)

    if filtered_df is not None:
        st.session_state.filtered_data = filtered_df  # preview-sized df
        st.session_state.selected_measures = measures
        st.session_state.selected_time_horizon = horizons
        st.session_state["filters_config"] = _snapshot_filters()

        st.success(f"{len(filtered_df):,} rows (preview)")
        with st.expander("Show Sample Filtered Data"):
            st.dataframe(filtered_df.head(50))

    # ---------------- Navigation Buttons ----------------
    st.markdown('---')
    col1, spacer, col3 = st.columns([3, 4, 3])

    with col1:
        styling.apply_button_color("red")
        if st.button("Back: Landing Page", key="back", type="primary", use_container_width=True):
            _go('landing')
            _safe_rerun()
        styling.close_button_color()

    with col3:
        styling.apply_button_color("green")
        if st.button(
            "Next: Analogue Templates",
            key="next",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.filtered_data is None
        ):
            _go('analog')
            _safe_rerun()
        styling.close_button_color()

# --------------- Analogue Page --------------------------

def analog_page():
    load_css()
    st.markdown("### Analogue Templates")
    templates = ["Time-to-Peak", "Gx Erosion Timeline", "First Launches in ATC3", "Formulation Advantage"]
    choice = st.selectbox("Choose template:", templates)

    if st.session_state.filtered_data is None:
        st.error("Please complete filtering first.")
        return

    df = st.session_state.filtered_data

    if choice == "Time-to-Peak":
        render_time_to_peak(df)
    elif choice == "Gx Erosion Timeline":
        render_gx_erosion_timeline(df)
    elif choice == "First Launches in ATC3":
        render_first_launches_atc3(df)
    elif choice == "Formulation Advantage":
        render_formulation_advantage(df)
    else:
        st.warning("Template not implemented yet.")

    st.markdown('---')
    col_left, _, col_right = st.columns([3, 4, 3])

    with col_left:
        styling.apply_button_color("red")
        if st.button("Back to Filter Selection", key="back_to_filter", type="primary", use_container_width=True):
            st.session_state["restore_filters"] = True
            _go('filter')
            _safe_rerun()
        styling.close_button_color()

    with col_right:
        styling.apply_button_color("green")
        if st.button(
            "Next: Attribute Analysis",
            key="next_attr",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.filtered_data is None
        ):
            _go('score')
            _safe_rerun()

# ---------------- Page Router ---------------------------
page_router = {
    "landing": landing_page,
    "filter": filter_page,
    "analog": analog_page,
    "score":   scoring_page,
}
page_router[st.session_state.page]()