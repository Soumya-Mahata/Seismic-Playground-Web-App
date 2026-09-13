"""
app.py — Seismic Processing & Interpretation Platform
=========================================================
Single-file entry point, mirroring the architecture of the reference
Petrophysics app: page modules are imported and called here, all
persistent state lives in `st.session_state` (via `utils.session_state`),
and navigation is a sidebar `st.radio` rather than Streamlit's native
multipage `pages/` folder — deliberately, so a single custom nav control
stays in charge of the workflow order instead of Streamlit auto-generating
a second, conflicting page list (see README "Architecture notes").

Run:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

import config as cfg
from modules.pipeline import PIPELINE_STAGES as _pipeline_stages
from utils import session_state as ss
from views import (attributes_view, data_management, export_view, home,
                    interpretation_view, migration_view, poststack_view,
                    preprocessing_view, stacking_view, velocity_nmo_view)

st.set_page_config(page_title=cfg.APP_TITLE, page_icon=cfg.APP_ICON,
                    layout="wide", initial_sidebar_state="expanded")

# Light-touch styling only — no hard-coded background/text colors, so the
# app stays legible in both Streamlit light and dark themes. Plotly
# figures are separately forced to a consistent white background inside
# visualization/theme.py, which is the right layer for that (scientific
# plots read best with consistent contrast; the app chrome around them
# should still follow the person's chosen theme).
st.markdown("""
<style>
    .stButton>button {border-radius:6px; font-weight:600;}
    .stTabs [data-baseweb="tab"] {font-size:14px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

ss.init_session_state()

PAGES = {
    "🏠 Home": home.render,
    "📂 Data Management": data_management.render,
    "⚙️ Pre-Stack Conditioning": preprocessing_view.render,
    "🎯 Velocity, NMO & DMO": velocity_nmo_view.render,
    "📚 Stacking": stacking_view.render,
    "🧹 Post-Stack Processing": poststack_view.render,
    "🗺️ Migration": migration_view.render,
    "🧭 Interpretation": interpretation_view.render,
    "🧬 Attributes": attributes_view.render,
    "💾 Export": export_view.render,
}

with st.sidebar:
    st.markdown(f"## {cfg.APP_ICON} {cfg.APP_TITLE}")
    st.caption(f"v{cfg.APP_VERSION}")
    page = st.radio("Navigate", list(PAGES.keys()), key="active_page", label_visibility="collapsed")
    st.divider()
    st.progress(ss.progress_fraction(),
                text=f"{len(st.session_state['completed_stages'])}/{len(_pipeline_stages)} stages")
    if st.session_state["survey"] is not None:
        st.caption(f"📂 {st.session_state['survey'].source_name}")
    if st.session_state["history"]:
        with st.expander("Processing history"):
            for h in st.session_state["history"]:
                st.caption(f"✓ {h.label}")

PAGES[page]()
