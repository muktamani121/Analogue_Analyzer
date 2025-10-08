# utils/ui_components.py
import streamlit as st

def render_clickable_card(key: str, name: str, desc: str, color_class: str):
    with st.form(key=f"form_{key}"):
        st.markdown(
            f"""
            <div class='data-box data-box-{color_class}' onclick="document.getElementById('btn_{key}').click();">
                <h2>{name}</h2>
                <p>{desc}</p>
            </div>
            <button id="btn_{key}" style="display:none;"></button>
            """,
            unsafe_allow_html=True
        )
        submitted = st.form_submit_button(label="")
        return submitted
