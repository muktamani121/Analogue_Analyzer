# File: utils/styling.py
import streamlit as st


def load_css():
    """Inject global CSS for the Analogue Analyzer app with left alignment, hover effects, and nav button styling."""
    st.markdown(
        """
        <style>
        
        .appview-container .main .block-container {
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100% !important;
        }

        .main-title {
            font-size: 3rem;
            text-align: left;
            margin-bottom: 0.4em;
            font-weight: 800;
            color: #e74c3c !important;
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: white;
            padding: 10px 0;
        }

        .about {
            font-size: 1.05rem;
            line-height: 1.58;
            color: #333;
            width: 100%;
            max-width: 100%;
            text-align: justify;
            margin: 0 0 2.5em 0;
            padding-left: 0;
        }

        .data-box {
            border-radius: 12px;
            padding: 1.5rem;
            min-height: 320px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.10);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            width: 100%;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .data-box:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }

        .data-box h2 {
            margin-top: 0;
            font-size: 1.6rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .data-box h2 img {
            height: 20px;
            width: auto;
            margin-bottom: -2px;
        }

        .data-box p {
            flex: 1;
            font-size: 0.95rem;
            line-height: 1.45;
        }

        .data-box-green { background: #27ae60; color: #fff; }
        .data-box-purple { background: #8e44ad; color: #fff; }
        .data-box-blue { background: #2980b9; color: #fff; }

        /* Default button styling */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            font-weight: 600;
            margin-top: 1rem;
        }

        /* apply_button_color wrapper classes */
        .red-button .stButton > button {
            background-color: #d9534f !important;
            color: white !important;
            width: auto !important;
        }
        .green-button .stButton > button {
            background-color: #5cb85c !important;
            color: white !important;
            width: auto !important;
        }
        .purple-button .stButton > button {
            background-color: #8e44ad !important;
            color: white !important;
            width: auto !important;
        }
        .blue-button .stButton > button {
            background-color: #2980b9 !important;
            color: white !important;
            width: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_banner():
    """Render the sticky top banner with the app title."""
    pass


def apply_button_color(color_type: str):
    """Wrap the next st.button() in a div that applies one of our button-color classes."""
    cls = {
        'red': 'red-button',
        'green': 'green-button',
        'grey': 'grey-button',
        'purple': 'purple-button',
        'blue': 'blue-button',
    }.get(color_type)
    if cls:
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)


def close_button_color():
    """Close the wrapper div opened by apply_button_color()."""
    st.markdown("</div>", unsafe_allow_html=True)