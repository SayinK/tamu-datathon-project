# Home.py — Landing page (no data loading)
import streamlit as st

st.set_page_config(
    page_title="Home • Mai Shan Yan",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#cd1b1b"  # use the alpha-less hex for CSS

# ---------- Accent CSS (uses your primary color) ----------
st.markdown(
    f"""
    <style>
      :root {{
        --primary: {PRIMARY};
        --primaryA04: {PRIMARY}14;  /* ~8% */
        --primaryA10: {PRIMARY}26;  /* ~15% */
        --primaryA20: {PRIMARY}33;  /* ~20% */
        --primaryA35: {PRIMARY}59;  /* ~35% */
      }}

      /* Headline accent underline */
      .accent-underline {{
        position: relative;
        display: inline-block;
      }}
      .accent-underline::after {{
        content: '';
        position: absolute;
        left: 0; right: 0; bottom: -10px;
        height: 6px; border-radius: 8px;
        background: linear-gradient(90deg, var(--primary), var(--primaryA10));
      }}

      /* Page links → pill buttons */
      a[data-testid="stPageLink"] {{
        display: inline-flex; align-items: center; gap: .5rem;
        padding: 10px 14px; border-radius: 999px;
        border: 1px solid var(--primaryA35);
        background: var(--primaryA04);
        text-decoration: none !important;
      }}
      a[data-testid="stPageLink"]:hover {{
        background: var(--primaryA10);
        border-color: var(--primary);
        box-shadow: 0 0 0 3px var(--primaryA10) inset;
      }}

      /* Feature cards with subtle red border + glow */
      .msy-card {{
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--primaryA20);
        box-shadow: 0 6px 24px rgba(0,0,0,.25), 0 0 0 1px var(--primaryA04) inset;
      }}
      .msy-card h3 {{ margin: 0 0 6px 0; }}

      /* Sidebar: active nav gets a slim primary bar */
      section[data-testid="stSidebar"] [aria-current="page"] {{
        border-left: 4px solid var(--primary);
        border-radius: 6px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero ----------
st.markdown(
    """
    <div style="padding:48px 0 12px 0;">
      <h1 style="font-size:42px; line-height:1.15; margin:0;">
        📦 Mai Shan Yan — <span class="accent-underline" style="opacity:.95;">Inventory Management</span>
      </h1>
      <p style="font-size:18px; color:#bdbdbd; margin-top:14px; max-width:900px;">
        A data-powered inventory management dashboard that helps restaurant managers
        monitor essentials, compare periods, and keep operations smooth.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Quick actions ----------
c1, c2 = st.columns([1.1, 1])
with c1:
    st.page_link("pages/Shipment_Dashboard.py", label="Open Shipments Dashboard", icon="📊")
with c2:
    st.page_link("pages/Monthly_Shipments.py", label="Open Monthly Sales", icon="🗓️")

st.divider()

# ---------- Feature tiles ----------
fc1, fc2 = st.columns(2)
with fc1:
    st.markdown(
        """
        <div class="msy-card">
          <h3>Fast overview</h3>
          <p style="opacity:.75; margin:0;">
            One place to browse key inventory pages and jump straight into the view you need.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with fc2:
    st.markdown(
        """
        <div class="msy-card">
          <h3>Actionable layouts</h3>
          <p style="opacity:.75; margin:0;">
            Clean charts and tables organized for quick checks and decisions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Footer ----------
st.markdown(
    """
    <div style="opacity:.55; font-size:13px; padding-top:28px;">
      Tip: Use the sidebar to switch pages anytime. This home screen is just a clean starting point.
    </div>
    """,
    unsafe_allow_html=True,
)
