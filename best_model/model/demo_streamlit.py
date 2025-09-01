import streamlit as st
from PIL import Image
import pandas as pd
from model import load, MATERIAL_NAMES, QUALITY_NAMES

st.set_page_config(page_title="RoadSafe Demo", page_icon="🛣️", layout="centered")

# --- Header
st.markdown(
    """
    <div style="text-align:center">
      <h1 style="margin-bottom:0.2rem;">🛣️ RoadSafe — Surface Classifier</h1>
      <p style="opacity:0.75;margin-top:0;">Surface <b>type</b> + <b>quality</b> from a single image</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Load once (cache)
@st.cache_resource
def get_session(weights_path: str):
    return load(weights_path)

session = get_session("weights/best_model.pt")

# --- Uploader
uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    res = session(img)

    # --- Nice summary
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Surface Type")
        st.markdown(f"### **{res['surface_type'].title()}**")
    with c2:
        st.subheader("Surface Quality")
        st.markdown(f"### **{res['surface_quality'].replace('_',' ').title()}**")

    # --- Probabilities
    st.divider()
    st.subheader("Class probabilities")

    type_df = pd.DataFrame(
        {"probability": res["surface_type_probs"]},
        index=[n.replace("_", " ").title() for n in MATERIAL_NAMES],
    )
    qual_df = pd.DataFrame(
        {"probability": res["surface_quality_probs"]},
        index=[n.replace("_", " ").title() for n in QUALITY_NAMES],
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Surface Type")
        st.bar_chart(type_df)
    with cc2:
        st.caption("Surface Quality")
        st.bar_chart(qual_df)

else:
    st.info("Upload a JPG/PNG road image to see predictions.")
    st.caption("Tip: the model expects roughly road-surface photos—oblique angles are okay.")
