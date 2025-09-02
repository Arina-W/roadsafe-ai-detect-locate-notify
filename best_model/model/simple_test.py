import streamlit as st
from PIL import Image
from model import load

session = load("weights/best_model.pt")

f = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB")
    res = session(img)
    st.write(res)
