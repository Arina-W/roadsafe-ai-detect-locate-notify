# RoadSafe — Surface Type & Quality

Minimal package to classify road **surface type** and **surface quality** with an EfficientNet-B7 multi-task model.

---

## 📂 Setup

1. Put your trained weights at:

```bash
best_model/weights/best_model.pt
```


2. Install dependencies:
```bash
cd best_model
pip install -r requirements.txt
```
3. (Optional for local package import):
```bash
pip install -e .
```

## 🐍 Use in any Python / Streamlit app

### You can import the package and call predict() or load() anywhere.

```bash
from model import load, predict

# Option 1: quick one-shot prediction
print(predict("road.jpg", weights="weights/best_model.pt"))

# Option 2: keep a session for multiple predictions
session = load("weights/best_model.pt")
res = session("road.jpg")
print(res["surface_type"], res["surface_quality"])
```

## In a Streamlit app:

```bash
import streamlit as st
from PIL import Image
from model import load

session = load("weights/best_model.pt")

f = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
if f:
    img = Image.open(f).convert("RGB")
    res = session(img)
    st.write(res)
```

## 🎨 Run the included demo

### We ship a tiny Streamlit demo inside the package that looks nice out of the box.

```bash
streamlit run model/demo_streamlit.py
```

