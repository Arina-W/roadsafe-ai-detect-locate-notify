import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from PIL import Image
import io
import gzip
import piexif
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import smtplib
from email.message import EmailMessage
from datetime import datetime
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os

# ---------------- Load environment variables ----------------
load_dotenv()
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")

# ---------------- Config ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "ResNet50": "esdk/my-efficientnet-model/resnet.pth",
    "EfficientNet-B7": "esdk/my-efficientnet-model/efficientnet_fp16.pt.gz"
}
DEFAULT_MODEL = "EfficientNet-B7"

SURFACE_TYPE_MAP = {"asphalt": 0, "concrete": 1, "paving_stones": 2, "unpaved": 3, "sett": 4}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}

SURFACE_QUALITY_MAP = {"excellent": 0, "good": 1, "intermediate": 2, "bad": 3, "very_bad": 4}
SURFACE_QUALITY_MAP_INV = {v: k for k, v in SURFACE_QUALITY_MAP.items()}

NUM_MATERIALS, NUM_QUALITIES = 5, 5

# ---------------- Load CSV ----------------
@st.cache_data
def load_emails():
    try:
        df = pd.read_csv("csv.csv")
        return df
    except Exception as e:
        st.error(f"Could not read csv.csv: {e}")
        return pd.DataFrame(columns=["city", "email"])

EMAIL_DF = load_emails()

# ---------------- Multi-head Models ----------------
class MultiHeadResNet50(nn.Module):
    def __init__(self, num_types=len(SURFACE_TYPE_MAP), num_qual=len(SURFACE_QUALITY_MAP)):
        super().__init__()
        base = models.resnet50(weights=None)
        in_feat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.fc_type = nn.Linear(in_feat, num_types)
        self.fc_qual = nn.Linear(in_feat, num_qual)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_type(features), self.fc_qual(features)

class MultiHeadEffNetB7(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        f = base.classifier[1].in_features
        self.mat = nn.Linear(f, NUM_MATERIALS)
        self.qual = nn.Linear(f, NUM_QUALITIES)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.mat(x), self.qual(x)

# ---------------- Model Loader ----------------
@st.cache_resource
def load_model(model_choice):
    try:
        if model_choice == "ResNet50":
            model = MultiHeadResNet50().to(device)
            local_path = hf_hub_download(repo_id="esdk/my-efficientnet-model", filename="resnet.pth")
            state_dict = torch.load(local_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            model = MultiHeadEffNetB7().to(device)
            local_path = hf_hub_download(repo_id="esdk/my-efficientnet-model", filename="efficientnet_fp16.pt.gz")
            with gzip.open(local_path, "rb") as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer, map_location=device)
            fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(fixed_state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# ---------------- Transform ----------------
def get_transform(model_choice):
    input_size = 600 if model_choice == "EfficientNet-B7" else 224
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ---------------- GPS & EXIF ----------------
def get_gps_coords_from_bytes(file_bytes):
    try:
        exif_dict = piexif.load(file_bytes)
        gps = exif_dict.get("GPS")
        if not gps:
            return None
        def rational_to_deg(rat):
            return rat[0][0]/rat[0][1] + rat[1][0]/rat[1][1]/60 + rat[2][0]/rat[2][1]/3600
        lat = rational_to_deg(gps[piexif.GPSIFD.GPSLatitude])
        lon = rational_to_deg(gps[piexif.GPSIFD.GPSLongitude])
        lat_ref = gps[piexif.GPSIFD.GPSLatitudeRef].decode()
        lon_ref = gps[piexif.GPSIFD.GPSLongitudeRef].decode()
        if lat_ref == "S": lat = -lat
        if lon_ref == "W": lon = -lon
        return lat, lon
    except:
        return None

def get_image_timestamp(file_bytes):
    try:
        exif_dict = piexif.load(file_bytes)
        dt_bytes = exif_dict.get("0th").get(piexif.ImageIFD.DateTime)
        if dt_bytes:
            return dt_bytes.decode()
        return None
    except:
        return None

# ---------------- Email ----------------
def send_email(to_email, subject, body, attachment_bytes=None, attachment_name="image.jpg"):
    if not GMAIL_USER or not GMAIL_PASSWORD:
        return False, "Email credentials not found. Set them in the .env file."
    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body, charset="utf-8")
    if attachment_bytes:
        msg.add_attachment(
            attachment_bytes,
            maintype="image",
            subtype="jpeg",
            filename=attachment_name
        )
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

# ---------------- Streamlit UI ----------------
st.title("Street Surface Classification & GPS")

# Model selection
model_choice = st.selectbox("Choose Model:", list(MODEL_PATHS.keys()), index=list(MODEL_PATHS.keys()).index(DEFAULT_MODEL))

# Load model immediately
with st.spinner("Loading model, please wait..."):
    model = load_model(model_choice)
    transform = get_transform(model_choice)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    # Prediction
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out_type, out_qual = model(img_tensor)
        main_pred = SURFACE_TYPE_MAP_INV[out_type.argmax(1).item()]
        sub_pred = SURFACE_QUALITY_MAP_INV[out_qual.argmax(1).item()]

    st.success(f"**Model:** {model_choice}")
    st.success(f"Predicted Surface Type: **{main_pred}**")
    st.success(f"Predicted Surface Quality: **{sub_pred}**")

    if sub_pred in ["excellent", "good"]:
        st.warning("The road seems to be in good condition. Are you sure you want to report it?")

    # Timestamps
    img_timestamp = get_image_timestamp(file_bytes)
    upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # GPS
    coords = get_gps_coords_from_bytes(file_bytes)
    lat, lon = coords if coords else (None, None)
    if coords:
        st.success(f"GPS metadata found: {lat}, {lon}")
    else:
        st.info("No GPS metadata found. Please select location on the map.")

    # Map
    map_center = [lat, lon] if lat and lon else [35.68, 139.76]
    m = folium.Map(location=map_center, zoom_start=16)
    if lat and lon:
        folium.Marker([lat, lon], tooltip="Detected Location").add_to(m)
    map_data = st_folium(m, width=700, height=500)

    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.success(f"Location selected: {lat}, {lon}")

    # Reverse geocoding
    street_name, city_name = None, None
    if lat and lon:
        try:
            geolocator = Nominatim(user_agent="street_app")
            location = geolocator.reverse((lat, lon), language='en')
            if location:
                addr = location.raw.get("address", {})
                street_name = addr.get("road")
                city_name = addr.get("city") or addr.get("town") or addr.get("suburb")
                if street_name:
                    st.info(f"Street detected: {street_name}")
                if city_name:
                    st.info(f"City detected: {city_name}")
        except:
            street_name, city_name = None, None

    # Lookup ward email
    to_email = "recipient@example.com"
    if city_name:
        match = EMAIL_DF[EMAIL_DF["city"].str.lower() == city_name.lower()]
        if not match.empty:
            to_email = match.iloc[0]["email"]
            st.success(f"Sending report to: {to_email}")

    # Send report
    if st.button("Send Report via Email"):
        subject = "Street Surface Report"
        body = f"""Street Surface Report
Surface Type: {main_pred}
Surface Quality: {sub_pred}
Street Name: {street_name if street_name else 'Unknown'}
City: {city_name if city_name else 'Unknown'}
GPS: {lat if lat else 'Unknown'}, {lon if lon else 'Unknown'}
Picture taken: {img_timestamp if img_timestamp else 'Unknown'}
Upload time: {upload_timestamp}
"""
        success, info = send_email(to_email, subject, body, file_bytes, "street_image.jpg")
        if success:
            st.success("Report sent successfully!")
        else:
            st.error(f"Failed to send email: {info}")
