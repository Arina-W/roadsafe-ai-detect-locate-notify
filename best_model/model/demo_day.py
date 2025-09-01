import io
import os
import json
import math
import base64
from datetime import datetime
from email.message import EmailMessage

import streamlit as st
from PIL import Image, ExifTags
import pandas as pd
import requests
from streamlit_folium import st_folium
import folium

# ==========================
# Imports for model session
# ==========================
try:
    from model import load as load_session, MATERIAL_NAMES, QUALITY_NAMES  # user may have model.py
except Exception:
    from core import load as load_session, MATERIAL_NAMES, QUALITY_NAMES   # or core.py as provided by you

# ==========================
# Page config & styles
# ==========================
st.set_page_config(page_title="RoadSafe — Report a Road Issue", page_icon="🛣️", layout="centered")

st.markdown(
    """
    <style>
      .center {text-align:center}
      .muted {opacity: .75}
      .chip {display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #2a2a2a; margin-right:6px}
      .ok {background: #0b4; color: white}
      .bad {background: #b20; color: white}
      .btn-row > div {display:inline-block; margin-right: .5rem}
      .coords {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Helpers
# ==========================
TOKYO_DEFAULT = (35.681236, 139.767125)  # Tokyo Station
BAD_QUALITIES = {"bad", "very_bad"}
GOODISH_QUALITIES = {"excellent", "good", "intermediate"}
ROAD_CONF_THRESHOLD = 0.50  # min prob for material head to call it a road-like shot

@st.cache_resource(show_spinner=False)
def get_model(weights_path: str = "weights/best_model.pt"):
    return load_session(weights_path)

# --- EXIF GPS extraction
_GPS_TAGS = None
for k, v in ExifTags.TAGS.items():
    if v == "GPSInfo":
        _GPS_TAGS = k
        break

def _to_deg(value):
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    return d + (m / 60.0) + (s / 3600.0)

def exif_latlon(pil_img: Image.Image):
    try:
        exif = pil_img._getexif()
        if not exif or _GPS_TAGS not in exif:
            return None
        gps = exif[_GPS_TAGS]
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
        lat = _to_deg(gps["GPSLatitude"]) if "GPSLatitude" in gps else None
        lon = _to_deg(gps["GPSLongitude"]) if "GPSLongitude" in gps else None
        if lat is None or lon is None:
            return None
        if gps.get("GPSLatitudeRef", "N") == "S":
            lat = -lat
        if gps.get("GPSLongitudeRef", "E") == "W":
            lon = -lon
        return (lat, lon)
    except Exception:
        return None

# --- Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# --- Overpass query (nearest ward/city office)
@st.cache_data(show_spinner=False)
def query_nearest_office(lat: float, lon: float, radius_m: int = 5000):
    """Return the best-matching ward/city office near (lat, lon).
    Strategy:
      1) Query Overpass with strong signals first (townhall + name regex)
      2) Fall back to broader government/administrative offices
      3) Score candidates by tag strength + distance, prefer offices with proper names
    """
    overpass_url = "https://overpass-api.de/api/interpreter"

    # Regex targets for JP + EN ward/city office names
    NAME_REGEX = r"Ward Office|City Office|City Hall|Town Hall|区役所|市役所|町役場|村役場|役場|出張所"

    def _mk_query(lat, lon, r, strict: bool):
        if strict:
            return f"""
            [out:json][timeout:25];
            (
              node["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
              way ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
              rel ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
            );
            out center tags;
            """
        else:
            return f"""
            [out:json][timeout:25];
            (
              node["amenity"="townhall"](around:{r},{lat},{lon});
              way ["amenity"="townhall"](around:{r},{lat},{lon});
              rel ["amenity"="townhall"](around:{r},{lat},{lon});
              node["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
              way ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
              rel ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
              node["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
              way ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
              rel ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
            );
            out center tags;
            """

    def _center(el):
        if el.get("type") == "node":
            return (el.get("lat"), el.get("lon"))
        c = el.get("center") or {}
        return (c.get("lat"), c.get("lon"))

    def _pick_name(tags: dict) -> str | None:
        # Prefer English if present, then generic name, then official variants
        for k in ("name:en", "official_name:en", "name", "official_name", "name:ja"):
            if tags.get(k):
                return tags[k]
        return None

    def _addr(tags: dict) -> str | None:
        parts = [
            tags.get("addr:postcode"),
            tags.get("addr:state") or tags.get("addr:province"),
            tags.get("addr:city"),
            tags.get("addr:district") or tags.get("addr:ward") or tags.get("addr:suburb"),
            tags.get("addr:neighbourhood"),
            tags.get("addr:street"),
            tags.get("addr:block_number"),
            tags.get("addr:housenumber"),
        ]
        s = " ".join([p for p in parts if p]).strip()
        return s or None

    def _score(tags: dict, dist_km: float) -> float:
        name = _pick_name(tags)
        has_name = 1.0 if name else 0.0
        amenity_townhall = 1.0 if tags.get("amenity") == "townhall" else 0.0
        admin_gov = 1.0 if (tags.get("office") == "government" and tags.get("government") == "administrative") else 0.0
        name_hit = 1.0 if name and (any(x in name for x in ["Ward Office","City Office","City Hall","Town Hall"]) or any(x in name for x in ["区役所","市役所","町役場","村役場","役場","出張所"])) else 0.0
        has_addr = 1.0 if _addr(tags) else 0.0
        has_email = 1.0 if (tags.get("contact:email") or tags.get("email")) else 0.0
        # Weighted sum minus distance penalty (1 point per km)
        base = 3*amenity_townhall + 2.5*name_hit + 1.5*admin_gov + 1.0*has_name + 0.5*has_addr + 0.25*has_email
        return base - (dist_km)

    # Try escalating radius and strictness
    for r in (2000, 5000, 10000):
        for strict in (True, False):
            try:
                q = _mk_query(lat, lon, r, strict)
                resp = requests.post(overpass_url, data=q, timeout=25)
                resp.raise_for_status()
                js = resp.json()
            except Exception:
                js = {}
            elems = js.get("elements", [])
            if not elems:
                continue
            best = None
            for el in elems:
                latc, lonc = _center(el)
                if latc is None or lonc is None:
                    continue
                dkm = haversine(lat, lon, latc, lonc)
                tags = el.get("tags", {})
                sc = _score(tags, dkm)
                item = {
                    "name": _pick_name(tags) or "(Unnamed Government Office)",
                    "lat": latc,
                    "lon": lonc,
                    "distance_km": dkm,
                    "address": _addr(tags),
                    "email": tags.get("contact:email") or tags.get("email"),
                    "phone": tags.get("contact:phone") or tags.get("phone"),
                    "website": tags.get("contact:website") or tags.get("website"),
                    "osm_tags": tags,
                    "score": sc,
                }
                if (best is None) or (item["score"] > best["score"]):
                    best = item
            if best:
                # Clean display name if it still looks generic
                if best["name"] in (None, "Nearest Government Office", "(Unnamed Government Office)"):
                    nm = best["osm_tags"].get("operator") or best["osm_tags"].get("owner")
                    if nm:
                        best["name"] = nm
                return best
    return None

# --- Google Maps link
def gmaps_link(lat: float, lon: float):
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# --- Email sending (Gmail SMTP via app password or any SMTP)
def send_email_report(to_email: str, subject: str, body: str, image_bytes: bytes = None, image_name: str = None) -> tuple[bool, str]:
    """Send an email using SMTP. Configure creds in Streamlit Secrets.
    Expected secrets:
      SMTP_HOST (default smtp.gmail.com)
      SMTP_PORT (default 587)
      SMTP_USER (sender email)
      SMTP_PASS (app password or SMTP password)
      SENDER_NAME (optional)
    Returns (ok, message)
    """
    host = st.secrets.get("SMTP_HOST", "smtp.gmail.com")
    port = int(st.secrets.get("SMTP_PORT", 587))
    user = st.secrets.get("SMTP_USER")
    pwd  = st.secrets.get("SMTP_PASS")
    sender_name = st.secrets.get("SENDER_NAME", "RoadSafe Bot")

    if not user or not pwd:
        return False, "SMTP credentials missing: set SMTP_USER and SMTP_PASS in secrets."

    msg = EmailMessage()
    msg["From"] = f"{sender_name} <{user}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    if image_bytes and image_name:
        msg.add_attachment(image_bytes, maintype="image", subtype=image_name.split(".")[-1].lower(), filename=image_name)

    import smtplib
    try:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls()
            s.login(user, pwd)
            s.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, f"Email error: {e}"

# --- Simple description generator (no external APIs required)
def describe_damage(surface_type: str, quality: str) -> str:
    q_txt = {
        "very_bad": "severely degraded",
        "bad": "noticeably degraded",
        "intermediate": "moderately worn",
        "good": "in good condition",
        "excellent": "in excellent condition",
    }[quality]
    return (
        f"Detected a {surface_type.replace('_',' ')} surface that appears {q_txt}. "
        f"Please consider inspecting this location for safety risks (trips, bicycle instability, vehicle damage)."
    )

# --- Session state init
if "pred" not in st.session_state:
    st.session_state.pred = None
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "img_name" not in st.session_state:
    st.session_state.img_name = None
if "manual_point" not in st.session_state:
    st.session_state.manual_point = None  # (lat, lon)
if "location_confirmed" not in st.session_state:
    st.session_state.location_confirmed = False
if "ward_office" not in st.session_state:
    st.session_state.ward_office = None
if "report_sent" not in st.session_state:
    st.session_state.report_sent = False
if "allow_submit_anyway" not in st.session_state:
    st.session_state.allow_submit_anyway = False

# --- Reset helpers
def reset_flow(full=False):
    st.session_state.manual_point = None
    st.session_state.location_confirmed = False
    st.session_state.ward_office = None
    st.session_state.allow_submit_anyway = False
    if full:
        st.session_state.pred = None
        st.session_state.img_bytes = None
        st.session_state.img_name = None

# ==========================
# Header
# ==========================
st.markdown(
    """
    <div class='center'>
      <h1 style="margin-bottom:0.2rem;">🛣️ RoadSafe — Report Road Damage</h1>
      <p class='muted' style='margin-top:0'>Upload → Validate → Pinpoint → Notify Ward Office</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# If user just sent a report
if st.session_state.report_sent:
    st.success("Report sent. Thank you very much! 🙏")
    st.caption("You can upload another photo below.")

# ==========================
# Upload
# ==========================
uploader = st.file_uploader("Upload a road photo (JPG/PNG)", type=["jpg", "jpeg", "png"], help="A clear road-surface photo works best.")

if uploader is not None:
    img = Image.open(uploader).convert("RGB")
    img_bytes = uploader.getvalue()
    st.session_state.img_bytes = img_bytes
    st.session_state.img_name = uploader.name

    st.image(img, caption="Input image", use_container_width=True)

    # Run model once per image
    with st.spinner("Analyzing the image…"):
        sess = get_model()
        result = sess(img)

    st.session_state.pred = result
    st.session_state.report_sent = False
    st.session_state.ward_office = None

# ==========================
# Prediction + gating
# ==========================
if st.session_state.pred:
    res = st.session_state.pred
    # Summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Surface Type", res["surface_type"].replace("_"," ").title())
    with c2:
        st.metric("Surface Quality", res["surface_quality"].replace("_"," ").title())
    with c3:
        type_max = max(res["surface_type_probs"]) if res.get("surface_type_probs") else 0.0
        st.metric("Type Confidence", f"{type_max*100:.1f}%")

    # Probabilities side-by-side
    tdf = pd.DataFrame({"probability": res["surface_type_probs"]}, index=[n.replace("_"," ").title() for n in MATERIAL_NAMES])
    qdf = pd.DataFrame({"probability": res["surface_quality_probs"]}, index=[n.replace("_"," ").title() for n in QUALITY_NAMES])
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Surface Type probabilities")
        st.bar_chart(tdf)
    with cc2:
        st.caption("Surface Quality probabilities")
        st.bar_chart(qdf)

    # Basic road/validity check via confidence
    looks_like_road = (max(res["surface_type_probs"]) >= ROAD_CONF_THRESHOLD)

    if not looks_like_road:
        st.warning("This photo might not be a road-surface shot. If you're sure it is, you can proceed anyway.")
        st.session_state.allow_submit_anyway = st.toggle("Proceed anyway", value=st.session_state.allow_submit_anyway)

    # Early exit if we don't consider it road-like and user didn't override
    if (not looks_like_road) and (not st.session_state.allow_submit_anyway):
        st.stop()

    # If quality is good-ish, discourage submission
    quality = res["surface_quality"]
    if quality in GOODISH_QUALITIES:
        st.info("The road appears to be in decent condition. We don't recommend submitting a report.")
        st.session_state.allow_submit_anyway = st.toggle("Report anyway", value=st.session_state.allow_submit_anyway)
        if not st.session_state.allow_submit_anyway:
            st.stop()

    # ==========================
    # Location step (EXIF → manual pinpoint)
    # ==========================
    st.divider()
    st.subheader("📍 Confirm location of the damage")

    # Try EXIF first
    if st.session_state.manual_point is None and not st.session_state.location_confirmed:
        with st.status("Checking photo metadata for GPS…", expanded=False) as status:
            gps = exif_latlon(Image.open(io.BytesIO(st.session_state.img_bytes))) if st.session_state.img_bytes else None
            if gps:
                st.session_state.manual_point = (float(gps[0]), float(gps[1]))
                status.update(label="Found GPS in metadata", state="complete")
            else:
                status.update(label="No GPS metadata found — please pinpoint on the map.", state="error")

    # Show map for manual/confirm flow
    latlon = st.session_state.manual_point or TOKYO_DEFAULT

    m = folium.Map(location=latlon, zoom_start=16 if st.session_state.manual_point else 12, control_scale=True)
    # Add a single marker if we have a point
    if st.session_state.manual_point:
        folium.Marker(location=st.session_state.manual_point, draggable=False, tooltip="Proposed location").add_to(m)

    st.caption("Click on the map to (re)place a single pin. Use Reset to clear it, then click again.")
    out = st_folium(m, width=720, height=420, key="map")

    # Handle click without reloading the whole page (Streamlit reruns, but we persist state)
    click = out.get("last_clicked") if isinstance(out, dict) else None
    if click and (click.get("lat") and click.get("lng")):
        st.session_state.manual_point = (float(click["lat"]), float(click["lng"]))
        st.session_state.location_confirmed = False
        st.session_state.ward_office = None
        st.rerun()  # immediately reflect the new marker without extra UI changes

    # Controls
    colA, colB, colC = st.columns([1,1,2])
    with colA:
        if st.button("Reset pin", type="secondary"):
            st.session_state.manual_point = None
            st.session_state.location_confirmed = False
            st.session_state.ward_office = None
            st.rerun()
    with colB:
        confirm_disabled = st.session_state.manual_point is None
        if st.button("Confirm location", type="primary", disabled=confirm_disabled):
            st.session_state.location_confirmed = True
            st.session_state.ward_office = None
            st.toast("Location confirmed ✔️")

    # Show coordinates
    lat, lon = (st.session_state.manual_point or (None, None))
    if lat and lon:
        st.caption(f"Chosen coordinates: <span class='coords'>{lat:.6f}, {lon:.6f}</span>", unsafe_allow_html=True)

    # ==========================
    # Ward office lookup
    # ==========================
    if st.session_state.location_confirmed and st.session_state.manual_point:
        st.divider()
        st.subheader("🏛️ Closest ward/city office")
        if st.session_state.ward_office is None:
            with st.spinner("Looking up nearest office…"):
                office = query_nearest_office(lat, lon)
                st.session_state.ward_office = office
        office = st.session_state.ward_office
        if office:
            gmaps = gmaps_link(office["lat"], office["lon"])
            st.success(
                f"Closest office identified: **{office['name']}**\n\n"
                + (f"Address: {office['address']}\n\n" if office.get("address") else "")
            )
            st.link_button("Open office location in Google Maps", gmaps, type="secondary")
        else:
            st.warning("We couldn't find a nearby ward/city office automatically. You can still send a report — we'll include the coordinates.")

        # ==========================
        # Report send section
        # ==========================
        st.divider()
        st.subheader("📧 Send the report")
        to_email = st.text_input("Recipient", value="tokyoreportaddress@gmail.com", disabled=True)

        # Compose message preview
        surf = res["surface_type"].replace("_"," ")
        qual = res["surface_quality"]
        desc = describe_damage(surf, qual)
        body_preview = f"""
RoadSafe report (demo)\n
When: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n
Surface type: {surf}\nSurface quality: {qual}\n
Location: {lat:.6f}, {lon:.6f}\nGoogle Maps: {gmaps_link(lat, lon)}\n\nWard office: {(office['name'] if office else 'N/A')}\nAddress: {(office['address'] if office and office.get('address') else 'N/A')}\n\nSummary: {desc}\n        """.strip()

        st.text_area("Email preview", value=body_preview, height=180)

        if st.button("Send report", type="primary"):
            with st.spinner("Sending…"):
                ok, msg = send_email_report(
                    to_email=to_email,
                    subject="RoadSafe Report (Demo)",
                    body=body_preview,
                    image_bytes=st.session_state.img_bytes,
                    image_name=st.session_state.img_name or "photo.jpg",
                )
            if ok:
                st.session_state.report_sent = True
                reset_flow(full=True)
                st.success("Report sent. Thank you very much!")
                st.balloons()
            else:
                st.error(msg)

# ==========================
# Footer / helper notes
# ==========================
st.divider()
st.caption(
    "Pro tip: set SMTP creds in `.streamlit/secrets.toml` → `SMTP_USER`, `SMTP_PASS` (Gmail app password), optional `SMTP_HOST`, `SMTP_PORT`, `SENDER_NAME`."
)
