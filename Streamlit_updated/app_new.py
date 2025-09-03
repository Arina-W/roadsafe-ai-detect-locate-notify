


# # RoadSafe — Streamlit app (stable map + robust EXIF/XMP GPS + confirm-to-lookup + hard reset)
# # --------------------------------------------------------------------------------------------
# import io, os, re, math, gzip, time
# from datetime import datetime
# from email.message import EmailMessage

# import streamlit as st
# from PIL import Image, ExifTags
# import requests
# import folium
# from streamlit_folium import st_folium

# # ML
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
# from huggingface_hub import hf_hub_download

# # ---- env ----
# from dotenv import load_dotenv
# load_dotenv()
# GMAIL_USER      = os.getenv("GMAIL_USER")
# GMAIL_PASSWORD  = os.getenv("GMAIL_PASSWORD")
# HF_REPO_ID      = os.getenv("HF_REPO_ID") or "esdk/my-efficientnet-model"
# HF_FILENAME     = os.getenv("HF_FILENAME") or "efficientnet_fp16.pt.gz"
# HF_TOKEN        = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
# DEMO_RECEIVER   = os.getenv("DEMO_RECEIVER")

# # ---- page & css ----
# st.set_page_config(page_title="RoadSafe — Report a Road Issue", page_icon="🛣️", layout="centered")
# st.markdown("""
# <style>
#   .center {text-align:center}
#   .muted {opacity:.75}
#   .section-title {font-size:1.05rem; font-weight:700; margin: 10px 0 4px}
#   .result-value {font-size:2.2rem; font-weight:800; margin: 0 0 6px}
#   .qual-ok  {color:#22c55e}
#   .qual-mid {color:#a16207}
#   .qual-bad {color:#ef4444}
#   .coords {font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace}
#   .folium-map { margin-bottom: 0 !important; }
#   div[data-testid="stVerticalBlock"] > div:has(.folium-map) { margin-bottom: 0 !important; }
#   .tiny {font-size: .85rem; opacity: .7}
# </style>
# """, unsafe_allow_html=True)

# # ---- labels/defaults ----
# MATERIAL_NAMES = ["asphalt", "concrete", "paving_stones", "unpaved", "sett"]
# QUALITY_NAMES  = ["excellent", "good", "intermediate", "bad", "very_bad"]
# TOKYO_DEFAULT  = (35.681236, 139.767125)

# # === MODEL ===
# class MultiHeadEffB7(nn.Module):
#     def __init__(self, n_type=len(MATERIAL_NAMES), n_qual=len(QUALITY_NAMES)):
#         super().__init__()
#         base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
#         self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
#         in_f = base.classifier[1].in_features
#         self.mat  = nn.Linear(in_f, n_type)
#         self.qual = nn.Linear(in_f, n_qual)
#     def forward(self, x):
#         z = self.features(x)
#         return self.mat(z), self.qual(z)

# @st.cache_resource(show_spinner=True)
# def get_model_session():
#     if not HF_REPO_ID or not HF_FILENAME:
#         st.error("HF_REPO_ID / HF_FILENAME not set in env."); st.stop()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultiHeadEffB7().to(device)

#     path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=HF_TOKEN)
#     if path.endswith(".gz"):
#         with gzip.open(path, "rb") as f:
#             state = torch.load(io.BytesIO(f.read()), map_location=device)
#     else:
#         state = torch.load(path, map_location=device)
#     if isinstance(state, dict) and "state_dict" in state:
#         state = state["state_dict"]
#     state = {k.replace("_orig_mod.","").replace("module.",""): v for k, v in state.items()}
#     model.load_state_dict(state, strict=False)
#     model.eval()

#     tfm = transforms.Compose([
#         transforms.Resize((600, 600)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ])

#     @torch.inference_mode()
#     def session(pil_img):
#         x = tfm(pil_img).unsqueeze(0).to(device)
#         t_logits, q_logits = model(x)
#         t_idx = int(torch.softmax(t_logits, 1).argmax())
#         q_idx = int(torch.softmax(q_logits, 1).argmax())
#         return {"surface_type": MATERIAL_NAMES[t_idx], "surface_quality": QUALITY_NAMES[q_idx]}
#     return session

# # ---- GPS extraction (EXIF first, then XMP fallback) ----
# _GPS_TAGS = 34853  # GPSInfo

# def _num(x):
#     try:
#         return float(getattr(x, "numerator", x)) / float(getattr(x, "denominator", 1))
#     except Exception:
#         try:
#             a, b = x
#             return float(a) / float(b)
#         except Exception:
#             return float(x)

# def _to_deg_any(v):
#     try:
#         if isinstance(v, (list, tuple)) and len(v) == 3:
#             d, m, s = v
#             return _num(d) + _num(m)/60.0 + _num(s)/3600.0
#         return _num(v)
#     except Exception:
#         return None

# def _exif_latlon_pillow(img: Image.Image):
#     """Try both Pillow EXIF paths: get_ifd(IFD.GPS) and _getexif()[GPSInfo]."""
#     # 1) Newer Pillow path
#     try:
#         ex = img.getexif()
#         IFD = getattr(ExifTags, "IFD", None)
#         if ex and IFD is not None:
#             try:
#                 gps_ifd = ex.get_ifd(IFD.GPS)
#             except Exception:
#                 gps_ifd = None
#             if gps_ifd:
#                 lat_raw = gps_ifd.get(2) or gps_ifd.get("GPSLatitude")
#                 lon_raw = gps_ifd.get(4) or gps_ifd.get("GPSLongitude")
#                 lat_ref = gps_ifd.get(1) or gps_ifd.get("GPSLatitudeRef") or "N"
#                 lon_ref = gps_ifd.get(3) or gps_ifd.get("GPSLongitudeRef") or "E"
#                 lat = _to_deg_any(lat_raw); lon = _to_deg_any(lon_raw)
#                 if lat is not None and lon is not None:
#                     if (lat_ref == "S") or (lat_ref == b"S"): lat = -lat
#                     if (lon_ref == "W") or (lon_ref == b"W"): lon = -lon
#                     return (float(lat), float(lon))
#     except Exception:
#         pass
#     # 2) Legacy Pillow path
#     try:
#         ex2 = getattr(img, "_getexif", lambda: None)()
#         if ex2 and _GPS_TAGS in ex2:
#             gps = ex2[_GPS_TAGS]
#             gps_map = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
#             lat = _to_deg_any(gps_map.get("GPSLatitude"))
#             lon = _to_deg_any(gps_map.get("GPSLongitude"))
#             if lat is not None and lon is not None:
#                 lat_ref = gps_map.get("GPSLatitudeRef", "N")
#                 lon_ref = gps_map.get("GPSLongitudeRef", "E")
#                 if (lat_ref == "S") or (lat_ref == b"S"): lat = -lat
#                 if (lon_ref == "W") or (lon_ref == b"W"): lon = -lon
#                 return (float(lat), float(lon))
#     except Exception:
#         pass
#     return None

# def _xmp_extract_coords(jpeg_bytes: bytes):
#     """Fallback: parse simple XMP packets that store exif:GPSLatitude/Longitude."""
#     try:
#         start = jpeg_bytes.find(b"<x:xmpmeta")
#         if start == -1:
#             return None
#         end = jpeg_bytes.find(b"</x:xmpmeta>", start)
#         if end == -1:
#             return None
#         xmp = jpeg_bytes[start:end+12].decode("utf-8", errors="ignore")

#         def pick(patterns):
#             for p in patterns:
#                 m = re.search(p, xmp)
#                 if m: return m.group(1).strip()
#             return None

#         lat_s = pick([r'GPSLatitude="([^"]+)"', r'exif:GPSLatitude>([^<]+)<'])
#         lon_s = pick([r'GPSLongitude="([^"]+)"', r'exif:GPSLongitude>([^<]+)<'])
#         lat_ref = pick([r'GPSLatitudeRef="([^"]+)"', r'exif:GPSLatitudeRef>([^<]+)<'])
#         lon_ref = pick([r'GPSLongitudeRef="([^"]+)"', r'exif:GPSLongitudeRef>([^<]+)<'])
#         if not (lat_s and lon_s): return None

#         def to_deg(val: str):
#             nums = re.findall(r'[-+]?\d+(?:\.\d+)?', val)
#             if not nums: return None
#             if len(nums) >= 3:
#                 d, m, s = map(float, nums[:3])
#                 return d + m/60 + s/3600
#             return float(nums[0])

#         lat = to_deg(lat_s); lon = to_deg(lon_s)
#         if lat is None or lon is None: return None
#         if (lat_ref or "").upper().startswith("S"): lat = -abs(lat)
#         if (lon_ref or "").upper().startswith("W"): lon = -abs(lon)
#         return (float(lat), float(lon))
#     except Exception:
#         return None

# def extract_gps_from_bytes(data: bytes):
#     # Try EXIF via Pillow (both ways)
#     try:
#         img = Image.open(io.BytesIO(data))
#         hit = _exif_latlon_pillow(img)
#         if hit: return hit
#     except Exception:
#         pass
#     # Fallback: XMP
#     hit = _xmp_extract_coords(data)
#     if hit: return hit
#     return None

# def gmaps_link(lat, lon): return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# # ---- Overpass (nearest ward office) ----
# def _haversine(a,b,c,d):
#     R=6371.0; to=math.pi/180.0
#     dlat,dlon=(c-a)*to,(d-b)*to
#     x=math.sin(dlat/2)**2+math.cos(a*to)*math.cos(c*to)*math.sin(dlon/2)**2
#     return 2*R*math.atan2(math.sqrt(x), math.sqrt(1-x))

# @st.cache_data(ttl=300, show_spinner=False)
# def find_nearest_ward_office(lat: float, lon: float, salt: int = 0):
#     NAME_REGEX = r"Ward Office|City Office|City Hall|Town Hall|区役所|市役所|町役場|村役場|役場|出張所"
#     UA = {"User-Agent":"roadsafe-demo/1.0 (+contact@yourdomain.tld)","Accept":"application/json","Accept-Encoding":"gzip"}
#     MIRRORS = [
#         "https://overpass-api.de/api/interpreter",
#         "https://overpass.kumi.systems/api/interpreter",
#         "https://overpass.openstreetmap.fr/api/interpreter",
#     ]

#     def _mk(r, strict):
#         if strict:
#             return f"""[out:json][timeout:25];
#             (node["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#              way ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#              rel ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon}););
#             out center tags;"""
#         return f"""[out:json][timeout:25];
#         (node["amenity"="townhall"](around:{r},{lat},{lon});
#          way ["amenity"="townhall"](around:{r},{lat},{lon});
#          rel ["amenity"="townhall"](around:{r},{lat},{lon});
#          node["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
#          way ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
#          rel ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
#          node["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#          way ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#          rel ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon}););
#         out center tags;"""

#     def _center(el):
#         if el.get("type") == "node":
#             return (el.get("lat"), el.get("lon"))
#         c = el.get("center") or {}
#         return (c.get("lat"), c.get("lon"))

#     def _addr(tags):
#         parts = [tags.get("addr:postcode"), tags.get("addr:state") or tags.get("addr:province"),
#                  tags.get("addr:city"), tags.get("addr:district") or tags.get("addr:ward") or tags.get("addr:suburb"),
#                  tags.get("addr:street"), tags.get("addr:block_number"), tags.get("addr:housenumber")]
#         s = " ".join([p for p in parts if p]).strip()
#         return s or None

#     def _score(tags, dkm):
#         nm = tags.get("name:en") or tags.get("official_name:en") or tags.get("name") or tags.get("official_name") or tags.get("name:ja")
#         has_name = 1.0 if nm else 0.0
#         amenity = 1.0 if tags.get("amenity") == "townhall" else 0.0
#         admin   = 1.0 if (tags.get("office")=="government" and tags.get("government")=="administrative") else 0.0
#         hit = 1.0 if nm and (any(x in nm for x in ["Ward Office","City Office","City Hall","Town Hall"]) or any(x in nm for x in ["区役所","市役所","町役場","村役場","役場","出張所"])) else 0.0
#         addr = 1.0 if _addr(tags) else 0.0
#         email= 1.0 if (tags.get("contact:email") or tags.get("email")) else 0.0
#         base = 3*amenity + 2.5*hit + 1.5*admin + 1.0*has_name + 0.5*addr + 0.25*email
#         return base - dkm

#     def _best(elems):
#         best=None
#         for el in elems:
#             latc, lonc = _center(el)
#             if latc is None or lonc is None: continue
#             dkm = _haversine(lat, lon, latc, lonc)
#             tags = el.get("tags", {}) or {}
#             sc   = _score(tags, dkm)
#             item = {"name": tags.get("name:en") or tags.get("official_name:en") or tags.get("name") or tags.get("official_name") or tags.get("name:ja") or "(Unnamed Government Office)",
#                     "lat": float(latc), "lon": float(lonc), "distance_km": float(dkm),
#                     "address": _addr(tags), "email": tags.get("contact:email") or tags.get("email"), "score": sc}
#             if best is None or item["score"] > best["score"]:
#                 best = item
#         return best

#     for r in (1500, 3000, 6000, 12000, 20000, 30000):
#         for strict in (True, False):
#             q = _mk(r, strict)
#             for base in MIRRORS:
#                 try:
#                     js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
#                 except Exception:
#                     continue
#                 best = _best(js.get("elements", []))
#                 if best: return best

#     # fallback using geocodeArea
#     try:
#         nom = requests.get("https://nominatim.openstreetmap.org/reverse",
#                            params={"format":"jsonv2","lat":lat,"lon":lon,"zoom":14,"addressdetails":1},
#                            headers=UA, timeout=10).json()
#         a = nom.get("address", {})
#         area = ", ".join([p for p in [a.get("city_district") or a.get("ward"),
#                                       a.get("city") or a.get("town") or a.get("county"),
#                                       a.get("state") or a.get("region"),
#                                       "Japan"] if p])
#         if area:
#             q = f"""[out:json][timeout:25];
#             {{geocodeArea:{area}}}->.a;
#             (node(area.a)["amenity"="townhall"]; way(area.a)["amenity"="townhall"]; rel(area.a)["amenity"="townhall"];
#              node(area.a)["office"="government"]["government"="administrative"];
#              way(area.a)["office"="government"]["government"="administrative"];
#              rel(area.a)["office"="government"]["government"="administrative"];
#              node(area.a)["name"~"{NAME_REGEX}"]; way(area.a)["name"~"{NAME_REGEX}"]; rel(area.a)["name"~"{NAME_REGEX}"];);
#             out center tags;"""
#             for base in MIRRORS:
#                 try:
#                     js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
#                 except Exception:
#                     continue
#                 best = _best(js.get("elements", []))
#                 if best: return best
#     except Exception:
#         pass

#     return None

# # ---- email helpers ----
# def mask_email(e: str) -> str:
#     if not e or "@" not in e: return "hidden@wardoffice.com"
#     local, domain = e.split("@", 1)
#     masked_local = (local[:2] + "***") if len(local) > 2 else (local[:1] + "***")
#     return f"{masked_local}@{domain}"

# def send_email_html(to_email, subject, html_body, text_fallback, image_bytes=None, image_name="image.jpg"):
#     if not GMAIL_USER or not GMAIL_PASSWORD:
#         return False, "Email credentials not found (set GMAIL_USER / GMAIL_PASSWORD)."
#     msg = EmailMessage()
#     msg["From"] = GMAIL_USER
#     msg["To"] = to_email
#     msg["Subject"] = subject
#     msg.set_content(text_fallback, charset="utf-8")
#     msg.add_alternative(html_body, subtype="html")
#     if image_bytes:
#         subtype = (image_name.split(".")[-1] or "jpeg").lower()
#         msg.get_payload()[1].add_related(image_bytes, maintype="image", subtype=subtype, cid="<photo1>")
#         msg.add_attachment(image_bytes, maintype="image", subtype=subtype, filename=image_name)
#     import smtplib
#     try:
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
#             s.login(GMAIL_USER, GMAIL_PASSWORD)
#             s.send_message(msg)
#         return True, "Email sent"
#     except Exception as e:
#         return False, f"{e}"

# # ---- romaji fallback ----
# try:
#     from unidecode import unidecode
# except Exception:
#     def unidecode(x): return x
# def romaji_email_from_name(name: str) -> str:
#     base = unidecode(name or "").lower()
#     base = re.sub(r"[^a-z0-9]+", "", base) or "wardoffice"
#     return f"{base}@wardoffice.com"

# # ---- state ----
# defaults = dict(
#     pred=None, img_bytes=None, img_name=None,
#     live_point=None, confirmed_point=None,
#     map_center=TOKYO_DEFAULT, map_zoom=12,
#     location_confirmed=False, ward_office=None, report_sent=False,
#     allow_submit_anyway=False,
#     ward_nonce=0,
#     uploader_nonce=0,
#     metadata_pin=False,
# )
# for k,v in defaults.items(): st.session_state.setdefault(k,v)

# def _reset_to_upload():
#     for k in defaults:
#         st.session_state[k] = defaults[k]
#     st.session_state.uploader_nonce += 1  # force uploader to clear

# # ---- header ----
# st.markdown("""
# <div class='center'>
#   <h1 style="margin-bottom:0.2rem;">🛣️ RoadSafe — Report Road Damage</h1>
#   <p class='muted' style='margin-top:0'>Upload → Validate → Pinpoint → Notify Ward Office</p>
# </div>
# """, unsafe_allow_html=True)

# # ---- upload → predict (GPS extracted from raw bytes: EXIF, then XMP) ----
# uploader = st.file_uploader(
#     "Upload a road photo (JPG/PNG)",
#     type=["jpg","jpeg","png"],
#     key=f"uploader-{st.session_state.uploader_nonce}",
# )
# if uploader is not None:
#     raw_bytes = uploader.getvalue()
#     img_raw = Image.open(io.BytesIO(raw_bytes))
#     gps = extract_gps_from_bytes(raw_bytes)

#     img = img_raw.convert("RGB")  # display / ML
#     st.session_state.img_bytes = raw_bytes
#     st.session_state.img_name = uploader.name
#     st.image(img, caption="Input image", use_container_width=True)

#     # Auto-pin & center if GPS found (user still confirms)
#     if gps and st.session_state.live_point is None:
#         pt = (float(gps[0]), float(gps[1]))
#         st.session_state.live_point = pt
#         st.session_state.map_center = pt
#         st.session_state.map_zoom = 16
#         st.session_state.metadata_pin = True

#     with st.spinner("Analyzing the image…"):
#         session = get_model_session()
#         st.session_state.pred = session(img)
#     st.session_state.report_sent = False

# # ---- results ----
# if st.session_state.pred:
#     res = st.session_state.pred
#     type_txt = res["surface_type"].replace("_"," ").title()
#     qual_raw = res["surface_quality"]
#     qual_txt = qual_raw.replace("_"," ").title()
#     qcls = "qual-ok" if qual_raw in ("excellent","good") else ("qual-mid" if qual_raw=="intermediate" else "qual-bad")

#     st.markdown("<div class='section-title'>Predicted Surface Type</div>", unsafe_allow_html=True)
#     st.markdown(f"<div class='result-value'>{type_txt}</div>", unsafe_allow_html=True)

#     st.markdown("<div class='section-title'>Predicted Surface Quality</div>", unsafe_allow_html=True)
#     st.markdown(f"<div class='result-value {qcls}'>{qual_txt}</div>", unsafe_allow_html=True)

#     if qual_raw in {"excellent", "good", "intermediate"} and not st.session_state.allow_submit_anyway:
#         st.info("This road seems to be in a decent state. Are you sure you want to report it?")
#         st.session_state.allow_submit_anyway = st.toggle("Report anyway", value=st.session_state.allow_submit_anyway)
#         if not st.session_state.allow_submit_anyway:
#             st.stop()

#     st.divider()
#     st.subheader("Confirm location of the damage")

#     if st.session_state.live_point is None:
#         st.warning("Location not found from your picture — please click on the map to drop a pin.")
#     elif st.session_state.metadata_pin:
#         st.info("Location read from your photo (GPS metadata). Please confirm, or click on the map to adjust.")
#         st.session_state.metadata_pin = False

#     @st.fragment
#     def map_and_followups():
#         # MAP (do not persist center/zoom so panning/zooming doesn't trigger reruns)
#         m = folium.Map(
#             location=st.session_state.map_center,
#             zoom_start=st.session_state.map_zoom,
#             control_scale=True,
#             tiles="OpenStreetMap"
#         )
#         if st.session_state.live_point:
#             folium.Marker(st.session_state.live_point, tooltip="Selected location").add_to(m)

#         out = st_folium(m, width=720, height=420, key="map-main", returned_objects=["last_clicked"])
#         if out and out.get("last_clicked"):
#             click = out["last_clicked"]
#             if click.get("lat") and click.get("lng"):
#                 st.session_state.live_point = (float(click["lat"]), float(click["lng"]))
#                 st.session_state.map_center = st.session_state.live_point
#                 st.session_state.location_confirmed = False
#                 st.session_state.confirmed_point = None
#                 st.session_state.ward_office = None

#         if st.session_state.live_point:
#             lat, lon = st.session_state.live_point
#             st.caption(f"Chosen coordinates: <span class='coords'>{lat:.6f}, {lon:.6f}</span>", unsafe_allow_html=True)

#         st.button(
#             "Confirm location",
#             key="confirm-location-btn",
#             type="primary",
#             use_container_width=True,
#             disabled=st.session_state.live_point is None,
#             on_click=lambda: st.session_state.update(
#                 location_confirmed=True,
#                 confirmed_point=st.session_state.live_point
#             )
#         )

#         if st.session_state.location_confirmed and st.session_state.confirmed_point and not st.session_state.report_sent:
#             st.divider()
#             st.subheader("Closest ward/city office  ↪")

#             if st.session_state.ward_office is None:
#                 with st.spinner("Looking up nearest office…"):
#                     latc, lonc = st.session_state.confirmed_point
#                     office = find_nearest_ward_office(latc, lonc, st.session_state.ward_nonce)
#                     st.session_state.ward_office = office

#             office = st.session_state.ward_office
#             if office:
#                 st.success(f"Closest office: **{office['name']}**")
#                 st.caption(f"📍 {office['lat']:.6f}, {office['lon']:.6f}")
#                 st.link_button("Open office in Google Maps", gmaps_link(office['lat'], office['lon']), use_container_width=True)
#             else:
#                 st.warning("Couldn’t identify a nearby office automatically. You can still send the report (coordinates included).")
#                 st.button("Retry lookup", key="retry-ward", on_click=lambda: st.session_state.update(ward_nonce=st.session_state.ward_nonce + 1, ward_office=None))

#             # SEND REPORT
#             st.divider()
#             st.subheader("Send the report")

#             if office and office.get("email"):
#                 display_recipient = office["email"]
#             else:
#                 display_recipient = romaji_email_from_name(office["name"] if office else "Ward Office")
#             st.caption("Email of the ward office — hidden to avoid spam.")
#             st.text_input("Recipient", value=mask_email(display_recipient), disabled=True)

#             actual_recipient = DEMO_RECEIVER or GMAIL_USER or display_recipient
#             type_txt = res["surface_type"].replace("_"," ").title()
#             qual_txt = res["surface_quality"].replace("_"," ").title()
#             latc, lonc = st.session_state.confirmed_point
#             when = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#             if office:
#                 office_addr_or_link = office.get('address') or gmaps_link(office['lat'], office['lon'])
#                 office_name = office.get('name', 'N/A')
#             else:
#                 office_addr_or_link = gmaps_link(latc, lonc)
#                 office_name = 'N/A'

#             core_text = f"""RoadSafe report (demo)
# When: {when}
# Surface type: {type_txt}
# Surface quality: {qual_txt}
# Location: {latc:.6f}, {lonc:.6f}
# Google Maps: {gmaps_link(latc, lonc)}
# Ward office: {office_name}
# Address: {office_addr_or_link}
# """
#             st.text_area("Preview (read-only)", value=core_text, height=160, disabled=True)
#             comment = st.text_area("Additional comments (optional)", placeholder="Add any useful details…")

#             html = f"""
#             <div style="font-family:system-ui;">
#               <h2>RoadSafe report (demo)</h2>
#               <p style="color:#555">When: {when}</p>
#               <img src="cid:photo1" style="max-width:640px;border-radius:8px;margin:8px 0 16px"/>
#               <ul style="line-height:1.6">
#                 <li><b>Surface type:</b> {type_txt}</li>
#                 <li><b>Surface quality:</b> {qual_txt}</li>
#                 <li><b>Location:</b> {latc:.6f}, {lonc:.6f}</li>
#                 <li><b>Google Maps:</b> <a href="{gmaps_link(latc,lonc)}">{gmaps_link(latc,lonc)}</a></li>
#                 <li><b>Ward office:</b> {office_name}</li>
#                 <li><b>Address:</b> {office_addr_or_link}</li>
#               </ul>
#               {"<p><b>Additional comments:</b><br>"+comment.replace('\\n','<br>')+"</p>" if comment.strip() else ""}
#               <p class="tiny">This message was generated for demo purposes.</p>
#             </div>
#             """.strip()

#             text = core_text + (f"\nAdditional comments:\n{comment}\n" if comment.strip() else "")

#             if st.button("Send report", type="primary", use_container_width=True, key="send-report-btn"):
#                 ok, info = send_email_html(
#                     to_email=actual_recipient,
#                     subject="RoadSafe Report (Demo)",
#                     html_body=html,
#                     text_fallback=text,
#                     image_bytes=st.session_state.img_bytes,
#                     image_name=st.session_state.img_name or "photo.jpg",
#                 )
#                 if ok:
#                     st.success("Report submitted successfully, thank you for your contribution.")
#                     time.sleep(2)
#                     st.toast("The current page will now refresh — feel free to submit another damage!", icon="✅")
#                     time.sleep(5)
#                     _reset_to_upload()
#                     st.rerun()
#                 else:
#                     st.error(f"Failed to send email: {info}")

#     map_and_followups()

# # ---- footer ----
# st.divider()
# st.caption("Environment variables required: HF_REPO_ID, HF_FILENAME, GMAIL_USER, GMAIL_PASSWORD, optional DEMO_RECEIVER.")



# # RoadSafe — Streamlit app (stable map + robust EXIF/XMP GPS + confirm-to-lookup + hard reset)
# # --------------------------------------------------------------------------------------------
# import io, os, re, math, gzip, time
# from datetime import datetime
# from email.message import EmailMessage

# import streamlit as st
# from PIL import Image, ExifTags
# import requests
# import folium
# from streamlit_folium import st_folium

# # ML
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
# from huggingface_hub import hf_hub_download

# # ---- env ----
# from dotenv import load_dotenv
# load_dotenv()
# GMAIL_USER      = os.getenv("GMAIL_USER")
# GMAIL_PASSWORD  = os.getenv("GMAIL_PASSWORD")
# HF_REPO_ID      = os.getenv("HF_REPO_ID") or "esdk/my-efficientnet-model"
# HF_FILENAME     = os.getenv("HF_FILENAME") or "efficientnet_fp16.pt.gz"
# HF_TOKEN        = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
# DEMO_RECEIVER   = os.getenv("DEMO_RECEIVER")

# # ---- page & css ----
# st.set_page_config(page_title="RoadSafe — Report a Road Issue", page_icon="🛣️", layout="centered")
# st.markdown("""
# <style>
#   .center {text-align:center}
#   .muted {opacity:.75}
#   .section-title {font-size:1.05rem; font-weight:700; margin: 10px 0 4px}
#   .result-value {font-size:2.2rem; font-weight:800; margin: 0 0 6px}
#   .qual-ok  {color:#22c55e}
#   .qual-mid {color:#a16207}
#   .qual-bad {color:#ef4444}
#   .coords {font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace}
#   .folium-map { margin-bottom: 0 !important; }
#   div[data-testid="stVerticalBlock"] > div:has(.folium-map) { margin-bottom: 0 !important; }
#   .tiny {font-size: .85rem; opacity: .7}
# </style>
# """, unsafe_allow_html=True)

# # ---- labels/defaults ----
# MATERIAL_NAMES = ["asphalt", "concrete", "paving_stones", "unpaved", "sett"]
# QUALITY_NAMES  = ["excellent", "good", "intermediate", "bad", "very_bad"]
# TOKYO_DEFAULT  = (35.681236, 139.767125)

# # === MODEL ===
# class MultiHeadEffB7(nn.Module):
#     def __init__(self, n_type=len(MATERIAL_NAMES), n_qual=len(QUALITY_NAMES)):
#         super().__init__()
#         base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
#         self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
#         in_f = base.classifier[1].in_features
#         self.mat  = nn.Linear(in_f, n_type)
#         self.qual = nn.Linear(in_f, n_qual)
#     def forward(self, x):
#         z = self.features(x)
#         return self.mat(z), self.qual(z)

# @st.cache_resource(show_spinner=True)
# def get_model_session():
#     if not HF_REPO_ID or not HF_FILENAME:
#         st.error("HF_REPO_ID / HF_FILENAME not set in env."); st.stop()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultiHeadEffB7().to(device)

#     path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=HF_TOKEN)
#     if path.endswith(".gz"):
#         with gzip.open(path, "rb") as f:
#             state = torch.load(io.BytesIO(f.read()), map_location=device)
#     else:
#         state = torch.load(path, map_location=device)
#     if isinstance(state, dict) and "state_dict" in state:
#         state = state["state_dict"]
#     state = {k.replace("_orig_mod.","").replace("module.",""): v for k, v in state.items()}
#     model.load_state_dict(state, strict=False)
#     model.eval()

#     tfm = transforms.Compose([
#         transforms.Resize((600, 600)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ])

#     @torch.inference_mode()
#     def session(pil_img):
#         x = tfm(pil_img).unsqueeze(0).to(device)
#         t_logits, q_logits = model(x)
#         t_idx = int(torch.softmax(t_logits, 1).argmax())
#         q_idx = int(torch.softmax(q_logits, 1).argmax())
#         return {"surface_type": MATERIAL_NAMES[t_idx], "surface_quality": QUALITY_NAMES[q_idx]}
#     return session

# # ---- GPS extraction (EXIF first, then XMP fallback) ----
# _GPS_TAGS = 34853  # GPSInfo

# def _num(x):
#     try:
#         return float(getattr(x, "numerator", x)) / float(getattr(x, "denominator", 1))
#     except Exception:
#         try:
#             a, b = x
#             return float(a) / float(b)
#         except Exception:
#             return float(x)

# def _to_deg_any(v):
#     try:
#         if isinstance(v, (list, tuple)) and len(v) == 3:
#             d, m, s = v
#             return _num(d) + _num(m)/60.0 + _num(s)/3600.0
#         return _num(v)
#     except Exception:
#         return None

# def _exif_latlon_pillow(img: Image.Image):
#     """Try both Pillow EXIF paths: get_ifd(IFD.GPS) and _getexif()[GPSInfo]."""
#     # 1) Newer Pillow path
#     try:
#         ex = img.getexif()
#         IFD = getattr(ExifTags, "IFD", None)
#         if ex and IFD is not None:
#             try:
#                 gps_ifd = ex.get_ifd(IFD.GPS)
#             except Exception:
#                 gps_ifd = None
#             if gps_ifd:
#                 lat_raw = gps_ifd.get(2) or gps_ifd.get("GPSLatitude")
#                 lon_raw = gps_ifd.get(4) or gps_ifd.get("GPSLongitude")
#                 lat_ref = gps_ifd.get(1) or gps_ifd.get("GPSLatitudeRef") or "N"
#                 lon_ref = gps_ifd.get(3) or gps_ifd.get("GPSLongitudeRef") or "E"
#                 lat = _to_deg_any(lat_raw); lon = _to_deg_any(lon_raw)
#                 if lat is not None and lon is not None:
#                     if (lat_ref == "S") or (lat_ref == b"S"): lat = -lat
#                     if (lon_ref == "W") or (lon_ref == b"W"): lon = -lon
#                     return (float(lat), float(lon))
#     except Exception:
#         pass
#     # 2) Legacy Pillow path
#     try:
#         ex2 = getattr(img, "_getexif", lambda: None)()
#         if ex2 and _GPS_TAGS in ex2:
#             gps = ex2[_GPS_TAGS]
#             gps_map = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
#             lat = _to_deg_any(gps_map.get("GPSLatitude"))
#             lon = _to_deg_any(gps_map.get("GPSLongitude"))
#             if lat is not None and lon is not None:
#                 lat_ref = gps_map.get("GPSLatitudeRef", "N")
#                 lon_ref = gps_map.get("GPSLongitudeRef", "E")
#                 if (lat_ref == "S") or (lat_ref == b"S"): lat = -lat
#                 if (lon_ref == "W") or (lon_ref == b"W"): lon = -lon
#                 return (float(lat), float(lon))
#     except Exception:
#         pass
#     return None

# def _xmp_extract_coords(jpeg_bytes: bytes):
#     """Fallback: parse simple XMP packets that store exif:GPSLatitude/Longitude."""
#     try:
#         start = jpeg_bytes.find(b"<x:xmpmeta")
#         if start == -1:
#             return None
#         end = jpeg_bytes.find(b"</x:xmpmeta>", start)
#         if end == -1:
#             return None
#         xmp = jpeg_bytes[start:end+12].decode("utf-8", errors="ignore")

#         def pick(patterns):
#             for p in patterns:
#                 m = re.search(p, xmp)
#                 if m: return m.group(1).strip()
#             return None

#         lat_s = pick([r'GPSLatitude="([^"]+)"', r'exif:GPSLatitude>([^<]+)<'])
#         lon_s = pick([r'GPSLongitude="([^"]+)"', r'exif:GPSLongitude>([^<]+)<'])
#         lat_ref = pick([r'GPSLatitudeRef="([^"]+)"', r'exif:GPSLatitudeRef>([^<]+)<'])
#         lon_ref = pick([r'GPSLongitudeRef="([^"]+)"', r'exif:GPSLongitudeRef>([^<]+)<'])
#         if not (lat_s and lon_s): return None

#         def to_deg(val: str):
#             nums = re.findall(r'[-+]?\d+(?:\.\d+)?', val)
#             if not nums: return None
#             if len(nums) >= 3:
#                 d, m, s = map(float, nums[:3])
#                 return d + m/60 + s/3600
#             return float(nums[0])

#         lat = to_deg(lat_s); lon = to_deg(lon_s)
#         if lat is None or lon is None: return None
#         if (lat_ref or "").upper().startswith("S"): lat = -abs(lat)
#         if (lon_ref or "").upper().startswith("W"): lon = -abs(lon)
#         return (float(lat), float(lon))
#     except Exception:
#         return None

# def extract_gps_from_bytes(data: bytes):
#     # Try EXIF via Pillow (both ways)
#     try:
#         img = Image.open(io.BytesIO(data))
#         hit = _exif_latlon_pillow(img)
#         if hit: return hit
#     except Exception:
#         pass
#     # Fallback: XMP
#     hit = _xmp_extract_coords(data)
#     if hit: return hit
#     return None

# def gmaps_link(lat, lon): return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# # ---- Overpass (nearest ward office) ----
# def _haversine(a,b,c,d):
#     R=6371.0; to=math.pi/180.0
#     dlat,dlon=(c-a)*to,(d-b)*to
#     x=math.sin(dlat/2)**2+math.cos(a*to)*math.cos(c*to)*math.sin(dlon/2)**2
#     return 2*R*math.atan2(math.sqrt(x), math.sqrt(1-x))

# @st.cache_data(ttl=300, show_spinner=False)
# def find_nearest_ward_office(lat: float, lon: float, salt: int = 0):
#     NAME_REGEX = r"Ward Office|City Office|City Hall|Town Hall|区役所|市役所|町役場|村役場|役場|出張所"
#     UA = {"User-Agent":"roadsafe-demo/1.0 (+contact@yourdomain.tld)","Accept":"application/json","Accept-Encoding":"gzip"}
#     MIRRORS = [
#         "https://overpass-api.de/api/interpreter",
#         "https://overpass.kumi.systems/api/interpreter",
#         "https://overpass.openstreetmap.fr/api/interpreter",
#     ]

#     def _mk(r, strict):
#         if strict:
#             return f"""[out:json][timeout:25];
#             (node["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#              way ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#              rel ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon}););
#             out center tags;"""
#         return f"""[out:json][timeout:25];
#         (node["amenity"="townhall"](around:{r},{lat},{lon});
#          way ["amenity"="townhall"](around:{r},{lat},{lon});
#          rel ["amenity"="townhall"](around:{r},{lat},{lon});
#          node["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
#          way ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
#          rel ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
#          node["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#          way ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#          rel ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon}););
#         out center tags;"""

#     def _center(el):
#         if el.get("type") == "node":
#             return (el.get("lat"), el.get("lon"))
#         c = el.get("center") or {}
#         return (c.get("lat"), c.get("lon"))

#     def _addr(tags):
#         parts = [tags.get("addr:postcode"), tags.get("addr:state") or tags.get("addr:province"),
#                  tags.get("addr:city"), tags.get("addr:district") or tags.get("addr:ward") or tags.get("addr:suburb"),
#                  tags.get("addr:street"), tags.get("addr:block_number"), tags.get("addr:housenumber")]
#         s = " ".join([p for p in parts if p]).strip()
#         return s or None

#     def _score(tags, dkm):
#         nm = tags.get("name:en") or tags.get("official_name:en") or tags.get("name") or tags.get("official_name") or tags.get("name:ja")
#         has_name = 1.0 if nm else 0.0
#         amenity = 1.0 if tags.get("amenity") == "townhall" else 0.0
#         admin   = 1.0 if (tags.get("office")=="government" and tags.get("government")=="administrative") else 0.0
#         hit = 1.0 if nm and (any(x in nm for x in ["Ward Office","City Office","City Hall","Town Hall"]) or any(x in nm for x in ["区役所","市役所","町役場","村役場","役場","出張所"])) else 0.0
#         addr = 1.0 if _addr(tags) else 0.0
#         email= 1.0 if (tags.get("contact:email") or tags.get("email")) else 0.0
#         base = 3*amenity + 2.5*hit + 1.5*admin + 1.0*has_name + 0.5*addr + 0.25*email
#         return base - dkm

#     def _best(elems):
#         best=None
#         for el in elems:
#             latc, lonc = _center(el)
#             if latc is None or lonc is None: continue
#             dkm = _haversine(lat, lon, latc, lonc)
#             tags = el.get("tags", {}) or {}
#             sc   = _score(tags, dkm)
#             item = {"name": tags.get("name:en") or tags.get("official_name:en") or tags.get("name") or tags.get("official_name") or tags.get("name:ja") or "(Unnamed Government Office)",
#                     "lat": float(latc), "lon": float(lonc), "distance_km": float(dkm),
#                     "address": _addr(tags), "email": tags.get("contact:email") or tags.get("email"), "score": sc}
#             if best is None or item["score"] > best["score"]:
#                 best = item
#         return best

#     for r in (1500, 3000, 6000, 12000, 20000, 30000):
#         for strict in (True, False):
#             q = _mk(r, strict)
#             for base in MIRRORS:
#                 try:
#                     js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
#                 except Exception:
#                     continue
#                 best = _best(js.get("elements", []))
#                 if best: return best

#     # fallback using geocodeArea
#     try:
#         nom = requests.get("https://nominatim.openstreetmap.org/reverse",
#                            params={"format":"jsonv2","lat":lat,"lon":lon,"zoom":14,"addressdetails":1},
#                            headers=UA, timeout=10).json()
#         a = nom.get("address", {})
#         area = ", ".join([p for p in [a.get("city_district") or a.get("ward"),
#                                       a.get("city") or a.get("town") or a.get("county"),
#                                       a.get("state") or a.get("region"),
#                                       "Japan"] if p])
#         if area:
#             q = f"""[out:json][timeout:25];
#             {{geocodeArea:{area}}}->.a;
#             (node(area.a)["amenity"="townhall"]; way(area.a)["amenity"="townhall"]; rel(area.a)["amenity"="townhall"];
#              node(area.a)["office"="government"]["government"="administrative"];
#              way(area.a)["office"="government"]["government"="administrative"];
#              rel(area.a)["office"="government"]["government"="administrative"];
#              node(area.a)["name"~"{NAME_REGEX}"]; way(area.a)["name"~"{NAME_REGEX}"]; rel(area.a)["name"~"{NAME_REGEX}"];);
#             out center tags;"""
#             for base in MIRRORS:
#                 try:
#                     js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
#                 except Exception:
#                     continue
#                 best = _best(js.get("elements", []))
#                 if best: return best
#     except Exception:
#         pass
#     return None

# # ---- email helpers ----
# def mask_email(e: str) -> str:
#     if not e or "@" not in e: return "hidden@wardoffice.com"
#     local, domain = e.split("@", 1)
#     masked_local = (local[:2] + "***") if len(local) > 2 else (local[:1] + "***")
#     return f"{masked_local}@{domain}"

# def send_email_html(to_email, subject, html_body, text_fallback, image_bytes=None, image_name="image.jpg"):
#     if not GMAIL_USER or not GMAIL_PASSWORD:
#         return False, "Email credentials not found (set GMAIL_USER / GMAIL_PASSWORD)."
#     msg = EmailMessage()
#     msg["From"] = GMAIL_USER
#     msg["To"] = to_email
#     msg["Subject"] = subject
#     msg.set_content(text_fallback, charset="utf-8")
#     msg.add_alternative(html_body, subtype="html")
#     if image_bytes:
#         subtype = (image_name.split(".")[-1] or "jpeg").lower()
#         msg.get_payload()[1].add_related(image_bytes, maintype="image", subtype=subtype, cid="<photo1>")
#         msg.add_attachment(image_bytes, maintype="image", subtype=subtype, filename=image_name)
#     import smtplib
#     try:
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
#             s.login(GMAIL_USER, GMAIL_PASSWORD)
#             s.send_message(msg)
#         return True, "Email sent"
#     except Exception as e:
#         return False, f"{e}"

# # ---- romaji fallback ----
# try:
#     from unidecode import unidecode
# except Exception:
#     def unidecode(x): return x
# def romaji_email_from_name(name: str) -> str:
#     base = unidecode(name or "").lower()
#     base = re.sub(r"[^a-z0-9]+", "", base) or "wardoffice"
#     return f"{base}@wardoffice.com"

# # ---- hard reset helper ----
# def hard_reset_all():
#     """Hard reset: wipe session state, clear caches, bust URL so widgets fully reset, then rerun."""
#     st.session_state.clear()
#     try:
#         st.cache_data.clear()
#     except Exception:
#         pass
#     ts = str(int(time.time()))
#     try:
#         st.query_params["_"] = ts
#     except Exception:
#         try:
#             st.experimental_set_query_params(**{"_": ts})
#         except Exception:
#             pass
#     st.rerun()

# # ---- state ----
# defaults = dict(
#     pred=None, img_bytes=None, img_name=None,
#     live_point=None, confirmed_point=None,
#     map_center=TOKYO_DEFAULT, map_zoom=12,
#     location_confirmed=False, ward_office=None, report_sent=False,
#     allow_submit_anyway=False,
#     ward_nonce=0,
#     uploader_nonce=0,
#     metadata_pin=False,
# )
# for k,v in defaults.items(): st.session_state.setdefault(k,v)

# def _reset_to_upload():
#     for k in defaults:
#         st.session_state[k] = defaults[k]
#     st.session_state.uploader_nonce += 1  # force uploader to clear

# # ---- header ----
# st.markdown("""
# <div class='center'>
#   <h1 style="margin-bottom:0.2rem;">🛣️ RoadSafe — Report Road Damage</h1>
#   <p class='muted' style='margin-top:0'>Upload → Validate → Pinpoint → Notify Ward Office</p>
# </div>
# """, unsafe_allow_html=True)

# # ---- upload → predict (GPS from EXIF/XMP) ----
# uploader = st.file_uploader(
#     "Upload a road photo (JPG/PNG)",
#     type=["jpg","jpeg","png"],
#     key=f"uploader-{st.session_state.uploader_nonce}",
# )
# if uploader is not None:
#     raw_bytes = uploader.getvalue()
#     img_raw = Image.open(io.BytesIO(raw_bytes))
#     gps = extract_gps_from_bytes(raw_bytes)

#     img = img_raw.convert("RGB")
#     st.session_state.img_bytes = raw_bytes
#     st.session_state.img_name = uploader.name
#     st.image(img, caption="Input image", use_container_width=True)

#     # Auto-pin & center if GPS found (user still confirms)
#     if gps and st.session_state.live_point is None:
#         pt = (float(gps[0]), float(gps[1]))
#         st.session_state.live_point = pt
#         st.session_state.map_center = pt
#         st.session_state.map_zoom = 16
#         st.session_state.metadata_pin = True

#     with st.spinner("Analyzing the image…"):
#         session = get_model_session()
#         st.session_state.pred = session(img)
#     st.session_state.report_sent = False

# # ---- results ----
# if st.session_state.pred:
#     res = st.session_state.pred
#     type_txt = res["surface_type"].replace("_"," ").title()
#     qual_raw = res["surface_quality"]
#     qual_txt = qual_raw.replace("_"," ").title()
#     qcls = "qual-ok" if qual_raw in ("excellent","good") else ("qual-mid" if qual_raw=="intermediate" else "qual-bad")

#     st.markdown("<div class='section-title'>Predicted Surface Type</div>", unsafe_allow_html=True)
#     st.markdown(f"<div class='result-value'>{type_txt}</div>", unsafe_allow_html=True)

#     st.markdown("<div class='section-title'>Predicted Surface Quality</div>", unsafe_allow_html=True)
#     st.markdown(f"<div class='result-value {qcls}'>{qual_txt}</div>", unsafe_allow_html=True)

#     if qual_raw in {"excellent", "good", "intermediate"} and not st.session_state.allow_submit_anyway:
#         st.info("This road seems to be in a decent state. Are you sure you want to report it?")
#         st.session_state.allow_submit_anyway = st.toggle("Report anyway", value=st.session_state.allow_submit_anyway)
#         if not st.session_state.allow_submit_anyway:
#             st.stop()

#     st.divider()
#     st.subheader("Confirm location of the damage")

#     if st.session_state.live_point is None:
#         st.warning("Location not found from your picture — please click on the map to drop a pin.")
#     elif st.session_state.metadata_pin:
#         st.info("Location read from your photo (GPS metadata). Please confirm, or click on the map to adjust.")
#         st.session_state.metadata_pin = False

#     @st.fragment
#     def map_and_followups():
#         # MAP (no center/zoom persistence ⇒ no reruns while panning/zooming)
#         m = folium.Map(
#             location=st.session_state.map_center,
#             zoom_start=st.session_state.map_zoom,
#             control_scale=True,
#             tiles="OpenStreetMap"
#         )
#         if st.session_state.live_point:
#             folium.Marker(st.session_state.live_point, tooltip="Selected location").add_to(m)

#         out = st_folium(m, width=720, height=420, key="map-main", returned_objects=["last_clicked"])
#         if out and out.get("last_clicked"):
#             click = out["last_clicked"]
#             if click.get("lat") and click.get("lng"):
#                 st.session_state.live_point = (float(click["lat"]), float(click["lng"]))
#                 st.session_state.map_center = st.session_state.live_point
#                 st.session_state.location_confirmed = False
#                 st.session_state.confirmed_point = None
#                 st.session_state.ward_office = None

#         if st.session_state.live_point:
#             lat, lon = st.session_state.live_point
#             st.caption(f"Chosen coordinates: <span class='coords'>{lat:.6f}, {lon:.6f}</span>", unsafe_allow_html=True)

#         st.button(
#             "Confirm location",
#             key="confirm-location-btn",
#             type="primary",
#             use_container_width=True,
#             disabled=st.session_state.live_point is None,
#             on_click=lambda: st.session_state.update(
#                 location_confirmed=True,
#                 confirmed_point=st.session_state.live_point
#             )
#         )

#         if st.session_state.location_confirmed and st.session_state.confirmed_point and not st.session_state.report_sent:
#             st.divider()
#             st.subheader("Closest ward/city office  ↪")

#             if st.session_state.ward_office is None:
#                 with st.spinner("Looking up nearest office…"):
#                     latc, lonc = st.session_state.confirmed_point
#                     office = find_nearest_ward_office(latc, lonc, st.session_state.ward_nonce)
#                     st.session_state.ward_office = office

#             office = st.session_state.ward_office
#             if office:
#                 st.success(f"Closest office: **{office['name']}**")
#                 st.caption(f"📍 {office['lat']:.6f}, {office['lon']:.6f}")
#                 st.link_button("Open office in Google Maps", gmaps_link(office['lat'], office['lon']), use_container_width=True)
#             else:
#                 st.warning("Couldn’t identify a nearby office automatically. You can still send the report (coordinates included).")
#                 st.button("Retry lookup", key="retry-ward", on_click=lambda: st.session_state.update(ward_nonce=st.session_state.ward_nonce + 1, ward_office=None))

#             # SEND REPORT
#             st.divider()
#             st.subheader("Send the report")

#             if office and office.get("email"):
#                 display_recipient = office["email"]
#             else:
#                 display_recipient = romaji_email_from_name(office["name"] if office else "Ward Office")
#             st.caption("Email of the ward office — hidden to avoid spam.")
#             st.text_input("Recipient", value=mask_email(display_recipient), disabled=True)

#             actual_recipient = DEMO_RECEIVER or GMAIL_USER or display_recipient
#             type_txt = res["surface_type"].replace("_"," ").title()
#             qual_txt = res["surface_quality"].replace("_"," ").title()
#             latc, lonc = st.session_state.confirmed_point
#             when = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#             if office:
#                 office_addr_or_link = office.get('address') or gmaps_link(office['lat'], office['lon'])
#                 office_name = office.get('name', 'N/A')
#             else:
#                 office_addr_or_link = gmaps_link(latc, lonc)
#                 office_name = 'N/A'

#             core_text = f"""RoadSafe report (demo)
# When: {when}
# Surface type: {type_txt}
# Surface quality: {qual_txt}
# Location: {latc:.6f}, {lonc:.6f}
# Google Maps: {gmaps_link(latc, lonc)}
# Ward office: {office_name}
# Address: {office_addr_or_link}
# """
#             st.text_area("Preview (read-only)", value=core_text, height=160, disabled=True)
#             comment = st.text_area("Additional comments (optional)", placeholder="Add any useful details…")

#             html = f"""
#             <div style="font-family:system-ui;">
#               <h2>RoadSafe report (demo)</h2>
#               <p style="color:#555">When: {when}</p>
#               <img src="cid:photo1" style="max-width:640px;border-radius:8px;margin:8px 0 16px"/>
#               <ul style="line-height:1.6">
#                 <li><b>Surface type:</b> {type_txt}</li>
#                 <li><b>Surface quality:</b> {qual_txt}</li>
#                 <li><b>Location:</b> {latc:.6f}, {lonc:.6f}</li>
#                 <li><b>Google Maps:</b> <a href="{gmaps_link(latc,lonc)}">{gmaps_link(latc,lonc)}</a></li>
#                 <li><b>Ward office:</b> {office_name}</li>
#                 <li><b>Address:</b> {office_addr_or_link}</li>
#               </ul>
#               {"<p><b>Additional comments:</b><br>"+comment.replace('\\n','<br>')+"</p>" if comment.strip() else ""}
#               <p class="tiny">This message was generated for demo purposes.</p>
#             </div>
#             """.strip()

#             text = core_text + (f"\nAdditional comments:\n{comment}\n" if comment.strip() else "")

#             if st.button("Send report", type="primary", use_container_width=True, key="send-report-btn"):
#                 ok, info = send_email_html(
#                     to_email=actual_recipient,
#                     subject="RoadSafe Report (Demo)",
#                     html_body=html,
#                     text_fallback=text,
#                     image_bytes=st.session_state.img_bytes,
#                     image_name=st.session_state.img_name or "photo.jpg",
#                 )
#                 if ok:
#                     st.success("Report submitted successfully, thank you for your contribution.")
#                     time.sleep(2)
#                     st.toast("The current page will now refresh — feel free to submit another damage!", icon="✅")
#                     time.sleep(5)
#                     hard_reset_all()
#                 else:
#                     st.error(f"Failed to send email: {info}")

#     map_and_followups()

# # ---- footer ----
# st.divider()
# st.caption("Environment variables required: HF_REPO_ID, HF_FILENAME, GMAIL_USER, GMAIL_PASSWORD, optional DEMO_RECEIVER.")




# RoadSafe — Streamlit app (stable map + robust EXIF/XMP GPS + confirm-to-lookup + hard reset)
# --------------------------------------------------------------------------------------------
import io, os, re, math, gzip, time
from datetime import datetime
from email.message import EmailMessage

import streamlit as st
from PIL import Image, ExifTags
import requests
import folium
from streamlit_folium import st_folium

# ML
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from huggingface_hub import hf_hub_download

# ---- env ----
from dotenv import load_dotenv
load_dotenv()
GMAIL_USER      = os.getenv("GMAIL_USER")
GMAIL_PASSWORD  = os.getenv("GMAIL_PASSWORD")
HF_REPO_ID      = os.getenv("HF_REPO_ID") or "esdk/my-efficientnet-model"
HF_FILENAME     = os.getenv("HF_FILENAME") or "efficientnet_fp16.pt.gz"
HF_TOKEN        = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
DEMO_RECEIVER   = os.getenv("DEMO_RECEIVER")

# ---- page & css ----
st.set_page_config(page_title="RoadSafe — Report a Road Issue", page_icon="🛣️", layout="centered")
st.markdown("""
<style>
  .center {text-align:center}
  .muted {opacity:.75}
  .section-title {font-size:1.05rem; font-weight:700; margin: 10px 0 4px}
  .result-value {font-size:2.2rem; font-weight:800; margin: 0 0 6px}
  .qual-ok  {color:#22c55e}
  .qual-mid {color:#a16207}
  .qual-bad {color:#ef4444}
  .coords {font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace}
  .folium-map { margin-bottom: 0 !important; }
  div[data-testid="stVerticalBlock"] > div:has(.folium-map) { margin-bottom: 0 !important; }
  .tiny {font-size: .85rem; opacity: .7}
</style>
""", unsafe_allow_html=True)

# ---- labels/defaults ----
MATERIAL_NAMES = ["asphalt", "concrete", "paving_stones", "unpaved", "sett"]
QUALITY_NAMES  = ["excellent", "good", "intermediate", "bad", "very_bad"]
TOKYO_DEFAULT  = (35.681236, 139.767125)

# === MODEL ===
class MultiHeadEffB7(nn.Module):
    def __init__(self, n_type=len(MATERIAL_NAMES), n_qual=len(QUALITY_NAMES)):
        super().__init__()
        base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        in_f = base.classifier[1].in_features
        self.mat  = nn.Linear(in_f, n_type)
        self.qual = nn.Linear(in_f, n_qual)
    def forward(self, x):
        z = self.features(x)
        return self.mat(z), self.qual(z)

@st.cache_resource(show_spinner=True)
def get_model_session():
    if not HF_REPO_ID or not HF_FILENAME:
        st.error("HF_REPO_ID / HF_FILENAME not set in env."); st.stop()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadEffB7().to(device)

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=HF_TOKEN)
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            state = torch.load(io.BytesIO(f.read()), map_location=device)
    else:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("_orig_mod.","").replace("module.",""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    @torch.inference_mode()
    def session(pil_img):
        x = tfm(pil_img).unsqueeze(0).to(device)
        t_logits, q_logits = model(x)
        t_idx = int(torch.softmax(t_logits, 1).argmax())
        q_idx = int(torch.softmax(q_logits, 1).argmax())
        return {"surface_type": MATERIAL_NAMES[t_idx], "surface_quality": QUALITY_NAMES[q_idx]}
    return session

# ---- GPS extraction (EXIF first, then XMP fallback) ----
_GPS_TAGS = 34853  # GPSInfo

def _num(x):
    try:
        return float(getattr(x, "numerator", x)) / float(getattr(x, "denominator", 1))
    except Exception:
        try:
            a, b = x
            return float(a) / float(b)
        except Exception:
            return float(x)

def _to_deg_any(v):
    try:
        if isinstance(v, (list, tuple)) and len(v) == 3:
            d, m, s = v
            return _num(d) + _num(m)/60.0 + _num(s)/3600.0
        return _num(v)
    except Exception:
        return None

def _exif_latlon_pillow(img: Image.Image):
    """Try both Pillow EXIF paths: get_ifd(IFD.GPS) and _getexif()[GPSInfo]."""
    # 1) Newer Pillow path
    try:
        ex = img.getexif()
        IFD = getattr(ExifTags, "IFD", None)
        if ex and IFD is not None:
            try:
                gps_ifd = ex.get_ifd(IFD.GPS)
            except Exception:
                gps_ifd = None
            if gps_ifd:
                lat_raw = gps_ifd.get(2) or gps_ifd.get("GPSLatitude")
                lon_raw = gps_ifd.get(4) or gps_ifd.get("GPSLongitude")
                lat_ref = gps_ifd.get(1) or gps_ifd.get("GPSLatitudeRef") or "N"
                lon_ref = gps_ifd.get(3) or gps_ifd.get("GPSLongitudeRef") or "E"
                lat = _to_deg_any(lat_raw); lon = _to_deg_any(lon_raw)
                if lat is not None and lon is not None:
                    if (lat_ref == "S") or (lat_ref == b"S"): lat = -lat
                    if (lon_ref == "W") or (lon_ref == b"W"): lon = -lon
                    return (float(lat), float(lon))
    except Exception:
        pass
    # 2) Legacy Pillow path
    try:
        ex2 = getattr(img, "_getexif", lambda: None)()
        if ex2 and _GPS_TAGS in ex2:
            gps = ex2[_GPS_TAGS]
            gps_map = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
            lat = _to_deg_any(gps_map.get("GPSLatitude"))
            lon = _to_deg_any(gps_map.get("GPSLongitude"))
            if lat is not None and lon is not None:
                lat_ref = gps_map.get("GPSLatitudeRef", "N")
                lon_ref = gps_map.get("GPSLongitudeRef", "E")
                if (lat_ref == "S") or (lat_ref == b"S"): lat = -lat
                if (lon_ref == "W") or (lon_ref == b"W"): lon = -lon
                return (float(lat), float(lon))
    except Exception:
        pass
    return None

def _xmp_extract_coords(jpeg_bytes: bytes):
    """Fallback: parse simple XMP packets that store exif:GPSLatitude/Longitude."""
    try:
        start = jpeg_bytes.find(b"<x:xmpmeta")
        if start == -1:
            return None
        end = jpeg_bytes.find(b"</x:xmpmeta>", start)
        if end == -1:
            return None
        xmp = jpeg_bytes[start:end+12].decode("utf-8", errors="ignore")

        def pick(patterns):
            for p in patterns:
                m = re.search(p, xmp)
                if m: return m.group(1).strip()
            return None

        lat_s = pick([r'GPSLatitude="([^"]+)"', r'exif:GPSLatitude>([^<]+)<'])
        lon_s = pick([r'GPSLongitude="([^"]+)"', r'exif:GPSLongitude>([^<]+)<'])
        lat_ref = pick([r'GPSLatitudeRef="([^"]+)"', r'exif:GPSLatitudeRef>([^<]+)<'])
        lon_ref = pick([r'GPSLongitudeRef="([^"]+)"', r'exif:GPSLongitudeRef>([^<]+)<'])
        if not (lat_s and lon_s): return None

        def to_deg(val: str):
            nums = re.findall(r'[-+]?\d+(?:\.\d+)?', val)
            if not nums: return None
            if len(nums) >= 3:
                d, m, s = map(float, nums[:3])
                return d + m/60 + s/3600
            return float(nums[0])

        lat = to_deg(lat_s); lon = to_deg(lon_s)
        if lat is None or lon is None: return None
        if (lat_ref or "").upper().startswith("S"): lat = -abs(lat)
        if (lon_ref or "").upper().startswith("W"): lon = -abs(lon)
        return (float(lat), float(lon))
    except Exception:
        return None

def extract_gps_from_bytes(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        hit = _exif_latlon_pillow(img)
        if hit: return hit
    except Exception:
        pass
    hit = _xmp_extract_coords(data)
    if hit: return hit
    return None

def gmaps_link(lat, lon): return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# ---- Overpass (nearest ward office) ----
def _haversine(a,b,c,d):
    R=6371.0; to=math.pi/180.0
    dlat,dlon=(c-a)*to,(d-b)*to
    x=math.sin(dlat/2)**2+math.cos(a*to)*math.cos(c*to)*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(x), math.sqrt(1-x))

@st.cache_data(ttl=300, show_spinner=False)
def find_nearest_ward_office(lat: float, lon: float, salt: int = 0):
    NAME_REGEX = r"Ward Office|City Office|City Hall|Town Hall|区役所|市役所|町役場|村役場|役場|出張所"
    UA = {"User-Agent":"roadsafe-demo/1.0 (+contact@yourdomain.tld)","Accept":"application/json","Accept-Encoding":"gzip"}
    MIRRORS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
    ]

    def _mk(r, strict):
        if strict:
            return f"""[out:json][timeout:25];
            (node["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
             way ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
             rel ["amenity"="townhall"]["name"~"{NAME_REGEX}"](around:{r},{lat},{lon}););
            out center tags;"""
        return f"""[out:json][timeout:25];
        (node["amenity"="townhall"](around:{r},{lat},{lon});
         way ["amenity"="townhall"](around:{r},{lat},{lon});
         rel ["amenity"="townhall"](around:{r},{lat},{lon});
         node["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
         way ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
         rel ["office"="government"]["government"="administrative"](around:{r},{lat},{lon});
         node["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
         way ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
         rel ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon}););
        out center tags;"""

    def _center(el):
        if el.get("type") == "node":
            return (el.get("lat"), el.get("lon"))
        c = el.get("center") or {}
        return (c.get("lat"), c.get("lon"))

    def _addr(tags):
        parts = [tags.get("addr:postcode"), tags.get("addr:state") or tags.get("addr:province"),
                 tags.get("addr:city"), tags.get("addr:district") or tags.get("addr:ward") or tags.get("addr:suburb"),
                 tags.get("addr:street"), tags.get("addr:block_number"), tags.get("addr:housenumber")]
        s = " ".join([p for p in parts if p]).strip()
        return s or None

    def _score(tags, dkm):
        nm = tags.get("name:en") or tags.get("official_name:en") or tags.get("name") or tags.get("official_name") or tags.get("name:ja")
        has_name = 1.0 if nm else 0.0
        amenity = 1.0 if tags.get("amenity") == "townhall" else 0.0
        admin   = 1.0 if (tags.get("office")=="government" and tags.get("government")=="administrative") else 0.0
        hit = 1.0 if nm and (any(x in nm for x in ["Ward Office","City Office","City Hall","Town Hall"]) or any(x in nm for x in ["区役所","市役所","町役場","村役場","役場","出張所"])) else 0.0
        addr = 1.0 if _addr(tags) else 0.0
        email= 1.0 if (tags.get("contact:email") or tags.get("email")) else 0.0
        base = 3*amenity + 2.5*hit + 1.5*admin + 1.0*has_name + 0.5*addr + 0.25*email
        return base - dkm

    def _best(elems):
        best=None
        for el in elems:
            latc, lonc = _center(el)
            if latc is None or lonc is None: continue
            dkm = _haversine(lat, lon, latc, lonc)
            tags = el.get("tags", {}) or {}
            sc   = _score(tags, dkm)
            item = {"name": tags.get("name:en") or tags.get("official_name:en") or tags.get("name") or tags.get("official_name") or tags.get("name:ja") or "(Unnamed Government Office)",
                    "lat": float(latc), "lon": float(lonc), "distance_km": float(dkm),
                    "address": _addr(tags), "email": tags.get("contact:email") or tags.get("email"), "score": sc}
            if best is None or item["score"] > best["score"]:
                best = item
        return best

    for r in (1500, 3000, 6000, 12000, 20000, 30000):
        for strict in (True, False):
            q = _mk(r, strict)
            for base in MIRRORS:
                try:
                    js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
                except Exception:
                    continue
                best = _best(js.get("elements", []))
                if best: return best

    # fallback using geocodeArea
    try:
        nom = requests.get("https://nominatim.openstreetmap.org/reverse",
                           params={"format":"jsonv2","lat":lat,"lon":lon,"zoom":14,"addressdetails":1},
                           headers=UA, timeout=10).json()
        a = nom.get("address", {})
        area = ", ".join([p for p in [a.get("city_district") or a.get("ward"),
                                      a.get("city") or a.get("town") or a.get("county"),
                                      a.get("state") or a.get("region"),
                                      "Japan"] if p])
        if area:
            q = f"""[out:json][timeout:25];
            {{geocodeArea:{area}}}->.a;
            (node(area.a)["amenity"="townhall"]; way(area.a)["amenity"="townhall"]; rel(area.a)["amenity"="townhall"];
             node(area.a)["office"="government"]["government"="administrative"];
             way(area.a)["office"="government"]["government"="administrative"];
             rel(area.a)["office"="government"]["government"="administrative"];
             node(area.a)["name"~"{NAME_REGEX}"]; way(area.a)["name"~"{NAME_REGEX}"]; rel(area.a)["name"~"{NAME_REGEX}"];);
            out center tags;"""
            for base in MIRRORS:
                try:
                    js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
                except Exception:
                    continue
                best = _best(js.get("elements", []))
                if best: return best
    except Exception:
        pass
    return None

# ---- email helpers ----
def mask_email(e: str) -> str:
    if not e or "@" not in e: return "hidden@wardoffice.com"
    local, domain = e.split("@", 1)
    masked_local = (local[:2] + "***") if len(local) > 2 else (local[:1] + "***")
    return f"{masked_local}@{domain}"

def send_email_html(to_email, subject, html_body, text_fallback, image_bytes=None, image_name="image.jpg"):
    if not GMAIL_USER or not GMAIL_PASSWORD:
        return False, "Email credentials not found (set GMAIL_USER / GMAIL_PASSWORD)."
    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(text_fallback, charset="utf-8")
    msg.add_alternative(html_body, subtype="html")
    if image_bytes:
        subtype = (image_name.split(".")[-1] or "jpeg").lower()
        msg.get_payload()[1].add_related(image_bytes, maintype="image", subtype=subtype, cid="<photo1>")
        msg.add_attachment(image_bytes, maintype="image", subtype=subtype, filename=image_name)
    import smtplib
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, f"{e}"

# ---- romaji fallback ----
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x
def romaji_email_from_name(name: str) -> str:
    base = unidecode(name or "").lower()
    base = re.sub(r"[^a-z0-9]+", "", base) or "wardoffice"
    return f"{base}@wardoffice.com"

# ---- hard reset helper ----
def hard_reset_all():
    """Hard reset: wipe session state, clear caches, bust URL so widgets fully reset, then rerun."""
    st.session_state.clear()
    try:
        st.cache_data.clear()
    except Exception:
        pass
    ts = str(int(time.time()))
    try:
        st.query_params["_"] = ts
    except Exception:
        try:
            st.experimental_set_query_params(**{"_": ts})
        except Exception:
            pass
    st.rerun()

# ---- state ----
defaults = dict(
    pred=None, img_bytes=None, img_name=None,
    live_point=None, confirmed_point=None,
    map_center=TOKYO_DEFAULT, map_zoom=12,
    location_confirmed=False, ward_office=None, report_sent=False,
    allow_submit_anyway=False,
    ward_nonce=0,
    metadata_pin=False,
)
for k,v in defaults.items(): st.session_state.setdefault(k,v)

# Helper: derive a stable uploader key from the URL "_"
def uploader_key():
    qs = None
    try:
        qs = st.query_params.get("_", None)
    except Exception:
        pass
    return f"uploader-{qs or '0'}"

# ---- header ----
st.markdown("""
<div class='center'>
  <h1 style="margin-bottom:0.2rem;">RoadSafe — Report Road Damage</h1>
  <p class='muted' style='margin-top:0'>Upload → Validate → Pinpoint → Notify Ward Office</p>
</div>
""", unsafe_allow_html=True)

# ---- upload → predict (GPS from EXIF/XMP) ----
uploader = st.file_uploader(
    "Upload a road photo (JPG/PNG)",
    type=["jpg","jpeg","png"],
    key=uploader_key(),  # <- key changes after hard_reset_all(), clearing any lingering filename
)
if uploader is not None:
    raw_bytes = uploader.getvalue()
    img_raw = Image.open(io.BytesIO(raw_bytes))
    gps = extract_gps_from_bytes(raw_bytes)

    img = img_raw.convert("RGB")
    st.session_state.img_bytes = raw_bytes
    st.session_state.img_name = uploader.name
    st.image(img, caption="Input image", use_container_width=True)

    # Auto-pin & center if GPS found (user still confirms)
    if gps and st.session_state.live_point is None:
        pt = (float(gps[0]), float(gps[1]))
        st.session_state.live_point = pt
        st.session_state.map_center = pt
        st.session_state.map_zoom = 16
        st.session_state.metadata_pin = True

    with st.spinner("Analyzing the image…"):
        session = get_model_session()
        st.session_state.pred = session(img)
    st.session_state.report_sent = False

# ---- results ----
if st.session_state.pred:
    res = st.session_state.pred
    type_txt = res["surface_type"].replace("_"," ").title()
    qual_raw = res["surface_quality"]
    qual_txt = qual_raw.replace("_"," ").title()
    qcls = "qual-ok" if qual_raw in ("excellent","good") else ("qual-mid" if qual_raw=="intermediate" else "qual-bad")

    st.markdown("<div class='section-title'>Predicted Surface Type</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-value'>{type_txt}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Predicted Surface Quality</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-value {qcls}'>{qual_txt}</div>", unsafe_allow_html=True)

    if qual_raw in {"excellent", "good", "intermediate"} and not st.session_state.allow_submit_anyway:
        st.info("This road seems to be in a decent state. Are you sure you want to report it?")
        st.session_state.allow_submit_anyway = st.toggle("Report anyway", value=st.session_state.allow_submit_anyway)
        if not st.session_state.allow_submit_anyway:
            st.stop()

    st.divider()
    st.subheader("Confirm location of the damage")

    if st.session_state.live_point is None:
        st.warning("Location not found from your picture — please click on the map to drop a pin.")
    elif st.session_state.metadata_pin:
        st.info("Location read from your photo (GPS metadata). Please confirm, or click on the map to adjust.")
        st.session_state.metadata_pin = False

    @st.fragment
    def map_and_followups():
        # MAP (no center/zoom persistence ⇒ no reruns while panning/zooming)
        m = folium.Map(
            location=st.session_state.map_center,
            zoom_start=st.session_state.map_zoom,
            control_scale=True,
            tiles="OpenStreetMap"
        )
        if st.session_state.live_point:
            folium.Marker(st.session_state.live_point, tooltip="Selected location").add_to(m)

        out = st_folium(m, width=720, height=420, key="map-main", returned_objects=["last_clicked"])
        if out and out.get("last_clicked"):
            click = out["last_clicked"]
            if click.get("lat") and click.get("lng"):
                st.session_state.live_point = (float(click["lat"]), float(click["lng"]))
                st.session_state.map_center = st.session_state.live_point
                st.session_state.location_confirmed = False
                st.session_state.confirmed_point = None
                st.session_state.ward_office = None

        if st.session_state.live_point:
            lat, lon = st.session_state.live_point
            st.caption(f"Chosen coordinates: <span class='coords'>{lat:.6f}, {lon:.6f}</span>", unsafe_allow_html=True)

        st.button(
            "Confirm location",
            key="confirm-location-btn",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.live_point is None,
            on_click=lambda: st.session_state.update(
                location_confirmed=True,
                confirmed_point=st.session_state.live_point
            )
        )

        if st.session_state.location_confirmed and st.session_state.confirmed_point and not st.session_state.report_sent:
            st.divider()
            st.subheader("Closest ward/city office  ↪")

            if st.session_state.ward_office is None:
                with st.spinner("Looking up nearest office…"):
                    latc, lonc = st.session_state.confirmed_point
                    office = find_nearest_ward_office(latc, lonc, st.session_state.get("ward_nonce", 0))
                    st.session_state.ward_office = office

            office = st.session_state.ward_office
            if office:
                st.success(f"Closest office: **{office['name']}**")
                st.caption(f"📍 {office['lat']:.6f}, {office['lon']:.6f}")
                st.link_button("Open office in Google Maps", gmaps_link(office['lat'], office['lon']), use_container_width=True)
            else:
                st.warning("Couldn’t identify a nearby office automatically. You can still send the report (coordinates included).")
                st.button("Retry lookup", key="retry-ward", on_click=lambda: st.session_state.update(ward_nonce=st.session_state.get("ward_nonce",0) + 1, ward_office=None))

            # SEND REPORT
            st.divider()
            st.subheader("Send the report")

            if office and office.get("email"):
                display_recipient = office["email"]
            else:
                display_recipient = romaji_email_from_name(office["name"] if office else "Ward Office")
            st.caption("Email of the ward office — hidden to avoid spam.")
            st.text_input("Recipient", value=mask_email(display_recipient), disabled=True)

            actual_recipient = DEMO_RECEIVER or GMAIL_USER or display_recipient
            type_txt = res["surface_type"].replace("_"," ").title()
            qual_txt = res["surface_quality"].replace("_"," ").title()
            latc, lonc = st.session_state.confirmed_point
            when = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if office:
                office_addr_or_link = office.get('address') or gmaps_link(office['lat'], office['lon'])
                office_name = office.get('name', 'N/A')
            else:
                office_addr_or_link = gmaps_link(latc, lonc)
                office_name = 'N/A'

            core_text = f"""RoadSafe report (demo)
When: {when}
Surface type: {type_txt}
Surface quality: {qual_txt}
Location: {latc:.6f}, {lonc:.6f}
Google Maps: {gmaps_link(latc, lonc)}
Ward office: {office_name}
Address: {office_addr_or_link}
"""
            st.text_area("Preview (read-only)", value=core_text, height=160, disabled=True)
            comment = st.text_area("Additional comments (optional)", placeholder="Add any useful details…")

            html = f"""
            <div style="font-family:system-ui;">
              <h2>RoadSafe report (demo)</h2>
              <p style="color:#555">When: {when}</p>
              <img src="cid:photo1" style="max-width:640px;border-radius:8px;margin:8px 0 16px"/>
              <ul style="line-height:1.6">
                <li><b>Surface type:</b> {type_txt}</li>
                <li><b>Surface quality:</b> {qual_txt}</li>
                <li><b>Location:</b> {latc:.6f}, {lonc:.6f}</li>
                <li><b>Google Maps:</b> <a href="{gmaps_link(latc,lonc)}">{gmaps_link(latc,lonc)}</a></li>
                <li><b>Ward office:</b> {office_name}</li>
                <li><b>Address:</b> {office_addr_or_link}</li>
              </ul>
              {"<p><b>Additional comments:</b><br>"+comment.replace('\\n','<br>')+"</p>" if comment.strip() else ""}
              <p class="tiny">This message was generated for demo purposes.</p>
            </div>
            """.strip()

            text = core_text + (f"\nAdditional comments:\n{comment}\n" if comment.strip() else "")

            if st.button("Send report", type="primary", use_container_width=True, key="send-report-btn"):
                ok, info = send_email_html(
                    to_email=actual_recipient,
                    subject="RoadSafe Report (Demo)",
                    html_body=html,
                    text_fallback=text,
                    image_bytes=st.session_state.img_bytes,
                    image_name=st.session_state.img_name or "photo.jpg",
                )
                if ok:
                    st.success("Report submitted successfully, thank you for your contribution.")
                    time.sleep(2)
                    st.toast("The current page will now refresh — feel free to submit another damage!", icon="✅")
                    time.sleep(5)
                    hard_reset_all()
                else:
                    st.error(f"Failed to send email: {info}")

    map_and_followups()

# ---- footer ----
st.divider()
st.caption("ROADSAFE")
