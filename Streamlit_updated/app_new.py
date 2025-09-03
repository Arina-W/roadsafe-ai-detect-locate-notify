# # # RoadSafe — Streamlit app (KL default, robust Overpass, EXIF prefill, clean reset)
# # # -------------------------------------------------------------------------------

# # import io, os, re, math, gzip, time, random
# # from datetime import datetime
# # from email.message import EmailMessage

# # import streamlit as st
# # from PIL import Image, ExifTags
# # import pandas as pd
# # import requests
# # import folium
# # from streamlit_folium import st_folium

# # # ML
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torchvision import transforms
# # from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
# # from huggingface_hub import hf_hub_download

# # # ---- env ----
# # from dotenv import load_dotenv
# # load_dotenv()
# # GMAIL_USER      = os.getenv("GMAIL_USER")
# # GMAIL_PASSWORD  = os.getenv("GMAIL_PASSWORD")
# # HF_REPO_ID      = os.getenv("HF_REPO_ID") or "esdk/my-efficientnet-model"
# # HF_FILENAME     = os.getenv("HF_FILENAME") or "efficientnet_fp16.pt.gz"
# # HF_TOKEN        = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
# # DEMO_RECEIVER   = os.getenv("DEMO_RECEIVER")

# # # ---- page & css ----
# # st.set_page_config(page_title="RoadSafe — Report a Road Issue", page_icon="🛣️", layout="centered")
# # st.markdown("""
# # <style>
# #   .center {text-align:center}
# #   .muted {opacity:.75}
# #   .section-title {font-size:1.05rem; font-weight:700; margin: 10px 0 4px}
# #   .result-value {font-size:2.2rem; font-weight:800; margin: 0 0 6px}
# #   .qual-ok  {color:#22c55e}
# #   .qual-mid {color:#a16207}
# #   .qual-bad {color:#ef4444}
# #   .coords {font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace}
# #   .folium-map { margin-bottom: 0 !important; }
# #   div[data-testid="stVerticalBlock"] > div:has(.folium-map) { margin-bottom: 0 !important; }
# #   .tiny {font-size: .85rem; opacity: .7}
# # </style>
# # """, unsafe_allow_html=True)

# # # ---- labels/defaults ----
# # MATERIAL_NAMES = ["asphalt", "concrete", "paving_stones", "unpaved", "sett"]
# # QUALITY_NAMES  = ["excellent", "good", "intermediate", "bad", "very_bad"]

# # # Kuala Lumpur city center
# # KL_DEFAULT = (3.139003, 101.686855)

# # # === MODEL ===
# # class MultiHeadEffB7(nn.Module):
# #     def __init__(self, n_type=len(MATERIAL_NAMES), n_qual=len(QUALITY_NAMES)):
# #         super().__init__()
# #         base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
# #         self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
# #         in_f = base.classifier[1].in_features
# #         self.mat  = nn.Linear(in_f, n_type)
# #         self.qual = nn.Linear(in_f, n_qual)
# #     def forward(self, x):
# #         z = self.features(x)
# #         return self.mat(z), self.qual(z)

# # @st.cache_resource(show_spinner=True)
# # def get_model_session():
# #     if not HF_REPO_ID or not HF_FILENAME:
# #         st.error("HF_REPO_ID / HF_FILENAME not set in env."); st.stop()
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model = MultiHeadEffB7().to(device)

# #     path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=HF_TOKEN)
# #     if path.endswith(".gz"):
# #         with gzip.open(path, "rb") as f:
# #             state = torch.load(io.BytesIO(f.read()), map_location=device)
# #     else:
# #         state = torch.load(path, map_location=device)

# #     if isinstance(state, dict) and "state_dict" in state:
# #         state = state["state_dict"]
# #     state = {k.replace("_orig_mod.","").replace("module.",""): v for k, v in state.items()}
# #     model.load_state_dict(state, strict=False)
# #     model.eval()

# #     tfm = transforms.Compose([
# #         transforms.Resize((600, 600)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
# #     ])

# #     @torch.inference_mode()
# #     def session(pil_img: Image.Image):
# #         x = tfm(pil_img).unsqueeze(0).to(device)
# #         t_logits, q_logits = model(x)
# #         t_probs = F.softmax(t_logits, dim=1).squeeze(0).cpu().tolist()
# #         q_probs = F.softmax(q_logits, dim=1).squeeze(0).cpu().tolist()
# #         t_idx = int(torch.tensor(t_probs).argmax())
# #         q_idx = int(torch.tensor(q_probs).argmax())
# #         return {
# #             "surface_type": MATERIAL_NAMES[t_idx],
# #             "surface_quality": QUALITY_NAMES[q_idx],
# #             "surface_type_probs": t_probs,
# #             "surface_quality_probs": q_probs,
# #         }
# #     return session

# # # ---- EXIF ----
# # _GPS_TAGS = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
# # def _to_deg(v): d,m,s=v; return d[0]/d[1] + (m[0]/m[1])/60 + (s[0]/s[1])/3600
# # def exif_latlon(img: Image.Image):
# #     try:
# #         exif = img._getexif()
# #         if not exif or _GPS_TAGS not in exif: return None
# #         gps = {ExifTags.GPSTAGS.get(k,k): v for k,v in exif[_GPS_TAGS].items()}
# #         lat = _to_deg(gps.get("GPSLatitude"))  if "GPSLatitude"  in gps else None
# #         lon = _to_deg(gps.get("GPSLongitude")) if "GPSLongitude" in gps else None
# #         if lat is None or lon is None: return None
# #         if gps.get("GPSLatitudeRef","N") == "S": lat = -lat
# #         if gps.get("GPSLongitudeRef","E") == "W": lon = -lon
# #         return (lat, lon)
# #     except Exception:
# #         return None

# # def gmaps_link(lat, lon): return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# # # ---- distance helper ----
# # def _haversine(a,b,c,d):
# #     R=6371.0; to=math.pi/180.0
# #     dlat,dlon=(c-a)*to,(d-b)*to
# #     x=math.sin(dlat/2)**2+math.cos(a*to)*math.cos(c*to)*math.sin(dlon/2)**2
# #     return 2*R*math.atan2(math.sqrt(x), math.sqrt(1-x))

# # # ---- robust worldwide (Malay-aware) Overpass + Nominatim fallback ----
# # @st.cache_data(ttl=300, show_spinner=False)
# # def find_nearest_ward_office(lat: float, lon: float, salt: int = 0):
# #     lat = float(lat); lon = float(lon)
# #     NAME_REGEX = r"(?i)(Ward Office|City Office|City Hall|Town Hall|City Council|Municipal Council|Majlis Perbandaran|Majlis Bandaraya|Majlis Daerah|Dewan Bandaraya|Pejabat Daerah|Pejabat Tanah|Pejabat Kerajaan|Pejabat Majlis|区役所|市役所|町役場|村役場|役場|出張所)"
# #     UA = {"User-Agent":"roadsafe-demo/1.2 (+contact@example.com)","Accept":"application/json","Accept-Encoding":"gzip"}
# #     MIRRORS = [
# #         "https://overpass-api.de/api/interpreter",
# #         "https://overpass.kumi.systems/api/interpreter",
# #         "https://overpass.openstreetmap.fr/api/interpreter",
# #     ]
# #     EN_TOK = [t.lower() for t in ["Ward Office","City Office","City Hall","Town Hall","City Council","Municipal Council"]]
# #     MS_TOK = [t.lower() for t in ["Majlis Perbandaran","Majlis Bandaraya","Majlis Daerah","Dewan Bandaraya","Pejabat Daerah","Pejabat Tanah","Pejabat Kerajaan","Pejabat Majlis","Majlis"]]
# #     JP_TOK = [t.lower() for t in ["区役所","市役所","町役場","村役場","役場","出張所"]]

# #     def _mk_query(r):
# #         return f"""[out:json][timeout:25];
# #         (
# #           node["amenity"="townhall"](around:{r},{lat},{lon});
# #           way ["amenity"="townhall"](around:{r},{lat},{lon});
# #           rel ["amenity"="townhall"](around:{r},{lat},{lon});

# #           node["office"="government"](around:{r},{lat},{lon});
# #           way ["office"="government"](around:{r},{lat},{lon});
# #           rel ["office"="government"](around:{r},{lat},{lon});

# #           node["office"="administrative"](around:{r},{lat},{lon});
# #           way ["office"="administrative"](around:{r},{lat},{lon});
# #           rel ["office"="administrative"](around:{r},{lat},{lon});

# #           node["government"](around:{r},{lat},{lon});
# #           way ["government"](around:{r},{lat},{lon});
# #           rel ["government"](around:{r},{lat},{lon});

# #           node["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
# #           way ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
# #           rel ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
# #         );
# #         out center tags;"""

# #     def _center(el):
# #         if el.get("type") == "node": return el.get("lat"), el.get("lon")
# #         c = el.get("center") or {}; return c.get("lat"), c.get("lon")

# #     def _addr(tags):
# #         parts = [
# #             tags.get("addr:postcode"),
# #             tags.get("addr:state") or tags.get("addr:province") or tags.get("addr:region"),
# #             tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:county"),
# #             tags.get("addr:district") or tags.get("addr:ward") or tags.get("addr:suburb"),
# #             tags.get("addr:street"), tags.get("addr:block_number"), tags.get("addr:housenumber"),
# #         ]
# #         s = " ".join([p for p in parts if p]).strip()
# #         return s or None

# #     def _score(tags, dkm):
# #         nm = (tags.get("name:en") or tags.get("official_name:en") or
# #               tags.get("name:ms") or tags.get("official_name:ms") or
# #               tags.get("name") or tags.get("official_name") or
# #               tags.get("name:ja") or "")
# #         nml = nm.lower()
# #         has_name = 1.0 if nm else 0.0
# #         amenity = 1.2 if tags.get("amenity") == "townhall" else 0.0
# #         admin   = 0.8 if (tags.get("office") in {"government","administrative"} or "government" in tags) else 0.0
# #         hit_en  = 1.2 if any(t in nml for t in EN_TOK) else 0.0
# #         hit_ms  = 1.5 if any(t in nml for t in MS_TOK) else 0.0
# #         hit_jp  = 1.0 if any(t in nml for t in JP_TOK) else 0.0
# #         addr    = 0.5 if _addr(tags) else 0.0
# #         email   = 0.25 if (tags.get("contact:email") or tags.get("email")) else 0.0
# #         return (amenity + admin + has_name + hit_en + hit_ms + hit_jp + addr + email) - dkm

# #     def _best(elements):
# #         best=None
# #         for el in elements:
# #             latc, lonc = _center(el)
# #             if latc is None or lonc is None: continue
# #             dkm = _haversine(lat, lon, float(latc), float(lonc))
# #             tags = el.get("tags", {}) or {}
# #             score = _score(tags, dkm)
# #             name = (tags.get("name:en") or tags.get("official_name:en") or
# #                     tags.get("name:ms") or tags.get("official_name:ms") or
# #                     tags.get("name") or tags.get("official_name") or
# #                     tags.get("name:ja") or "(Unnamed Government Office)")
# #             item = {
# #                 "name": name, "lat": float(latc), "lon": float(lonc),
# #                 "distance_km": float(dkm), "address": _addr(tags),
# #                 "email": tags.get("contact:email") or tags.get("email"), "score": score
# #             }
# #             if best is None or item["score"] > best["score"]: best=item
# #         return best

# #     # Overpass escalating radius
# #     for r in (1500, 3000, 6000, 12000, 20000, 35000):
# #         q = _mk_query(r)
# #         for base in MIRRORS:
# #             try:
# #                 js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
# #             except Exception:
# #                 continue
# #             best = _best(js.get("elements", []))
# #             if best: return best

# #     # Nominatim fallback within local bbox
# #     try:
# #         d = 0.25
# #         viewbox = f"{lon-d},{lat+d},{lon+d},{lat-d}"
# #         queries = [
# #             "majlis bandaraya","majlis perbandaran","majlis daerah",
# #             "dewan bandaraya","city hall","municipal council","city council","town hall",
# #             "pejabat daerah","pejabat kerajaan",
# #         ]
# #         best=None
# #         for q in queries:
# #             try:
# #                 resp = requests.get(
# #                     "https://nominatim.openstreetmap.org/search",
# #                     params={"q": q,"format": "jsonv2","limit": 10,"addressdetails": 1,"viewbox": viewbox,"bounded": 1},
# #                     headers=UA, timeout=10
# #                 )
# #                 hits = resp.json()
# #             except Exception:
# #                 continue
# #             for h in hits:
# #                 latc, lonc = float(h["lat"]), float(h["lon"])
# #                 dkm = _haversine(lat, lon, latc, lonc)
# #                 name = h.get("name") or (h.get("display_name","").split(",")[0]) or "Nearest Government Office"
# #                 a = h.get("address", {})
# #                 addr = None
# #                 if a:
# #                     addr_parts = [a.get("postcode"), a.get("state") or a.get("region"),
# #                                   a.get("city") or a.get("town") or a.get("county"),
# #                                   a.get("suburb") or a.get("neighbourhood"),
# #                                   a.get("road"), a.get("house_number")]
# #                     addr = " ".join([p for p in addr_parts if p]).strip() or None
# #                 item = {"name": name,"lat": latc,"lon": lonc,"distance_km": dkm,"address": addr,"email": None,"score": 0.0}
# #                 if (best is None) or (dkm < best["distance_km"]): best=item
# #         if best: return best
# #     except Exception:
# #         pass

# #     return None

# # # ---- optional CSV fallback (e.g., city_malaysia.csv with columns city,email) ----
# # @st.cache_data(show_spinner=False)
# # def load_city_csv():
# #     candidates = [
# #         "city_malaysia.csv","ward_offices.csv","city.csv","csv.csv",
# #         "/mnt/data/city_malaysia.csv"
# #     ]
# #     for path in candidates:
# #         if os.path.exists(path):
# #             try:
# #                 df = pd.read_csv(path)
# #                 if {"city","email"}.issubset(df.columns):
# #                     return df
# #             except Exception:
# #                 pass
# #     return None

# # def city_email_fallback(lat, lon):
# #     try:
# #         j=requests.get("https://nominatim.openstreetmap.org/reverse",
# #                        params={"format":"jsonv2","lat":lat,"lon":lon,"zoom":14,"addressdetails":1},
# #                        headers={"User-Agent":"roadsafe-demo"}, timeout=12).json()
# #         city=j.get("address",{}).get("city") or j.get("address",{}).get("town") or j.get("address",{}).get("county")
# #         df=load_city_csv()
# #         if df is not None and city:
# #             hit=df[df["city"].str.lower()==city.lower()]
# #             if not hit.empty:
# #                 return {"name": f"{city} Municipal Office", "email": hit.iloc[0]["email"], "lat":lat,"lon":lon,"address":None}
# #     except Exception:
# #         pass
# #     return None

# # # ---- email helpers ----
# # def mask_email(e: str) -> str:
# #     if not e or "@" not in e: return "hidden@wardoffice.com"
# #     local, domain = e.split("@", 1)
# #     masked_local = (local[:2] + "***") if len(local) > 2 else (local[:1] + "***")
# #     return f"{masked_local}@{domain}"

# # def send_email_html(to_email, subject, html_body, text_fallback, image_bytes=None, image_name="image.jpg"):
# #     if not GMAIL_USER or not GMAIL_PASSWORD:
# #         return False, "Email credentials not found (set GMAIL_USER / GMAIL_PASSWORD)."
# #     msg = EmailMessage()
# #     msg["From"] = GMAIL_USER
# #     msg["To"] = to_email
# #     msg["Subject"] = subject
# #     msg.set_content(text_fallback, charset="utf-8")
# #     msg.add_alternative(html_body, subtype="html")
# #     if image_bytes:
# #         subtype = (image_name.split(".")[-1] or "jpeg").lower()
# #         msg.get_payload()[1].add_related(image_bytes, maintype="image", subtype=subtype, cid="<photo1>")
# #         msg.add_attachment(image_bytes, maintype="image", subtype=subtype, filename=image_name)

# #     import smtplib
# #     try:
# #         with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
# #             s.login(GMAIL_USER, GMAIL_PASSWORD)
# #             s.send_message(msg)
# #         return True, "Email sent"
# #     except Exception as e:
# #         return False, f"{e}"

# # # ---- romaji fallback (for display masking) ----
# # try:
# #     from unidecode import unidecode
# # except Exception:
# #     def unidecode(x): return x
# # def romaji_email_from_name(name: str) -> str:
# #     base = unidecode(name or "").lower()
# #     base = re.sub(r"[^a-z0-9]+", "", base) or "wardoffice"
# #     return f"{base}@wardoffice.com"

# # # ---- STATE ----
# # defaults = dict(
# #     pred=None, img_bytes=None, img_name=None,
# #     live_point=None, confirmed_point=None,
# #     map_center=KL_DEFAULT, map_zoom=12,
# #     location_confirmed=False, ward_office=None, report_sent=False,
# #     allow_submit_anyway=False, exif_prefilled=False,
# # )
# # for k,v in defaults.items(): st.session_state.setdefault(k,v)
# # st.session_state.setdefault("uploader_key", f"uploader-{int(time.time()*1000)}")

# # def _hard_reset():
# #     # fully reset state and rotate uploader key so filename disappears
# #     for k in list(defaults.keys()):
# #         st.session_state[k] = defaults[k]
# #     st.session_state["uploader_key"] = f"uploader-{int(time.time()*1000)}"
# #     # tiny delay so user sees the toast before the rerun
# #     time.sleep(0.05)
# #     st.rerun()

# # # ---- header ----
# # st.markdown("""
# # <div class='center'>
# #   <h1 style="margin-bottom:0.2rem;">🛣️ RoadSafe — Report Road Damage</h1>
# #   <p class='muted' style='margin-top:0'>Upload → Validate → Pinpoint → Notify Ward Office</p>
# # </div>
# # """, unsafe_allow_html=True)

# # # ---- upload → predict ----
# # uploader = st.file_uploader(
# #     "Upload a road photo (JPG/PNG)",
# #     type=["jpg","jpeg","png"],
# #     key=st.session_state.uploader_key,
# #     help="A clear road-surface photo works best."
# # )
# # if uploader is not None:
# #     img = Image.open(uploader).convert("RGB")
# #     st.session_state.img_bytes = uploader.getvalue()
# #     st.session_state.img_name = uploader.name
# #     st.image(img, caption="Input image", use_container_width=True)

# #     with st.spinner("Analyzing the image…"):
# #         session = get_model_session()
# #         st.session_state.pred = session(img)
# #     st.session_state.report_sent = False

# #     # EXIF prefill (do not auto-confirm; just show pin + ask to confirm)
# #     gps = exif_latlon(Image.open(io.BytesIO(st.session_state.img_bytes)))
# #     if gps:
# #         st.session_state.live_point = (float(gps[0]), float(gps[1]))
# #         st.session_state.map_center = st.session_state.live_point
# #         st.session_state.map_zoom = 16
# #         st.session_state.exif_prefilled = True

# # # ---- results ----
# # if st.session_state.pred:
# #     res = st.session_state.pred
# #     type_txt = res["surface_type"].replace("_"," ").title()
# #     qual_raw = res["surface_quality"]
# #     qual_txt = qual_raw.replace("_"," ").title()
# #     qcls = "qual-ok" if qual_raw in ("excellent","good") else ("qual-mid" if qual_raw=="intermediate" else "qual-bad")

# #     st.markdown("<div class='section-title'>Predicted Surface Type</div>", unsafe_allow_html=True)
# #     st.markdown(f"<div class='result-value'>{type_txt}</div>", unsafe_allow_html=True)

# #     st.markdown("<div class='section-title'>Predicted Surface Quality</div>", unsafe_allow_html=True)
# #     st.markdown(f"<div class='result-value {qcls}'>{qual_txt}</div>", unsafe_allow_html=True)

# #     # optional gate
# #     if qual_raw in {"excellent", "good", "intermediate"} and not st.session_state.allow_submit_anyway:
# #         st.info("This road seems to be in a decent state. Are you sure you want to report it?")
# #         st.session_state.allow_submit_anyway = st.toggle("Report anyway", value=st.session_state.allow_submit_anyway)
# #         if not st.session_state.allow_submit_anyway:
# #             st.stop()

# #     st.divider()
# #     st.subheader("Confirm location of the damage")

# #     # user hint based on EXIF presence
# #     if st.session_state.exif_prefilled and st.session_state.live_point:
# #         st.success("Location found from your picture (GPS metadata). Please confirm the pin or click elsewhere to adjust.")
# #     else:
# #         st.warning("Location not found from your picture — please click on the map to drop a pin.")

# #     # ---- FRAGMENT: map + confirm + (after confirm) office + email ----
# #     @st.fragment
# #     def map_and_followups():
# #         # MAP (don't persist center/zoom to avoid flicker; only track clicks)
# #         m = folium.Map(
# #             location=st.session_state.map_center,
# #             zoom_start=st.session_state.map_zoom,
# #             control_scale=True,
# #             tiles="OpenStreetMap"
# #         )
# #         if st.session_state.live_point:
# #             folium.Marker(st.session_state.live_point, tooltip="Selected location").add_to(m)

# #         out = st_folium(m, width=720, height=420, key="map-main", returned_objects=["last_clicked"])
# #         if out and out.get("last_clicked"):
# #             click = out["last_clicked"]
# #             if click.get("lat") and click.get("lng"):
# #                 st.session_state.live_point = (float(click["lat"]), float(click["lng"]))
# #                 st.session_state.location_confirmed = False
# #                 st.session_state.confirmed_point = None
# #                 st.session_state.ward_office = None
# #                 st.session_state.exif_prefilled = False  # user changed it

# #         if st.session_state.live_point:
# #             lat, lon = st.session_state.live_point
# #             st.caption(f"Chosen coordinates: <span class='coords'>{lat:.6f}, {lon:.6f}</span>", unsafe_allow_html=True)

# #         # CONFIRM
# #         st.button(
# #             "Confirm location",
# #             key="confirm-location-btn",
# #             type="primary",
# #             use_container_width=True,
# #             disabled=st.session_state.live_point is None,
# #             on_click=lambda: st.session_state.update(
# #                 location_confirmed=True,
# #                 confirmed_point=st.session_state.live_point
# #             )
# #         )

# #         # AFTER CONFIRM: office lookup & email (inside same fragment → immediate)
# #         if st.session_state.location_confirmed and st.session_state.confirmed_point and not st.session_state.report_sent:
# #             st.divider()
# #             st.subheader("Closest ward/city office ↩")

# #             if st.session_state.ward_office is None:
# #                 with st.spinner("Looking up nearest office…"):
# #                     lat, lon = st.session_state.confirmed_point
# #                     office = find_nearest_ward_office(lat, lon, salt=random.randint(0,99999)) or city_email_fallback(lat, lon)
# #                     st.session_state.ward_office = office

# #             office = st.session_state.ward_office
# #             if office:
# #                 st.success(f"Closest office: **{office['name']}**")
# #                 st.caption(f"📍 {office['lat']:.6f}, {office['lon']:.6f}")
# #                 st.link_button("Open office in Google Maps", gmaps_link(office['lat'], office['lon']), use_container_width=True)
# #             else:
# #                 st.warning("Couldn’t identify a nearby office automatically. You can still send the report (coordinates included).")
# #                 st.button("Retry lookup", key="retry-office", on_click=lambda: st.session_state.update(ward_office=None))

# #             # SEND REPORT
# #             st.divider()
# #             st.subheader("Send the report")

# #             if office and office.get("email"):
# #                 display_recipient = office["email"]
# #             else:
# #                 display_recipient = romaji_email_from_name(office["name"] if office else "Ward Office")
# #             st.caption("Email of the ward office — hidden to avoid spam.")
# #             st.text_input("Recipient", value=mask_email(display_recipient), disabled=True)

# #             actual_recipient = DEMO_RECEIVER or GMAIL_USER or display_recipient
# #             type_txt = res["surface_type"].replace("_"," ").title()
# #             qual_txt = res["surface_quality"].replace("_"," ").title()
# #             latc, lonc = st.session_state.confirmed_point
# #             when = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# #             office_addr_or_link = (office.get('address') if (office and office.get('address'))
# #                                    else (gmaps_link(office['lat'], office['lon']) if office else gmaps_link(latc, lonc)))
# #             core_text = f"""RoadSafe report (demo)
# # When: {when}
# # Surface type: {type_txt}
# # Surface quality: {qual_txt}
# # Location: {latc:.6f}, {lonc:.6f}
# # Google Maps: {gmaps_link(latc, lonc)}
# # Ward office: {(office['name'] if office else 'N/A')}
# # Address: {office_addr_or_link}
# # """
# #             st.text_area("Preview (read-only)", value=core_text, height=160, disabled=True)
# #             comment = st.text_area("Additional comments (optional)", placeholder="Add any useful details…")

# #             html = f"""
# #             <div style="font-family:system-ui;">
# #               <h2>RoadSafe report (demo)</h2>
# #               <p style="color:#555">When: {when}</p>
# #               <img src="cid:photo1" style="max-width:640px;border-radius:8px;margin:8px 0 16px"/>
# #               <ul style="line-height:1.6">
# #                 <li><b>Surface type:</b> {type_txt}</li>
# #                 <li><b>Surface quality:</b> {qual_txt}</li>
# #                 <li><b>Location:</b> {latc:.6f}, {lonc:.6f}</li>
# #                 <li><b>Google Maps:</b> <a href="{gmaps_link(latc,lonc)}">{gmaps_link(latc,lonc)}</a></li>
# #                 <li><b>Ward office:</b> {(office['name'] if office else 'N/A')}</li>
# #                 <li><b>Address:</b> {office_addr_or_link}</li>
# #               </ul>
# #               {"<p><b>Additional comments:</b><br>"+comment.replace('\n','<br>')+"</p>" if comment.strip() else ""}
# #               <p class="tiny">This message was generated for demo purposes.</p>
# #             </div>
# #             """.strip()

# #             text = core_text + (f"\nAdditional comments:\n{comment}\n" if comment.strip() else "")

# #             if st.button("Send report", type="primary", use_container_width=True, key="send-report-btn"):
# #                 ok, info = send_email_html(
# #                     to_email=actual_recipient,
# #                     subject="RoadSafe Report (Demo)",
# #                     html_body=html,
# #                     text_fallback=text,
# #                     image_bytes=st.session_state.img_bytes,
# #                     image_name=st.session_state.img_name or "photo.jpg",
# #                 )
# #                 if ok:
# #                     st.success("Report submitted successfully, thank you for your contribution.")
# #                     time.sleep(2)
# #                     st.toast("The current page will now refresh — feel free to submit another damage!", icon="✅")
# #                     time.sleep(3)
# #                     _hard_reset()
# #                 else:
# #                     st.error(f"Failed to send email: {info}")

# #     # render the fragment
# #     map_and_followups()

# # # ---- footer ----
# # st.divider()
# # st.caption("Env vars: HF_REPO_ID, HF_FILENAME, (HF_TOKEN if private), GMAIL_USER, GMAIL_PASSWORD, optional DEMO_RECEIVER.")



# # RoadSafe — Streamlit app (KL default, robust Overpass, robust EXIF, clean reset)
# # -------------------------------------------------------------------------------

# import io, os, re, math, gzip, time, random
# from datetime import datetime
# from email.message import EmailMessage

# import streamlit as st
# from PIL import Image, ExifTags
# import pandas as pd
# import requests
# import folium
# from streamlit_folium import st_folium

# # ML
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
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

# # Kuala Lumpur city center
# KL_DEFAULT = (3.139003, 101.686855)

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
#     def session(pil_img: Image.Image):
#         x = tfm(pil_img).unsqueeze(0).to(device)
#         t_logits, q_logits = model(x)
#         t_probs = F.softmax(t_logits, dim=1).squeeze(0).cpu().tolist()
#         q_probs = F.softmax(q_logits, dim=1).squeeze(0).cpu().tolist()
#         t_idx = int(torch.tensor(t_probs).argmax())
#         q_idx = int(torch.tensor(q_probs).argmax())
#         return {
#             "surface_type": MATERIAL_NAMES[t_idx],
#             "surface_quality": QUALITY_NAMES[q_idx],
#             "surface_type_probs": t_probs,
#             "surface_quality_probs": q_probs,
#         }
#     return session

# # ---- EXIF (robust) ----
# _GPS_TAGS = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), 34853)

# def _rat_to_float(x):
#     # Handle (num,den), IFDRational, ints, floats, strings
#     try:
#         if isinstance(x, tuple) and len(x) == 2:
#             num, den = x
#             den = float(den) if den not in (0, 0.0) else 1.0
#             return float(num) / den
#         return float(x)
#     except Exception:
#         try:
#             return float(str(x))
#         except Exception:
#             return None

# def _dms_to_deg(dms):
#     if not (isinstance(dms, (list, tuple)) and len(dms) == 3):
#         return None
#     d = _rat_to_float(dms[0]); m = _rat_to_float(dms[1]); s = _rat_to_float(dms[2])
#     if None in (d, m, s): return None
#     return d + (m / 60.0) + (s / 3600.0)

# def exif_latlon(img_or_bytes):
#     """
#     Robustly extract GPS lat/lon from EXIF. Works with Pillow getexif/_getexif,
#     different rational formats, and optionally piexif if available.
#     """
#     # Try Pillow first
#     try:
#         img = None
#         if isinstance(img_or_bytes, (bytes, bytearray)):
#             img = Image.open(io.BytesIO(img_or_bytes))
#         elif isinstance(img_or_bytes, Image.Image):
#             img = img_or_bytes
#         else:
#             return None

#         data = None
#         if hasattr(img, "getexif"):
#             try:
#                 data = img.getexif()
#             except Exception:
#                 data = None
#         if not data and hasattr(img, "_getexif"):
#             try:
#                 data = img._getexif()
#             except Exception:
#                 data = None
#         if not data:
#             raise ValueError("no EXIF")

#         # get GPS IFD dict
#         gps_ifd = None
#         try:
#             # For Exif object (Pillow >=7)
#             gps_ifd = data.get_ifd(_GPS_TAGS)  # might raise
#         except Exception:
#             gps_ifd = data.get(_GPS_TAGS) if isinstance(data, dict) else None

#         if not gps_ifd:
#             raise ValueError("no GPS IFD")

#         # Map tag ids -> names
#         gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

#         lat_dms = gps.get("GPSLatitude"); lon_dms = gps.get("GPSLongitude")
#         if not lat_dms or not lon_dms:
#             raise ValueError("no GPS lat/lon values")

#         lat = _dms_to_deg(lat_dms); lon = _dms_to_deg(lon_dms)
#         if lat is None or lon is None:
#             raise ValueError("cannot convert DMS")

#         lat_ref = str(gps.get("GPSLatitudeRef", "N")).upper()
#         lon_ref = str(gps.get("GPSLongitudeRef", "E")).upper()
#         if lat_ref.startswith("S"): lat = -lat
#         if lon_ref.startswith("W"): lon = -lon
#         return (float(lat), float(lon))
#     except Exception:
#         # Optional fallback using piexif, if available
#         try:
#             import piexif
#             if isinstance(img_or_bytes, (bytes, bytearray)):
#                 exif_dict = piexif.load(img_or_bytes)
#             else:
#                 buf = io.BytesIO()
#                 img_or_bytes.save(buf, format="JPEG")
#                 exif_dict = piexif.load(buf.getvalue())
#             gps = exif_dict.get("GPS", {})
#             lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
#             lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
#             lat_ref = (gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N") or b"N").decode(errors="ignore").upper()
#             lon_ref = (gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E") or b"E").decode(errors="ignore").upper()
#             if lat_dms and lon_dms:
#                 def rr(v):
#                     return (v[0] / v[1]) if isinstance(v, tuple) and len(v)==2 and v[1] else float(v)
#                 lat = rr(lat_dms[0]) + rr(lat_dms[1])/60 + rr(lat_dms[2])/3600
#                 lon = rr(lon_dms[0]) + rr(lon_dms[1])/60 + rr(lon_dms[2])/3600
#                 if lat_ref.startswith("S"): lat = -lat
#                 if lon_ref.startswith("W"): lon = -lon
#                 return (float(lat), float(lon))
#         except Exception:
#             pass
#         return None

# def gmaps_link(lat, lon): return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# # ---- distance helper ----
# def _haversine(a,b,c,d):
#     R=6371.0; to=math.pi/180.0
#     dlat,dlon=(c-a)*to,(d-b)*to
#     x=math.sin(dlat/2)**2+math.cos(a*to)*math.cos(c*to)*math.sin(dlon/2)**2
#     return 2*R*math.atan2(math.sqrt(x), math.sqrt(1-x))

# # ---- robust worldwide (Malay-aware) Overpass + Nominatim fallback ----
# @st.cache_data(ttl=300, show_spinner=False)
# def find_nearest_ward_office(lat: float, lon: float, salt: int = 0):
#     lat = float(lat); lon = float(lon)
#     NAME_REGEX = r"(?i)(Ward Office|City Office|City Hall|Town Hall|City Council|Municipal Council|Majlis Perbandaran|Majlis Bandaraya|Majlis Daerah|Dewan Bandaraya|Pejabat Daerah|Pejabat Tanah|Pejabat Kerajaan|Pejabat Majlis|区役所|市役所|町役場|村役場|役場|出張所)"
#     UA = {"User-Agent":"roadsafe-demo/1.2 (+contact@example.com)","Accept":"application/json","Accept-Encoding":"gzip"}
#     MIRRORS = [
#         "https://overpass-api.de/api/interpreter",
#         "https://overpass.kumi.systems/api/interpreter",
#         "https://overpass.openstreetmap.fr/api/interpreter",
#     ]
#     EN_TOK = [t.lower() for t in ["Ward Office","City Office","City Hall","Town Hall","City Council","Municipal Council"]]
#     MS_TOK = [t.lower() for t in ["Majlis Perbandaran","Majlis Bandaraya","Majlis Daerah","Dewan Bandaraya","Pejabat Daerah","Pejabat Tanah","Pejabat Kerajaan","Pejabat Majlis","Majlis"]]
#     JP_TOK = [t.lower() for t in ["区役所","市役所","町役場","村役場","役場","出張所"]]

#     def _mk_query(r):
#         return f"""[out:json][timeout:25];
#         (
#           node["amenity"="townhall"](around:{r},{lat},{lon});
#           way ["amenity"="townhall"](around:{r},{lat},{lon});
#           rel ["amenity"="townhall"](around:{r},{lat},{lon});

#           node["office"="government"](around:{r},{lat},{lon});
#           way ["office"="government"](around:{r},{lat},{lon});
#           rel ["office"="government"](around:{r},{lat},{lon});

#           node["office"="administrative"](around:{r},{lat},{lon});
#           way ["office"="administrative"](around:{r},{lat},{lon});
#           rel ["office"="administrative"](around:{r},{lat},{lon});

#           node["government"](around:{r},{lat},{lon});
#           way ["government"](around:{r},{lat},{lon});
#           rel ["government"](around:{r},{lat},{lon});

#           node["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#           way ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#           rel ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
#         );
#         out center tags;"""

#     def _center(el):
#         if el.get("type") == "node": return el.get("lat"), el.get("lon")
#         c = el.get("center") or {}; return c.get("lat"), c.get("lon")

#     def _addr(tags):
#         parts = [
#             tags.get("addr:postcode"),
#             tags.get("addr:state") or tags.get("addr:province") or tags.get("addr:region"),
#             tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:county"),
#             tags.get("addr:district") or tags.get("addr:ward") or tags.get("addr:suburb"),
#             tags.get("addr:street"), tags.get("addr:block_number"), tags.get("addr:housenumber"),
#         ]
#         s = " ".join([p for p in parts if p]).strip()
#         return s or None

#     def _score(tags, dkm):
#         nm = (tags.get("name:en") or tags.get("official_name:en") or
#               tags.get("name:ms") or tags.get("official_name:ms") or
#               tags.get("name") or tags.get("official_name") or
#               tags.get("name:ja") or "")
#         nml = nm.lower()
#         has_name = 1.0 if nm else 0.0
#         amenity = 1.2 if tags.get("amenity") == "townhall" else 0.0
#         admin   = 0.8 if (tags.get("office") in {"government","administrative"} or "government" in tags) else 0.0
#         hit_en  = 1.2 if any(t in nml for t in EN_TOK) else 0.0
#         hit_ms  = 1.5 if any(t in nml for t in MS_TOK) else 0.0
#         hit_jp  = 1.0 if any(t in nml for t in JP_TOK) else 0.0
#         addr    = 0.5 if _addr(tags) else 0.0
#         email   = 0.25 if (tags.get("contact:email") or tags.get("email")) else 0.0
#         return (amenity + admin + has_name + hit_en + hit_ms + hit_jp + addr + email) - dkm

#     def _best(elements):
#         best=None
#         for el in elements:
#             latc, lonc = _center(el)
#             if latc is None or lonc is None: continue
#             dkm = _haversine(lat, lon, float(latc), float(lonc))
#             tags = el.get("tags", {}) or {}
#             score = _score(tags, dkm)
#             name = (tags.get("name:en") or tags.get("official_name:en") or
#                     tags.get("name:ms") or tags.get("official_name:ms") or
#                     tags.get("name") or tags.get("official_name") or
#                     tags.get("name:ja") or "(Unnamed Government Office)")
#             item = {
#                 "name": name, "lat": float(latc), "lon": float(lonc),
#                 "distance_km": float(dkm), "address": _addr(tags),
#                 "email": tags.get("contact:email") or tags.get("email"), "score": score
#             }
#             if best is None or item["score"] > best["score"]: best=item
#         return best

#     for r in (1500, 3000, 6000, 12000, 20000, 35000):
#         q = _mk_query(r)
#         for base in MIRRORS:
#             try:
#                 js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
#             except Exception:
#                 continue
#             best = _best(js.get("elements", []))
#             if best: return best

#     # Nominatim fallback
#     try:
#         d = 0.25
#         viewbox = f"{lon-d},{lat+d},{lon+d},{lat-d}"
#         queries = [
#             "majlis bandaraya","majlis perbandaran","majlis daerah",
#             "dewan bandaraya","city hall","municipal council","city council","town hall",
#             "pejabat daerah","pejabat kerajaan",
#         ]
#         best=None
#         for q in queries:
#             try:
#                 resp = requests.get(
#                     "https://nominatim.openstreetmap.org/search",
#                     params={"q": q,"format": "jsonv2","limit": 10,"addressdetails": 1,"viewbox": viewbox,"bounded": 1},
#                     headers=UA, timeout=10
#                 )
#                 hits = resp.json()
#             except Exception:
#                 continue
#             for h in hits:
#                 latc, lonc = float(h["lat"]), float(h["lon"])
#                 dkm = _haversine(lat, lon, latc, lonc)
#                 name = h.get("name") or (h.get("display_name","").split(",")[0]) or "Nearest Government Office"
#                 a = h.get("address", {})
#                 addr = None
#                 if a:
#                     addr_parts = [a.get("postcode"), a.get("state") or a.get("region"),
#                                   a.get("city") or a.get("town") or a.get("county"),
#                                   a.get("suburb") or a.get("neighbourhood"),
#                                   a.get("road"), a.get("house_number")]
#                     addr = " ".join([p for p in addr_parts if p]).strip() or None
#                 item = {"name": name,"lat": latc,"lon": lonc,"distance_km": dkm,"address": addr,"email": None,"score": 0.0}
#                 if (best is None) or (dkm < best["distance_km"]): best=item
#         if best: return best
#     except Exception:
#         pass

#     return None

# # ---- optional CSV fallback (city,email) ----
# @st.cache_data(show_spinner=False)
# def load_city_csv():
#     candidates = [
#         "city_malaysia.csv","ward_offices.csv","city.csv","csv.csv",
#         "/mnt/data/city_malaysia.csv"
#     ]
#     for path in candidates:
#         if os.path.exists(path):
#             try:
#                 df = pd.read_csv(path)
#                 if {"city","email"}.issubset(df.columns):
#                     return df
#             except Exception:
#                 pass
#     return None

# def city_email_fallback(lat, lon):
#     try:
#         j=requests.get("https://nominatim.openstreetmap.org/reverse",
#                        params={"format":"jsonv2","lat":lat,"lon":lon,"zoom":14,"addressdetails":1},
#                        headers={"User-Agent":"roadsafe-demo"}, timeout=12).json()
#         city=j.get("address",{}).get("city") or j.get("address",{}).get("town") or j.get("address",{}).get("county")
#         df=load_city_csv()
#         if df is not None and city:
#             hit=df[df["city"].str.lower()==city.lower()]
#             if not hit.empty:
#                 return {"name": f"{city} Municipal Office", "email": hit.iloc[0]["email"], "lat":lat,"lon":lon,"address":None}
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

# # ---- STATE ----
# defaults = dict(
#     pred=None, img_bytes=None, img_name=None,
#     live_point=None, confirmed_point=None,
#     map_center=KL_DEFAULT, map_zoom=12,
#     location_confirmed=False, ward_office=None, report_sent=False,
#     allow_submit_anyway=False, exif_prefilled=False,
# )
# for k,v in defaults.items(): st.session_state.setdefault(k,v)
# st.session_state.setdefault("uploader_key", f"uploader-{int(time.time()*1000)}")

# def _hard_reset():
#     for k in list(defaults.keys()):
#         st.session_state[k] = defaults[k]
#     st.session_state["uploader_key"] = f"uploader-{int(time.time()*1000)}"
#     time.sleep(0.05)
#     st.rerun()

# # ---- header ----
# st.markdown("""
# <div class='center'>
#   <h1 style="margin-bottom:0.2rem;">🛣️ RoadSafe — Report Road Damage</h1>
#   <p class='muted' style='margin-top:0'>Upload → Validate → Pinpoint → Notify Ward Office</p>
# </div>
# """, unsafe_allow_html=True)

# # ---- upload → predict ----
# uploader = st.file_uploader(
#     "Upload a road photo (JPG/PNG)",
#     type=["jpg","jpeg","png"],
#     key=st.session_state.uploader_key,
#     help="A clear road-surface photo works best."
# )
# if uploader is not None:
#     img = Image.open(uploader).convert("RGB")
#     st.session_state.img_bytes = uploader.getvalue()
#     st.session_state.img_name = uploader.name
#     st.image(img, caption="Input image", use_container_width=True)

#     with st.spinner("Analyzing the image…"):
#         session = get_model_session()
#         st.session_state.pred = session(img)
#     st.session_state.report_sent = False

#     # EXIF prefill (no auto-confirm)
#     gps = exif_latlon(st.session_state.img_bytes)
#     if gps:
#         st.session_state.live_point = (float(gps[0]), float(gps[1]))
#         st.session_state.map_center = st.session_state.live_point
#         st.session_state.map_zoom = 16
#         st.session_state.exif_prefilled = True

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

#     if st.session_state.exif_prefilled and st.session_state.live_point:
#         st.success("Location found from your picture (GPS metadata). Please confirm the pin or click elsewhere to adjust.")
#     else:
#         st.warning("Location not found from your picture — please click on the map to drop a pin.")

#     @st.fragment
#     def map_and_followups():
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
#                 st.session_state.location_confirmed = False
#                 st.session_state.confirmed_point = None
#                 st.session_state.ward_office = None
#                 st.session_state.exif_prefilled = False

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
#             st.subheader("Closest ward/city office ↩")

#             if st.session_state.ward_office is None:
#                 with st.spinner("Looking up nearest office…"):
#                     lat, lon = st.session_state.confirmed_point
#                     office = find_nearest_ward_office(lat, lon, salt=random.randint(0,99999)) or city_email_fallback(lat, lon)
#                     st.session_state.ward_office = office

#             office = st.session_state.ward_office
#             if office:
#                 st.success(f"Closest office: **{office['name']}**")
#                 st.caption(f"📍 {office['lat']:.6f}, {office['lon']:.6f}")
#                 st.link_button("Open office in Google Maps", gmaps_link(office['lat'], office['lon']), use_container_width=True)
#             else:
#                 st.warning("Couldn’t identify a nearby office automatically. You can still send the report (coordinates included).")
#                 st.button("Retry lookup", key="retry-office", on_click=lambda: st.session_state.update(ward_office=None))

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

#             office_addr_or_link = (office.get('address') if (office and office.get('address'))
#                                    else (gmaps_link(office['lat'], office['lon']) if office else gmaps_link(latc, lonc)))
#             core_text = f"""RoadSafe report (demo)
# When: {when}
# Surface type: {type_txt}
# Surface quality: {qual_txt}
# Location: {latc:.6f}, {lonc:.6f}
# Google Maps: {gmaps_link(latc, lonc)}
# Ward office: {(office['name'] if office else 'N/A')}
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
#                 <li><b>Ward office:</b> {(office['name'] if office else 'N/A')}</li>
#                 <li><b>Address:</b> {office_addr_or_link}</li>
#               </ul>
#               {"<p><b>Additional comments:</b><br>"+comment.replace('\n','<br>')+"</p>" if comment.strip() else ""}
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
#                     time.sleep(3)
#                     _hard_reset()
#                 else:
#                     st.error(f"Failed to send email: {info}")

#     map_and_followups()

# # ---- footer ----
# st.divider()
# st.caption(
#     "[Wahab Arina](https://www.linkedin.com/in/arina-w/)  |  "
#     "[Sadaka Elie](https://www.linkedin.com/in/eliesdk/)  |  "
#     "[Scuderi Marcello](https://www.linkedin.com/in/marcelloscuderi/)"
# )


# RoadSafe — Streamlit app (KL default, robust Overpass, robust EXIF, clean reset)
# -------------------------------------------------------------------------------

import io, os, re, math, gzip, time, random
from datetime import datetime
from email.message import EmailMessage

import streamlit as st
from PIL import Image, ExifTags
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium

# ML
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Kuala Lumpur city center
KL_DEFAULT = (3.139003, 101.686855)

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
    def session(pil_img: Image.Image):
        x = tfm(pil_img).unsqueeze(0).to(device)
        t_logits, q_logits = model(x)
        t_probs = F.softmax(t_logits, dim=1).squeeze(0).cpu().tolist()
        q_probs = F.softmax(q_logits, dim=1).squeeze(0).cpu().tolist()
        t_idx = int(torch.tensor(t_probs).argmax())
        q_idx = int(torch.tensor(q_probs).argmax())
        return {
            "surface_type": MATERIAL_NAMES[t_idx],
            "surface_quality": QUALITY_NAMES[q_idx],
            "surface_type_probs": t_probs,
            "surface_quality_probs": q_probs,
        }
    return session

# ---- EXIF (robust) ----
_GPS_TAGS = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), 34853)

def _rat_to_float(x):
    # Handle (num,den), IFDRational, ints, floats, strings
    try:
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            den = float(den) if den not in (0, 0.0) else 1.0
            return float(num) / den
        return float(x)
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return None

def _dms_to_deg(dms):
    if not (isinstance(dms, (list, tuple)) and len(dms) == 3):
        return None
    d = _rat_to_float(dms[0]); m = _rat_to_float(dms[1]); s = _rat_to_float(dms[2])
    if None in (d, m, s): return None
    return d + (m / 60.0) + (s / 3600.0)

def exif_latlon(img_or_bytes):
    """
    Robustly extract GPS lat/lon from EXIF. Works with Pillow getexif/_getexif,
    different rational formats, and optionally piexif if available.
    """
    # Try Pillow first
    try:
        img = None
        if isinstance(img_or_bytes, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img_or_bytes))
        elif isinstance(img_or_bytes, Image.Image):
            img = img_or_bytes
        else:
            return None

        data = None
        if hasattr(img, "getexif"):
            try:
                data = img.getexif()
            except Exception:
                data = None
        if not data and hasattr(img, "_getexif"):
            try:
                data = img._getexif()
            except Exception:
                data = None
        if not data:
            raise ValueError("no EXIF")

        # get GPS IFD dict
        gps_ifd = None
        try:
            # For Exif object (Pillow >=7)
            gps_ifd = data.get_ifd(_GPS_TAGS)  # might raise
        except Exception:
            gps_ifd = data.get(_GPS_TAGS) if isinstance(data, dict) else None

        if not gps_ifd:
            raise ValueError("no GPS IFD")

        # Map tag ids -> names
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

        lat_dms = gps.get("GPSLatitude"); lon_dms = gps.get("GPSLongitude")
        if not lat_dms or not lon_dms:
            raise ValueError("no GPS lat/lon values")

        lat = _dms_to_deg(lat_dms); lon = _dms_to_deg(lon_dms)
        if lat is None or lon is None:
            raise ValueError("cannot convert DMS")

        lat_ref = str(gps.get("GPSLatitudeRef", "N")).upper()
        lon_ref = str(gps.get("GPSLongitudeRef", "E")).upper()
        if lat_ref.startswith("S"): lat = -lat
        if lon_ref.startswith("W"): lon = -lon
        return (float(lat), float(lon))
    except Exception:
        # Optional fallback using piexif, if available
        try:
            import piexif
            if isinstance(img_or_bytes, (bytes, bytearray)):
                exif_dict = piexif.load(img_or_bytes)
            else:
                buf = io.BytesIO()
                img_or_bytes.save(buf, format="JPEG")
                exif_dict = piexif.load(buf.getvalue())
            gps = exif_dict.get("GPS", {})
            lat_dms = gps.get("piexif.GPSIFD.GPSLatitude") if isinstance(gps.get("GPSLatitude"), str) else gps.get("GPSLatitude")
            lon_dms = gps.get("piexif.GPSIFD.GPSLongitude") if isinstance(gps.get("GPSLongitude"), str) else gps.get("GPSLongitude")
            lat_ref = (gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N") or b"N").decode(errors="ignore").upper()
            lon_ref = (gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E") or b"E").decode(errors="ignore").upper()
            if lat_dms and lon_dms:
                def rr(v):
                    return (v[0] / v[1]) if isinstance(v, tuple) and len(v)==2 and v[1] else float(v)
                lat = rr(lat_dms[0]) + rr(lat_dms[1])/60 + rr(lat_dms[2])/3600
                lon = rr(lon_dms[0]) + rr(lon_dms[1])/60 + rr(lon_dms[2])/3600
                if lat_ref.startswith("S"): lat = -lat
                if lon_ref.startswith("W"): lon = -lon
                return (float(lat), float(lon))
        except Exception:
            pass
        return None

def gmaps_link(lat, lon): return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# ---- distance helper ----
def _haversine(a,b,c,d):
    R=6371.0; to=math.pi/180.0
    dlat,dlon=(c-a)*to,(d-b)*to
    x=math.sin(dlat/2)**2+math.cos(a*to)*math.cos(c*to)*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(x), math.sqrt(1-x))

# ---- robust worldwide (Malay-aware) Overpass + Nominatim fallback ----
@st.cache_data(ttl=300, show_spinner=False)
def find_nearest_ward_office(lat: float, lon: float, salt: int = 0):
    lat = float(lat); lon = float(lon)
    NAME_REGEX = r"(?i)(Ward Office|City Office|City Hall|Town Hall|City Council|Municipal Council|Majlis Perbandaran|Majlis Bandaraya|Majlis Daerah|Dewan Bandaraya|Pejabat Daerah|Pejabat Tanah|Pejabat Kerajaan|Pejabat Majlis|区役所|市役所|町役場|村役場|役場|出張所)"
    UA = {"User-Agent":"roadsafe-demo/1.2 (+contact@example.com)","Accept":"application/json","Accept-Encoding":"gzip"}
    MIRRORS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
    ]
    EN_TOK = [t.lower() for t in ["Ward Office","City Office","City Hall","Town Hall","City Council","Municipal Council"]]
    MS_TOK = [t.lower() for t in ["Majlis Perbandaran","Majlis Bandaraya","Majlis Daerah","Dewan Bandaraya","Pejabat Daerah","Pejabat Tanah","Pejabat Kerajaan","Pejabat Majlis","Majlis"]]
    JP_TOK = [t.lower() for t in ["区役所","市役所","町役場","村役場","役場","出張所"]]

    def _mk_query(r):
        return f"""[out:json][timeout:25];
        (
          node["amenity"="townhall"](around:{r},{lat},{lon});
          way ["amenity"="townhall"](around:{r},{lat},{lon});
          rel ["amenity"="townhall"](around:{r},{lat},{lon});

          node["office"="government"](around:{r},{lat},{lon});
          way ["office"="government"](around:{r},{lat},{lon});
          rel ["office"="government"](around:{r},{lat},{lon});

          node["office"="administrative"](around:{r},{lat},{lon});
          way ["office"="administrative"](around:{r},{lat},{lon});
          rel ["office"="administrative"](around:{r},{lat},{lon});

          node["government"](around:{r},{lat},{lon});
          way ["government"](around:{r},{lat},{lon});
          rel ["government"](around:{r},{lat},{lon});

          node["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
          way ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
          rel ["name"~"{NAME_REGEX}"](around:{r},{lat},{lon});
        );
        out center tags;"""

    def _center(el):
        if el.get("type") == "node": return el.get("lat"), el.get("lon")
        c = el.get("center") or {}; return c.get("lat"), c.get("lon")

    def _addr(tags):
        parts = [
            tags.get("addr:postcode"),
            tags.get("addr:state") or tags.get("addr:province") or tags.get("addr:region"),
            tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:county"),
            tags.get("addr:district") or tags.get("addr:ward") or tags.get("addr:suburb"),
            tags.get("addr:street"), tags.get("addr:block_number"), tags.get("addr:housenumber"),
        ]
        s = " ".join([p for p in parts if p]).strip()
        return s or None

    def _score(tags, dkm):
        nm = (tags.get("name:en") or tags.get("official_name:en") or
              tags.get("name:ms") or tags.get("official_name:ms") or
              tags.get("name") or tags.get("official_name") or
              tags.get("name:ja") or "")
        nml = nm.lower()
        has_name = 1.0 if nm else 0.0
        amenity = 1.2 if tags.get("amenity") == "townhall" else 0.0
        admin   = 0.8 if (tags.get("office") in {"government","administrative"} or "government" in tags) else 0.0
        hit_en  = 1.2 if any(t in nml for t in EN_TOK) else 0.0
        hit_ms  = 1.5 if any(t in nml for t in MS_TOK) else 0.0
        hit_jp  = 1.0 if any(t in nml for t in JP_TOK) else 0.0
        addr    = 0.5 if _addr(tags) else 0.0
        email   = 0.25 if (tags.get("contact:email") or tags.get("email")) else 0.0
        return (amenity + admin + has_name + hit_en + hit_ms + hit_jp + addr + email) - dkm

    def _best(elements):
        best=None
        for el in elements:
            latc, lonc = _center(el)
            if latc is None or lonc is None: continue
            dkm = _haversine(lat, lon, float(latc), float(lonc))
            tags = el.get("tags", {}) or {}
            score = _score(tags, dkm)
            name = (tags.get("name:en") or tags.get("official_name:en") or
                    tags.get("name:ms") or tags.get("official_name:ms") or
                    tags.get("name") or tags.get("official_name") or
                    tags.get("name:ja") or "(Unnamed Government Office)")
            item = {
                "name": name, "lat": float(latc), "lon": float(lonc),
                "distance_km": float(dkm), "address": _addr(tags),
                "email": tags.get("contact:email") or tags.get("email"), "score": score
            }
            if best is None or item["score"] > best["score"]: best=item
        return best

    for r in (1500, 3000, 6000, 12000, 20000, 35000):
        q = _mk_query(r)
        for base in MIRRORS:
            try:
                js = requests.post(base, data={"data": q}, headers=UA, timeout=15).json()
            except Exception:
                continue
            best = _best(js.get("elements", []))
            if best: return best

    # Nominatim fallback
    try:
        d = 0.25
        viewbox = f"{lon-d},{lat+d},{lon+d},{lat-d}"
        queries = [
            "majlis bandaraya","majlis perbandaran","majlis daerah",
            "dewan bandaraya","city hall","municipal council","city council","town hall",
            "pejabat daerah","pejabat kerajaan",
        ]
        best=None
        for q in queries:
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": q,"format": "jsonv2","limit": 10,"addressdetails": 1,"viewbox": viewbox,"bounded": 1},
                    headers=UA, timeout=10
                )
                hits = resp.json()
            except Exception:
                continue
            for h in hits:
                latc, lonc = float(h["lat"]), float(h["lon"])
                dkm = _haversine(lat, lon, latc, lonc)
                name = h.get("name") or (h.get("display_name","").split(",")[0]) or "Nearest Government Office"
                a = h.get("address", {})
                addr = None
                if a:
                    addr_parts = [a.get("postcode"), a.get("state") or a.get("region"),
                                  a.get("city") or a.get("town") or a.get("county"),
                                  a.get("suburb") or a.get("neighbourhood"),
                                  a.get("road"), a.get("house_number")]
                    addr = " ".join([p for p in addr_parts if p]).strip() or None
                item = {"name": name,"lat": latc,"lon": lonc,"distance_km": dkm,"address": addr,"email": None,"score": 0.0}
                if (best is None) or (dkm < best["distance_km"]): best=item
        if best: return best
    except Exception:
        pass

    return None

# ---- optional CSV fallback (city,email) ----
@st.cache_data(show_spinner=False)
def load_city_csv():
    candidates = [
        "city_malaysia.csv","ward_offices.csv","city.csv","csv.csv",
        "/mnt/data/city_malaysia.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if {"city","email"}.issubset(df.columns):
                    return df
            except Exception:
                pass
    return None

def city_email_fallback(lat, lon):
    try:
        j=requests.get("https://nominatim.openstreetmap.org/reverse",
                       params={"format":"jsonv2","lat":lat,"lon":lon,"zoom":14,"addressdetails":1},
                       headers={"User-Agent":"roadsafe-demo"}, timeout=12).json()
        city=j.get("address",{}).get("city") or j.get("address",{}).get("town") or j.get("address",{}).get("county")
        df=load_city_csv()
        if df is not None and city:
            hit=df[df["city"].str.lower()==city.lower()]
            if not hit.empty:
                return {"name": f"{city} Municipal Office", "email": hit.iloc[0]["email"], "lat":lat,"lon":lon,"address":None}
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

# ---- STATE ----
defaults = dict(
    pred=None, img_bytes=None, img_name=None,
    live_point=None, confirmed_point=None,
    map_center=KL_DEFAULT, map_zoom=12,
    location_confirmed=False, ward_office=None, report_sent=False,
    allow_submit_anyway=False, exif_prefilled=False,
)
for k,v in defaults.items(): st.session_state.setdefault(k,v)
st.session_state.setdefault("uploader_key", f"uploader-{int(time.time()*1000)}")

def _hard_reset():
    for k in list(defaults.keys()):
        st.session_state[k] = defaults[k]
    st.session_state["uploader_key"] = f"uploader-{int(time.time()*1000)}"
    time.sleep(0.05)
    st.rerun()

# ---- header ----
st.markdown("""
<div class='center'>
  <h1 style="margin-bottom:0.2rem;">RoadSafe — Report Road Damage</h1>
  <p class='muted' style='margin-top:0'>Upload → Validate → Pinpoint → Notify Ward Office</p>
</div>
""", unsafe_allow_html=True)

# ---- upload → predict ----
uploader = st.file_uploader(
    "Upload a road photo (JPG/PNG)",
    type=["jpg","jpeg","png"],
    key=st.session_state.uploader_key,
    help="A clear road-surface photo works best."
)
if uploader is not None:
    img = Image.open(uploader).convert("RGB")
    st.session_state.img_bytes = uploader.getvalue()
    st.session_state.img_name = uploader.name
    st.image(img, caption="Input image", use_container_width=True)

    with st.spinner("Analyzing the image…"):
        session = get_model_session()
        st.session_state.pred = session(img)
    st.session_state.report_sent = False

    # EXIF prefill (no auto-confirm)
    gps = exif_latlon(st.session_state.img_bytes)
    if gps:
        st.session_state.live_point = (float(gps[0]), float(gps[1]))
        st.session_state.map_center = st.session_state.live_point
        st.session_state.map_zoom = 16
        st.session_state.exif_prefilled = True

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

    if st.session_state.exif_prefilled and st.session_state.live_point:
        st.success("Location found from your picture (GPS metadata). Please confirm the pin or click elsewhere to adjust.")
    else:
        st.warning("Location not found from your picture — please click on the map to drop a pin.")

    @st.fragment
    def map_and_followups():
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
                st.session_state.location_confirmed = False
                st.session_state.confirmed_point = None
                st.session_state.ward_office = None
                st.session_state.exif_prefilled = False

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
            st.subheader("Closest ward/city office ↩")

            if st.session_state.ward_office is None:
                with st.spinner("Looking up nearest office…"):
                    lat, lon = st.session_state.confirmed_point
                    office = find_nearest_ward_office(lat, lon, salt=random.randint(0,99999)) or city_email_fallback(lat, lon)
                    st.session_state.ward_office = office

            office = st.session_state.ward_office
            if office:
                st.success(f"Closest office: **{office['name']}**")
                st.caption(f"📍 {office['lat']:.6f}, {office['lon']:.6f}")
                st.link_button("Open office in Google Maps", gmaps_link(office['lat'], office['lon']), use_container_width=True)
            else:
                st.warning("Couldn’t identify a nearby office automatically. You can still send the report (coordinates included).")
                st.button("Retry lookup", key="retry-office", on_click=lambda: st.session_state.update(ward_office=None))

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

            office_addr_or_link = (office.get('address') if (office and office.get('address'))
                                   else (gmaps_link(office['lat'], office['lon']) if office else gmaps_link(latc, lonc)))
            core_text = f"""RoadSafe report (demo)
When: {when}
Surface type: {type_txt}
Surface quality: {qual_txt}
Location: {latc:.6f}, {lonc:.6f}
Google Maps: {gmaps_link(latc, lonc)}
Ward office: {(office['name'] if office else 'N/A')}
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
                <li><b>Ward office:</b> {(office['name'] if office else 'N/A')}</li>
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
                    time.sleep(3)
                    _hard_reset()
                else:
                    st.error(f"Failed to send email: {info}")

    map_and_followups()

# ---- footer ----
st.divider()
st.markdown("""
<style>
  .footer-links { text-align:center; margin-top:8px; }
  .footer-links a {
    color:#b91c1c;           /* dark red */
    font-weight:600;
    text-decoration:none;
  }
  .footer-links a:hover {
    color:#ef4444;           /* softer red on hover */
    text-decoration:underline;
  }
  .footer-sep { color:#9ca3af; margin:0 18px; }
</style>
<div class="footer-links">
  <a href="https://www.linkedin.com/in/eliesdk/" target="_blank" rel="noopener">Wahab Arina</a>
  <span class="footer-sep">|</span>
  <a href="https://www.linkedin.com/in/marcelloscuderi/" target="_blank" rel="noopener">Scuderi Marcello</a>
  <span class="footer-sep">|</span>
    <a href="https://www.linkedin.com/in/eliesdk/" target="_blank" rel="noopener">Sadaka Elie</a>
</div>
""", unsafe_allow_html=True)
