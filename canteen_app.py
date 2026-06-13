import os, pathlib
import unicodedata
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

CNN_MODEL_PATH  = "food_model.h5"
YOLO_MODEL_PATH = "egg.pt"
CLASS_NAMES_TXT = "class_names.txt"
IMG_SIZE        = 128

PRICE_MAP = {
    "cơm":                   10_000,
    "đậu hũ sốt cà":        25_000,
    "cá hú kho":             30_000,
    "thịt kho trứng":        30_000,
    "thịt kho":              25_000,
    "canh chua có cá":       25_000,
    "canh chua không cá":    10_000,
    "sườn nướng":            30_000,
    "canh rau cải thảo":      7_000,
    "canh rau muống":         7_000,
    "rau xào lagim":         10_000,
    "rau xào củ sắn":        10_000,
    "rau xào đậu que":       10_000,
    "rau xào đậu đũa":       10_000,
    "trứng chiên":           25_000,
    "trứng chiên thịt":      30_000,
    "không rõ":                   0,
}

DISPLAY_NAMES = {
    "cơm":                  "Cơm trắng",
    "đậu hũ sốt cà":       "Đậu hũ sốt cà",
    "cá hú kho":            "Cá hú kho",
    "thịt kho trứng":       "Thịt kho trứng",
    "thịt kho":             "Thịt kho",
    "canh chua có cá":      "Canh chua có cá",
    "canh chua không cá":   "Canh chua không cá",
    "sườn nướng":           "Sườn nướng",
    "canh rau cải thảo":    "Canh rau cải thảo",
    "canh rau muống":       "Canh rau muống",
    "rau xào lagim":        "Rau xào lagim",
    "rau xào củ sắn":       "Rau xào củ sắn",
    "rau xào đậu que":      "Rau xào đậu que",
    "rau xào đậu đũa":      "Rau xào đậu đũa",
    "trứng chiên":          "Trứng chiên",
    "trứng chiên thịt":     "Trứng chiên thịt",
    "không rõ":             "Không rõ",
}

EGG_SURCHARGE       = 6_000
YOLO_CONF_THRESHOLD = 0.35
YOLO_CLASS_NAMES    = {0: "egg half", 1: "egg whole"}

# Default compartment regions (x, y, w, h) as ratio of image size
DEFAULT_COMPARTMENTS = {
    "Top-Left":      (0.03, 0.03, 0.44, 0.48),
    "Top-Right":     (0.53, 0.03, 0.44, 0.48),
    "Bottom-Left":   (0.03, 0.55, 0.27, 0.42),
    "Bottom-Center": (0.34, 0.55, 0.32, 0.42),
    "Bottom-Right":  (0.70, 0.55, 0.27, 0.42),
}

def normalize_text(t):
    return unicodedata.normalize("NFC", t.strip().lower())

def load_class_names(path):
    sd = pathlib.Path(__file__).parent
    for c in [pathlib.Path(path), sd / path]:
        if c.exists():
            return [l.strip() for l in c.read_text(encoding="utf-8").splitlines() if l.strip()]
    return list(PRICE_MAP.keys())

@st.cache_resource(show_spinner="Đang tải mô hình CNN…")
def load_cnn():
    if not TF_AVAILABLE: return None
    try: return tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        st.warning(f"CNN: {e}"); return None

@st.cache_resource(show_spinner="Đang tải YOLO…")
def load_yolo():
    if not YOLO_AVAILABLE: return None
    try: return _YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.warning(f"YOLO: {e}"); return None

def crop_compartment(img, region):
    h, w = img.shape[:2]
    x, y = int(region[0]*w), int(region[1]*h)
    return img[y:y+int(region[3]*h), x:x+int(region[2]*w)]

def apply_camera_adjustments(pil_img, brightness, contrast, saturation, sharpness):
    """Apply camera-like adjustments to a PIL image."""
    img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def draw_compartment_overlays(img_np, compartments, active_slot=None):
    """Draw compartment boxes on the image for visualization."""
    overlay = img_np.copy()
    h, w = img_np.shape[:2]
    for slot, (rx, ry, rw, rh) in compartments.items():
        x1 = int(rx * w)
        y1 = int(ry * h)
        x2 = int((rx + rw) * w)
        y2 = int((ry + rh) * h)
        color = (200, 75, 24) if slot == active_slot else (100, 160, 100)
        thickness = 3 if slot == active_slot else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        label_bg_x2 = min(x1 + len(slot)*10 + 10, w)
        cv2.rectangle(overlay, (x1, y1), (label_bg_x2, y1 + 22), color, -1)
        cv2.putText(overlay, slot, (x1 + 4, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay

def predict_dish(model, crop, class_names):
    if model is None:
        idx = np.random.randint(0, len(class_names))
        return class_names[idx], float(np.random.uniform(0.65, 0.98))
    r = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    p = model.predict(np.expand_dims(r.astype("float32")/255., 0), verbose=0)[0]
    i = int(np.argmax(p))
    return (class_names[i] if i < len(class_names) else "không rõ"), float(p[i])

def count_eggs_yolo(yolo, img):
    ann = img.copy()
    if yolo is None:
        n = np.random.randint(1, 4); h, w = img.shape[:2]
        for i in range(n):
            x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
            cv2.rectangle(ann, (x1,y1), (x1+100,y1+100), (220,100,40), 3)
            cv2.putText(ann, f"egg {i+1}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, .7, (220,100,40), 2)
        return n, ann
    res = yolo(img, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    n = 0
    for box in res.boxes:
        n += 1; x1,y1,x2,y2 = map(int, box.xyxy[0])
        lbl = YOLO_CLASS_NAMES.get(int(box.cls[0]), "egg")
        cv2.rectangle(ann, (x1,y1), (x2,y2), (220,100,40), 3)
        cv2.putText(ann, f"{lbl} {float(box.conf[0]):.0%}", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, .65, (220,100,40), 2)
    return n, ann

def fmt(n): return f"{n:,}₫".replace(",", ".")

# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Canteen Auto-Billing", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Instrument Sans', sans-serif;
  color: #1A1916;
}

[data-testid="stAppViewContainer"] {
  background-color: #F5F2EC;
  background-image: radial-gradient(circle, #C8C4BA 1px, transparent 1px);
  background-size: 28px 28px;
}
[data-testid="stHeader"] { display: none !important; }
[data-testid="stMain"] { background: transparent !important; }

.block-container {
  padding-top: 6.5rem !important;
  max-width: 1280px !important;
}

/* ── Sticky navbar ── */
.navbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 99999;
  background: rgba(245, 242, 236, 0.95);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 2px solid #DDD9D0;
  padding: 0 3rem;
  display: flex;
  align-items: center;
  height: 72px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.navbar-wordmark {
  font-family: 'Instrument Serif', serif;
  font-size: 32px;
  font-weight: 600;
  color: #1A1916;
  letter-spacing: -0.2px;
  line-height: 1;
}
.navbar-wordmark em { font-style: italic; color: #C84B18; }
.navbar-sep { width: 1px; height: 24px; background: #DDD9D0; margin: 0 12px; }
.navbar-sub { font-size: 15px; color: #7A7670; font-weight: 500; }
.navbar-demo {
  margin-left: auto;
  font-size: 13px; font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #9A9690;
  border: 1px solid #DDD9D0;
  border-radius: 4px;
  padding: 4px 10px;
  background: #EDEAE3;
}

/* ── Settings panel ── */
.settings-panel {
  background: #FDFCF8;
  border: 1px solid #E4E0D8;
  border-radius: 14px;
  padding: 20px;
  margin-bottom: 16px;
}
.settings-title {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #7A7670;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.settings-title-icon { font-size: 16px; }
.crop-slot-badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: #C84B18;
  color: #fff;
  border-radius: 4px;
  padding: 2px 8px;
  margin-bottom: 10px;
}
.reset-hint {
  font-size: 12px;
  color: #9A9690;
  margin-top: 8px;
  font-style: italic;
}

/* ── Field label ── */
.field-label {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #7A7670;
  margin-bottom: 12px;
}

/* ── Divider ── */
.divider-label {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 32px 0 24px;
}
.divider-line { flex: 1; height: 1px; background: #DDD9D0; }
.divider-text {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #9A9690;
}

/* ── Dish card ── */
.d-slot {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #B0ADA5;
  margin-bottom: 6px;
}
.d-name { font-size: 16px; font-weight: 700; color: #1A1916; line-height: 1.3; margin-bottom: 10px; }
.d-bar { background: #EAE7E0; height: 3px; border-radius: 2px; margin-bottom: 6px; }
.d-fill { height: 3px; border-radius: 2px; }
.d-pct { font-size: 12px; color: #9A9690; margin-bottom: 8px; font-weight: 500; }
.d-price { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600; color: #1F7A42; }

/* ── Egg panel ── */
.egg-row {
  background: #FDFCF8;
  border: 1px solid #E4E0D8;
  border-left: 4px solid #C84B18;
  border-radius: 0 10px 10px 0;
  padding: 16px 18px;
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}
.egg-n { font-family: 'JetBrains Mono', monospace; font-size: 40px; font-weight: 600; color: #C84B18; line-height: 1; flex-shrink: 0; }
.egg-lbl { font-size: 15px; font-weight: 700; color: #1A1916; margin-bottom: 4px; }
.egg-note { font-size: 13px; color: #7A7670; font-weight: 500; }
.egg-pill {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px; font-weight: 600;
  color: #A03010; background: #FDF0E8;
  border: 1px solid #F0C8A8;
  border-radius: 6px; padding: 6px 12px; flex-shrink: 0;
}

/* ── Bill ── */
.bill {
  background: #FDFCF8;
  border: 1px solid #E4E0D8;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
.bill-header { padding: 16px 20px 14px; border-bottom: 2px solid #EAE7E0; }
.bill-header-title { font-size: 16px; font-weight: 700; color: #1A1916; }
.bill-header-sub { font-size: 12px; color: #9A9690; margin-top: 2px; font-weight: 500; }
.b-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 12px 20px;
  border-bottom: 1px solid #F2EFE8;
  font-size: 15px;
  color: #3A3832;
  font-weight: 500;
}
.b-slot { font-size: 12px; color: #B0ADA5; margin-left: 6px; font-weight: 600; }
.b-amt { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: #4A4740; flex-shrink: 0; margin-left: 8px; }
.b-egg {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 12px 20px;
  border-bottom: 1px solid #F0C8A8;
  background: #FDF0E8;
  font-size: 15px;
  color: #A03010;
  font-weight: 600;
}
.bill-total { padding: 20px; display: flex; justify-content: space-between; align-items: center; background: #1A1916; }
.bill-total-lbl { font-size: 14px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #9A9690; }
.bill-total-amt { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 600; color: #6ECC8A; }

/* ── Streamlit overrides ── */
.stButton > button {
  width: 100% !important;
  background: #1A1916 !important;
  color: #F5F2EC !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Instrument Sans', sans-serif !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  padding: 12px 20px !important;
  letter-spacing: 0.02em !important;
  transition: background .15s !important;
}
.stButton > button:hover:not(:disabled) { background: #333028 !important; }
.stButton > button:disabled { background: #E4E0D8 !important; color: #B8B4AC !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: #E4E0D8 !important;
  border-radius: 12px !important;
  background: #FDFCF8 !important;
}
div[data-testid="stImage"] img { border-radius: 8px; }
.stAlert { border-radius: 10px !important; font-size: 14px !important; font-weight: 500 !important; }
hr { border: none !important; border-top: 1px solid #DDD9D0 !important; }
.stRadio label { font-size: 14px !important; font-weight: 500 !important; }
.stSlider label { font-size: 13px !important; font-weight: 600 !important; color: #4A4740 !important; }
.stExpander { border-radius: 12px !important; border-color: #E4E0D8 !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()
demo = (cnn_model is None) or (yolo_model is None)

# ── Init session state for compartments ──
if "compartments" not in st.session_state:
    st.session_state.compartments = dict(DEFAULT_COMPARTMENTS)

# ── Sticky navbar ──
demo_tag = '<span class="navbar-demo">Demo</span>' if demo else ""
st.markdown(f"""
<div class="navbar">
  <div class="navbar-wordmark">Canteen <em>Auto-Billing</em></div>
  <div class="navbar-sep"></div>
  <div class="navbar-sub">Nhận diện món · Tính tiền tự động</div>
  {demo_tag}
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# Layout: left input panel | right settings panel
# ════════════════════════════════════════════════════════
col_in, col_settings = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<div class="field-label">Nguồn ảnh</div>', unsafe_allow_html=True)
    mode = st.radio("src", ["Tải ảnh lên", "Chụp webcam"], horizontal=True, label_visibility="collapsed")
    tray_image_raw = None
    if mode == "Tải ảnh lên":
        up = st.file_uploader("img", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up: tray_image_raw = Image.open(up).convert("RGB")
    else:
        cam = st.camera_input("cam", label_visibility="collapsed")
        if cam: tray_image_raw = Image.open(cam).convert("RGB")

with col_settings:
    # ── Camera adjustments ──
    st.markdown("""
    <div class="settings-title">
      <span class="settings-title-icon"></span> Chỉnh thông số ảnh
    </div>""", unsafe_allow_html=True)

    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        brightness = st.slider(" Độ sáng", 0.5, 2.0, 1.0, 0.05, key="brightness")
        contrast   = st.slider(" Tương phản", 0.5, 2.0, 1.0, 0.05, key="contrast")
    with cam_col2:
        saturation = st.slider(" Màu sắc", 0.0, 2.0, 1.0, 0.05, key="saturation")
        sharpness  = st.slider(" Độ sắc nét", 0.0, 2.0, 1.0, 0.05, key="sharpness")

    if st.button("↺ Reset về mặc định", key="reset_cam"):
        st.session_state.brightness = 1.0
        st.session_state.contrast   = 1.0
        st.session_state.saturation = 1.0
        st.session_state.sharpness  = 1.0
        st.rerun()

# ── Apply camera adjustments ──
tray_image = None
if tray_image_raw:
    tray_image = apply_camera_adjustments(
        tray_image_raw,
        st.session_state.get("brightness", 1.0),
        st.session_state.get("contrast",   1.0),
        st.session_state.get("saturation", 1.0),
        st.session_state.get("sharpness",  1.0),
    )

# ── Crop editor ──
st.markdown("""
<div class="divider-label">
  <div class="divider-line"></div>
  <div class="divider-text">Chỉnh vùng cắt từng ô</div>
  <div class="divider-line"></div>
</div>""", unsafe_allow_html=True)

crop_col_img, crop_col_sliders = st.columns([3, 2], gap="large")

with crop_col_sliders:
    slot_names = list(DEFAULT_COMPARTMENTS.keys())
    active_slot = st.selectbox(
        "Chọn ô để chỉnh:",
        slot_names,
        key="active_slot",
        format_func=lambda s: {"Top-Left":"↖ Top-Left","Top-Right":"↗ Top-Right",
                                "Bottom-Left":"↙ Bottom-Left","Bottom-Center":"↓ Bottom-Center",
                                "Bottom-Right":"↘ Bottom-Right"}.get(s, s)
    )

    st.markdown(f'<div class="crop-slot-badge">{active_slot}</div>', unsafe_allow_html=True)

    cur = st.session_state.compartments[active_slot]
    def_cur = DEFAULT_COMPARTMENTS[active_slot]

    new_x = st.slider("← → Vị trí X (trái/phải)", 0.0, 0.95, float(cur[0]), 0.01, key=f"cx_{active_slot}")
    new_y = st.slider("↑ ↓ Vị trí Y (trên/dưới)", 0.0, 0.95, float(cur[1]), 0.01, key=f"cy_{active_slot}")
    new_w = st.slider("↔ Chiều rộng",              0.05, 0.95, float(cur[2]), 0.01, key=f"cw_{active_slot}")
    new_h = st.slider("↕ Chiều cao",               0.05, 0.95, float(cur[3]), 0.01, key=f"ch_{active_slot}")

    # Update compartment on slider change
    st.session_state.compartments[active_slot] = (new_x, new_y, new_w, new_h)

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button(f"↺ Reset ô này", key=f"reset_{active_slot}"):
            st.session_state.compartments[active_slot] = DEFAULT_COMPARTMENTS[active_slot]
            st.rerun()
    with bcol2:
        if st.button("↺ Reset tất cả", key="reset_all_crops"):
            st.session_state.compartments = dict(DEFAULT_COMPARTMENTS)
            st.rerun()

    st.markdown(f"""
    <div class="reset-hint">
      Mặc định: X={def_cur[0]:.2f} Y={def_cur[1]:.2f} W={def_cur[2]:.2f} H={def_cur[3]:.2f}
    </div>""", unsafe_allow_html=True)

with crop_col_img:
    if tray_image:
        img_np_preview = np.array(tray_image)
        overlay_img = draw_compartment_overlays(img_np_preview, st.session_state.compartments, active_slot)
        st.image(overlay_img, caption="Xem trước vùng cắt (ô đỏ = đang chỉnh)", use_container_width=True)

        # Show current crop preview
        cur_region = st.session_state.compartments[active_slot]
        crop_preview = crop_compartment(img_np_preview, cur_region)
        if crop_preview.size > 0:
            st.image(crop_preview, caption=f"Preview crop: {active_slot}", use_container_width=True)
    else:
        st.markdown("""
        <div style="margin-top:8px;padding:40px 20px;background:#FDFCF8;border:1px dashed #DDD9D0;
          border-radius:12px;text-align:center;">
          <div style="font-size:32px;margin-bottom:10px">🖼️</div>
          <div style="font-size:14px;font-weight:600;color:#3A3832;margin-bottom:6px;">Tải ảnh lên để xem trước vùng cắt</div>
          <div style="font-size:13px;color:#9A9690;">Các ô sẽ hiển thị trên ảnh sau khi tải</div>
        </div>""", unsafe_allow_html=True)

# ── Analyze button ──
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
btn_col, _ = st.columns([1, 2])
with btn_col:
    go = st.button("🍱 Nhận diện & tính tiền", disabled=(tray_image is None))

# ═══════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════
if go and tray_image:
    img_np = np.array(tray_image)
    cnn_results = {}
    with st.spinner("Đang phân tích…"):
        for slot, region in st.session_state.compartments.items():
            crop = crop_compartment(img_np, region)
            dish, conf = predict_dish(cnn_model, crop, CLASS_NAMES)
            dk = normalize_text(dish)
            cnn_results[slot] = dict(
                crop=crop, dish=dish,
                display=DISPLAY_NAMES.get(dk, dish),
                conf=conf, price=PRICE_MAP.get(dk, 0)
            )

    has_egg_dish = any(normalize_text(r["dish"]) == "thịt kho trứng" for r in cnn_results.values())
    egg_count, annotated_np = 0, img_np.copy()
    if has_egg_dish:
        with st.spinner("Đang đếm trứng…"):
            egg_count, annotated_np = count_eggs_yolo(yolo_model, img_np)

    extra_eggs = max(0, egg_count - (1 if has_egg_dish else 0))
    egg_charge = extra_eggs * EGG_SURCHARGE

    st.markdown("""
    <div class="divider-label">
      <div class="divider-line"></div>
      <div class="divider-text">Kết quả phân tích</div>
      <div class="divider-line"></div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="field-label" style="margin-bottom:14px">Các ô trong khay</div>', unsafe_allow_html=True)

        def dish_card(col, slot, r):
            pct = int(r["conf"] * 100)
            bc  = "#1F7A42" if pct >= 80 else "#A06010" if pct >= 60 else "#A03030"
            price_str = fmt(r["price"]) if r["price"] else "—"
            with col:
                st.image(Image.fromarray(r["crop"]), use_container_width=True)
                st.markdown(f"""
                <div style="padding:6px 0 14px">
                  <div class="d-slot">{slot}</div>
                  <div class="d-name">{r["display"]}</div>
                  <div class="d-bar"><div class="d-fill" style="width:{pct}%;background:{bc}"></div></div>
                  <div class="d-pct">{pct}% tin cậy</div>
                  <div class="d-price">{price_str}</div>
                </div>""", unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2, gap="small")
        dish_card(r1c1, "Top-Left",  cnn_results["Top-Left"])
        dish_card(r1c2, "Top-Right", cnn_results["Top-Right"])
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3, gap="small")
        dish_card(r2c1, "Bottom-Left",   cnn_results["Bottom-Left"])
        dish_card(r2c2, "Bottom-Center", cnn_results["Bottom-Center"])
        dish_card(r2c3, "Bottom-Right",  cnn_results["Bottom-Right"])

        if has_egg_dish:
            st.markdown("""
            <div class="divider-label" style="margin-top:24px">
              <div class="divider-line"></div>
              <div class="divider-text">Phát hiện trứng</div>
              <div class="divider-line"></div>
            </div>""", unsafe_allow_html=True)
            st.image(annotated_np, caption=f"YOLO — phát hiện {egg_count} quả", use_container_width=True)

    with right:
        if has_egg_dish:
            note = f"+{extra_eggs} quả thêm · {fmt(EGG_SURCHARGE)}/quả" if extra_eggs > 0 else "1 trứng đã bao gồm trong giá"
            pill = f'<div class="egg-pill">+{fmt(egg_charge)}</div>' if egg_charge > 0 else ""
            st.markdown(f"""
            <div class="egg-row">
              <div class="egg-n">{egg_count}</div>
              <div>
                <div class="egg-lbl">Trứng phát hiện được</div>
                <div class="egg-note">{note}</div>
              </div>
              {pill}
            </div>""", unsafe_allow_html=True)

        total = sum(r["price"] for r in cnn_results.values()) + egg_charge

        rows_html = "".join(f"""
        <div class="b-row">
          <span>{r["display"]}<span class="b-slot">{slot}</span></span>
          <span class="b-amt">{"—" if r["price"]==0 else fmt(r["price"])}</span>
        </div>""" for slot, r in cnn_results.items())

        egg_html = f"""
        <div class="b-egg">
          <span>Trứng thêm ×{extra_eggs}</span>
          <span class="b-amt" style="color:#A03010">+{fmt(egg_charge)}</span>
        </div>""" if egg_charge > 0 else ""

        from datetime import datetime
        now = datetime.now().strftime("%H:%M · %d/%m/%Y")

        st.markdown(f"""
        <div class="bill">
          <div class="bill-header">
            <div class="bill-header-title">Hóa đơn</div>
            <div class="bill-header-sub">{now}</div>
          </div>
          {rows_html}{egg_html}
          <div class="bill-total">
            <div class="bill-total-lbl">Tổng cộng</div>
            <div class="bill-total-amt">{fmt(total)}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.success(f"Thanh toán: **{fmt(total)}**")
