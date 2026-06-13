import os, pathlib, time
import unicodedata
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
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

COMPARTMENTS = {
    "Top-Left":      (0.03, 0.03, 0.44, 0.48),
    "Top-Right":     (0.53, 0.03, 0.44, 0.48),
    "Bottom-Left":   (0.03, 0.55, 0.27, 0.42),
    "Bottom-Center": (0.34, 0.55, 0.32, 0.42),
    "Bottom-Right":  (0.70, 0.55, 0.27, 0.42),
}

# Màu cho từng ô (BGR cho OpenCV)
SLOT_COLORS = {
    "Top-Left":      (52,  199, 89),
    "Top-Right":     (0,   122, 255),
    "Bottom-Left":   (255, 149, 0),
    "Bottom-Center": (175, 82,  222),
    "Bottom-Right":  (255, 59,  48),
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

def draw_compartment_overlays(img_np, cnn_results):
    """Vẽ khung + nhãn realtime lên ảnh gốc"""
    ann = img_np.copy()
    h, w = ann.shape[:2]
    for slot, region in COMPARTMENTS.items():
        x = int(region[0] * w)
        y = int(region[1] * h)
        bw = int(region[2] * w)
        bh = int(region[3] * h)
        color = SLOT_COLORS.get(slot, (255, 255, 255))

        # Vẽ khung bo góc
        thickness = 3
        corner_len = min(bw, bh) // 5
        # 4 góc
        pts = [
            ((x, y+corner_len), (x, y), (x+corner_len, y)),
            ((x+bw-corner_len, y), (x+bw, y), (x+bw, y+corner_len)),
            ((x+bw, y+bh-corner_len), (x+bw, y+bh), (x+bw-corner_len, y+bh)),
            ((x+corner_len, y+bh), (x, y+bh), (x, y+bh-corner_len)),
        ]
        for p1, p2, p3 in pts:
            cv2.line(ann, p1, p2, color, thickness, cv2.LINE_AA)
            cv2.line(ann, p2, p3, color, thickness, cv2.LINE_AA)

        # Nếu đã có kết quả nhận diện → vẽ nhãn
        if cnn_results and slot in cnn_results:
            r = cnn_results[slot]
            label = r["display"]
            conf  = int(r["conf"] * 100)
            price = r["price"]
            price_str = f"{price:,}d".replace(",", ".") if price else ""

            # Nền nhãn (mờ)
            label_text = f"{label}  {conf}%"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            pad = 6
            lx, ly = x + 6, y + 6
            overlay = ann.copy()
            cv2.rectangle(overlay, (lx, ly), (lx + tw + pad*2, ly + th + pad*2 + (18 if price_str else 0)), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.55, ann, 0.45, 0, ann)

            # Text nhãn
            cv2.putText(ann, label_text,
                        (lx + pad, ly + th + pad),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            if price_str:
                cv2.putText(ann, price_str,
                            (lx + pad, ly + th + pad + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 230, 140), 1, cv2.LINE_AA)
    return ann

def fmt(n): return f"{n:,}₫".replace(",", ".")

def run_analysis(img_np, cnn_model, yolo_model, class_names):
    """Chạy toàn bộ pipeline nhận diện, trả về dict kết quả"""
    cnn_results = {}
    for slot, region in COMPARTMENTS.items():
        crop = crop_compartment(img_np, region)
        dish, conf = predict_dish(cnn_model, crop, class_names)
        dk = normalize_text(dish)
        cnn_results[slot] = dict(
            crop=crop, dish=dish,
            display=DISPLAY_NAMES.get(dk, dish),
            conf=conf, price=PRICE_MAP.get(dk, 0)
        )

    has_egg_dish = any(normalize_text(r["dish"]) == "thịt kho trứng" for r in cnn_results.values())
    egg_count, annotated_np = 0, img_np.copy()
    if has_egg_dish:
        egg_count, annotated_np = count_eggs_yolo(yolo_model, img_np)

    extra_eggs = max(0, egg_count - (1 if has_egg_dish else 0))
    egg_charge = extra_eggs * EGG_SURCHARGE
    total = sum(r["price"] for r in cnn_results.values()) + egg_charge

    return {
        "cnn_results": cnn_results,
        "has_egg_dish": has_egg_dish,
        "egg_count": egg_count,
        "annotated_np": annotated_np,
        "extra_eggs": extra_eggs,
        "egg_charge": egg_charge,
        "total": total,
    }

# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Canteen Auto-Billing", page_icon="🍱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif; color: #1A1916; }

[data-testid="stAppViewContainer"] {
  background-color: #F5F2EC;
  background-image: radial-gradient(circle, #C8C4BA 1px, transparent 1px);
  background-size: 28px 28px;
}
[data-testid="stHeader"] { display: none !important; }
[data-testid="stMain"] { background: transparent !important; }

.block-container { padding-top: 6.5rem !important; max-width: 1280px !important; }

.navbar {
  position: fixed; top: 0; left: 0; right: 0; z-index: 99999;
  background: rgba(245,242,236,0.95); backdrop-filter: blur(12px);
  border-bottom: 2px solid #DDD9D0; padding: 0 3rem;
  display: flex; align-items: center; height: 72px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.03);
}
.navbar-wordmark { font-family:'Instrument Serif',serif; font-size:32px; font-weight:600; color:#1A1916; letter-spacing:-0.2px; line-height:1; }
.navbar-wordmark em { font-style:italic; color:#C84B18; }
.navbar-sep { width:1px; height:24px; background:#DDD9D0; margin:0 12px; }
.navbar-sub { font-size:15px; color:#7A7670; font-weight:500; }
.navbar-demo { margin-left:auto; font-size:13px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#9A9690; border:1px solid #DDD9D0; border-radius:4px; padding:4px 10px; background:#EDEAE3; }

/* Realtime badge */
.rt-badge {
  margin-left: auto;
  display: flex; align-items: center; gap: 8px;
  font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: #1F7A42;
  background: #E8F7ED; border: 1px solid #A8DDB8; border-radius: 20px;
  padding: 5px 14px;
}
.rt-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #1F7A42;
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.4; transform:scale(.7); }
}

.field-label { font-size:13px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#7A7670; margin-bottom:12px; }
.divider-label { display:flex; align-items:center; gap:12px; margin:32px 0 24px; }
.divider-line { flex:1; height:1px; background:#DDD9D0; }
.divider-text { font-size:13px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#9A9690; }

/* Mode tabs */
.mode-hint { font-size:13px; color:#9A9690; margin-top:-6px; margin-bottom:14px; font-weight:500; }

.d-slot { font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#B0ADA5; margin-bottom:6px; }
.d-name { font-size:16px; font-weight:700; color:#1A1916; line-height:1.3; margin-bottom:10px; }
.d-bar { background:#EAE7E0; height:3px; border-radius:2px; margin-bottom:6px; }
.d-fill { height:3px; border-radius:2px; }
.d-pct { font-size:12px; color:#9A9690; margin-bottom:8px; font-weight:500; }
.d-price { font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:600; color:#1F7A42; }

.egg-row { background:#FDFCF8; border:1px solid #E4E0D8; border-left:4px solid #C84B18; border-radius:0 10px 10px 0; padding:16px 18px; display:flex; align-items:center; gap:16px; margin-bottom:16px; }
.egg-n { font-family:'JetBrains Mono',monospace; font-size:40px; font-weight:600; color:#C84B18; line-height:1; flex-shrink:0; }
.egg-lbl { font-size:15px; font-weight:700; color:#1A1916; margin-bottom:4px; }
.egg-note { font-size:13px; color:#7A7670; font-weight:500; }
.egg-pill { margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:600; color:#A03010; background:#FDF0E8; border:1px solid #F0C8A8; border-radius:6px; padding:6px 12px; flex-shrink:0; }

.bill { background:#FDFCF8; border:1px solid #E4E0D8; border-radius:14px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.02); }
.bill-header { padding:16px 20px 14px; border-bottom:2px solid #EAE7E0; }
.bill-header-title { font-size:16px; font-weight:700; color:#1A1916; }
.bill-header-sub { font-size:12px; color:#9A9690; margin-top:2px; font-weight:500; }
.b-row { display:flex; justify-content:space-between; align-items:baseline; padding:12px 20px; border-bottom:1px solid #F2EFE8; font-size:15px; color:#3A3832; font-weight:500; }
.b-slot { font-size:12px; color:#B0ADA5; margin-left:6px; font-weight:600; }
.b-amt { font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:600; color:#4A4740; flex-shrink:0; margin-left:8px; }
.b-egg { display:flex; justify-content:space-between; align-items:baseline; padding:12px 20px; border-bottom:1px solid #F0C8A8; background:#FDF0E8; font-size:15px; color:#A03010; font-weight:600; }
.bill-total { padding:20px; display:flex; justify-content:space-between; align-items:center; background:#1A1916; }
.bill-total-lbl { font-size:14px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#9A9690; }
.bill-total-amt { font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:600; color:#6ECC8A; }

.stButton > button { width:100% !important; background:#1A1916 !important; color:#F5F2EC !important; border:none !important; border-radius:10px !important; font-family:'Instrument Sans',sans-serif !important; font-size:15px !important; font-weight:700 !important; padding:12px 20px !important; letter-spacing:.02em !important; transition:background .15s !important; }
.stButton > button:hover:not(:disabled) { background:#333028 !important; }
.stButton > button:disabled { background:#E4E0D8 !important; color:#B8B4AC !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#E4E0D8 !important; border-radius:12px !important; background:#FDFCF8 !important; }
div[data-testid="stImage"] img { border-radius:8px; }
.stAlert { border-radius:10px !important; font-size:14px !important; font-weight:500 !important; }
hr { border:none !important; border-top:1px solid #DDD9D0 !important; }
.stRadio label { font-size:14px !important; font-weight:500 !important; }

/* Freeze button */
.freeze-hint { font-size:13px; color:#7A7670; text-align:center; margin-top:8px; font-weight:500; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = load_class_names(CLASS_NAMES_TXT)
cnn_model   = load_cnn()
yolo_model  = load_yolo()
demo = (cnn_model is None) or (yolo_model is None)

# ── Sticky navbar ──
demo_tag = '<span class="navbar-demo">Demo</span>' if demo else '<span class="rt-badge"><span class="rt-dot"></span>Realtime</span>'
st.markdown(f"""
<div class="navbar">
  <div class="navbar-wordmark">Canteen <em>Auto-Billing</em></div>
  <div class="navbar-sep"></div>
  <div class="navbar-sub">Nhận diện món · Tính tiền tự động</div>
  {demo_tag}
</div>
""", unsafe_allow_html=True)

# ── Session state ──
if "frozen_result" not in st.session_state:
    st.session_state.frozen_result = None
if "frozen_img" not in st.session_state:
    st.session_state.frozen_img = None
if "last_frame_id" not in st.session_state:
    st.session_state.last_frame_id = None

# ── Input panel ──
col_in, col_out = st.columns([1, 2], gap="large")

with col_in:
    st.markdown('<div class="field-label">Nguồn ảnh</div>', unsafe_allow_html=True)
    mode = st.radio("src", ["Tải ảnh lên", "Camera realtime"], horizontal=True, label_visibility="collapsed")

    tray_image = None
    current_frame_id = None

    if mode == "Tải ảnh lên":
        st.markdown('<div class="mode-hint">Tải ảnh lên để nhận diện tức thì</div>', unsafe_allow_html=True)
        up = st.file_uploader("img", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up:
            tray_image = Image.open(up).convert("RGB")
            current_frame_id = up.file_id  # stable ID per upload
            st.session_state.frozen_result = None  # reset freeze khi đổi ảnh
    else:
        st.markdown('<div class="mode-hint">Camera tự động nhận diện mỗi khi bạn chụp frame mới</div>', unsafe_allow_html=True)
        cam = st.camera_input("📷 Hướng camera vào khay cơm", label_visibility="collapsed")
        if cam:
            tray_image = Image.open(cam).convert("RGB")
            current_frame_id = id(cam)

    # Nút Chốt hóa đơn
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    freeze_col, clear_col = st.columns(2)
    with freeze_col:
        do_freeze = st.button("📌 Chốt hóa đơn", disabled=(tray_image is None and st.session_state.frozen_result is None))
    with clear_col:
        do_clear = st.button("🔄 Làm mới", disabled=(st.session_state.frozen_result is None))

    if do_clear:
        st.session_state.frozen_result = None
        st.session_state.frozen_img    = None
        st.rerun()

# ── Realtime analysis (auto, không cần nút) ──
result = None
display_img_np = None

if tray_image is not None:
    img_np = np.array(tray_image)

    # Chạy nhận diện mỗi khi có frame mới
    if current_frame_id != st.session_state.last_frame_id or st.session_state.frozen_result is None:
        with st.spinner(""):
            result = run_analysis(img_np, cnn_model, yolo_model, CLASS_NAMES)
        st.session_state.last_frame_id = current_frame_id

        # Nếu đang frozen thì giữ kết quả cũ, không ghi đè
        if st.session_state.frozen_result is None:
            display_img_np = draw_compartment_overlays(img_np, result["cnn_results"])
        else:
            result = st.session_state.frozen_result
            display_img_np = st.session_state.frozen_img

        if do_freeze and result is not None:
            st.session_state.frozen_result = result
            st.session_state.frozen_img    = display_img_np
    else:
        result = st.session_state.frozen_result
        display_img_np = st.session_state.frozen_img

elif st.session_state.frozen_result is not None:
    result = st.session_state.frozen_result
    display_img_np = st.session_state.frozen_img

# Handle freeze AFTER first analysis
if do_freeze and tray_image is not None and result is not None:
    if st.session_state.frozen_result is None:
        img_np = np.array(tray_image)
        st.session_state.frozen_result = result
        st.session_state.frozen_img    = draw_compartment_overlays(img_np, result["cnn_results"])
    result = st.session_state.frozen_result
    display_img_np = st.session_state.frozen_img

# ── Hiển thị ảnh realtime với overlay ──
with col_in:
    if display_img_np is not None:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.session_state.frozen_result is not None:
            st.markdown('<div style="font-size:12px;font-weight:700;color:#C84B18;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">📌 Đã chốt</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:12px;font-weight:700;color:#1F7A42;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">● Đang nhận diện</div>', unsafe_allow_html=True)
        st.image(display_img_np, use_container_width=True)
    elif tray_image is None and st.session_state.frozen_result is None:
        st.markdown("""
        <div style="margin-top:8px;padding:24px 20px;background:#FDFCF8;border:1px dashed #DDD9D0;
          border-radius:12px;text-align:center;">
          <div style="font-size:14px;font-weight:600;color:#3A3832;margin-bottom:6px;">Hướng camera vào khay cơm</div>
          <div style="font-size:13px;color:#9A9690;">Hệ thống tự động nhận diện & tính tiền ngay</div>
        </div>""", unsafe_allow_html=True)

# ── Kết quả realtime ──
if result:
    cnn_results = result["cnn_results"]
    has_egg_dish = result["has_egg_dish"]
    egg_count    = result["egg_count"]
    annotated_np = result["annotated_np"]
    extra_eggs   = result["extra_eggs"]
    egg_charge   = result["egg_charge"]
    total        = result["total"]

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

        frozen_badge = ' <span style="font-size:11px;color:#C84B18;font-weight:600;letter-spacing:.06em">📌 ĐÃ CHỐT</span>' if st.session_state.frozen_result else ""

        st.markdown(f"""
        <div class="bill">
          <div class="bill-header">
            <div class="bill-header-title">Hóa đơn{frozen_badge}</div>
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
