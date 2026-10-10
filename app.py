"""TomatoGuard - tomato leaf disease detection with weather early warning.

Needs two files next to this one (both downloaded from the Colab notebook):
    tomato_model.keras
    class_names.json
"""
import html
import json
from statistics import mean

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = "tomato_model.keras"
CLASSES_PATH = "class_names.json"
IMG_SIZE = (224, 224)
MIN_CONFIDENCE = 0.60  # below this we ask for a clearer photo

# Public link of the TomatoGuard website. Leave empty to hide the "back to website" link.
WEBSITE_URL = "https://hyperbeat08v-design.github.io/tomatoguard/"

# ---------------------------------------------------------------- texts
UI = {
    "en": {
        "hero_title": "Show us the leaf. We name the disease.",
        "subtitle": "Upload a photo of a tomato leaf. TomatoGuard tells you what it has and what to do next.",
        "steps": ["Take a photo", "Upload it", "Get the plan"],
        "tab_diag": "Check a leaf",
        "tab_weather": "Weather risk",
        "tip": "For the best result, take a close-up of one leaf in good light, with a plain background.",
        "upload": "Upload a clear photo of one tomato leaf",
        "camera": "Or take a photo with the camera",
        "analyzing": "Analyzing the leaf...",
        "result": "Result",
        "confidence": "Confidence",
        "unsure_title": "Not sure about this photo",
        "unsure": "Take a clearer, close-up photo of a single leaf in good light and try again.",
        "all_probs": "Probability for each class",
        "advice": "What to do",
        "disclaimer": "Student project. The model knows only 4 classes (3 diseases + healthy) and is trained on tomato leaves, so a non-tomato or blurry photo can give a wrong answer. Confirm with your local agriculture officer before spraying anything, and follow the product label.",
        "city": "Enter your village or city",
        "get_weather": "Check risk",
        "city_not_found": "Could not find that place. Try the nearest big town.",
        "weather_error": "Could not get weather data right now. Check the internet and try again.",
        "weather_intro": "TomatoGuard checks today's temperature and humidity in your town and shows which disease the weather favours.",
        "today": "Today's average",
        "temp": "Temperature",
        "hum": "Humidity",
        "risk_for": "Disease risk from today's weather",
        "risk_note": "This is a simple rule-based guide for learning, not a scientific forecast.",
        "level": {"low": "Low", "medium": "Medium", "high": "High"},
        "risk_label": {
            "Late_Blight": "Late Blight",
            "Early_Blight": "Early Blight",
            "Leaf_Curl_Virus": "Leaf Curl Virus (whiteflies)",
        },
        "risk_rule": {
            "Late_Blight": "High: humidity 85%+ and 10 to 25 °C",
            "Early_Blight": "High: humidity 80%+ and 24 to 30 °C",
            "Leaf_Curl_Virus": "High: 28 °C or more and humidity 60% or less",
        },
        "model_error": "The model could not be loaded. Make sure tomato_model.keras and class_names.json are in the same folder as app.py.",
        "back": "Back to the website",
        "footer": "TomatoGuard, a PBL project. Not a substitute for advice from an agriculture officer.",
    },
    "hi": {
        "hero_title": "पत्ती दिखाइए, रोग का नाम जानिए।",
        "subtitle": "टमाटर की पत्ती की फोटो अपलोड करें। TomatoGuard बताएगा कि उसमें क्या रोग है और आगे क्या करना है।",
        "steps": ["फोटो लें", "अपलोड करें", "उपाय पाएं"],
        "tab_diag": "पत्ती जांचें",
        "tab_weather": "मौसम जोखिम",
        "tip": "बेहतर नतीजे के लिए अच्छी रोशनी में, सादी पृष्ठभूमि के साथ, एक पत्ती की पास से फोटो लें।",
        "upload": "टमाटर की एक पत्ती की साफ़ फोटो अपलोड करें",
        "camera": "या कैमरे से फोटो लें",
        "analyzing": "पत्ती की जांच हो रही है...",
        "result": "नतीजा",
        "confidence": "भरोसा",
        "unsure_title": "इस फोटो के बारे में पक्का नहीं है",
        "unsure": "अच्छी रोशनी में एक पत्ती की पास से साफ़ फोटो लेकर फिर कोशिश करें।",
        "all_probs": "हर वर्ग की संभावना",
        "advice": "क्या करें",
        "disclaimer": "यह एक छात्र प्रोजेक्ट है। मॉडल सिर्फ 4 वर्ग (3 रोग + स्वस्थ) पहचानता है और टमाटर की पत्तियों पर सिखाया गया है, इसलिए दूसरी फसल या धुंधली फोटो पर गलत नतीजा आ सकता है। कोई भी दवा छिड़कने से पहले स्थानीय कृषि अधिकारी से सलाह लें और उत्पाद के लेबल का पालन करें।",
        "city": "अपना गांव या शहर लिखें",
        "get_weather": "जोखिम देखें",
        "city_not_found": "यह जगह नहीं मिली। नज़दीकी बड़े शहर का नाम लिखें।",
        "weather_error": "अभी मौसम की जानकारी नहीं मिल पाई। इंटरनेट जांचकर फिर कोशिश करें।",
        "weather_intro": "TomatoGuard आपके शहर का आज का तापमान और नमी देखकर बताता है कि मौसम किस रोग के लिए अनुकूल है।",
        "today": "आज का औसत",
        "temp": "तापमान",
        "hum": "नमी",
        "risk_for": "आज के मौसम से रोग का जोखिम",
        "risk_note": "यह सीखने के लिए एक सरल नियम-आधारित अनुमान है, वैज्ञानिक पूर्वानुमान नहीं।",
        "level": {"low": "कम", "medium": "मध्यम", "high": "अधिक"},
        "risk_label": {
            "Late_Blight": "पछेती झुलसा (लेट ब्लाइट)",
            "Early_Blight": "अगेती झुलसा (अर्ली ब्लाइट)",
            "Leaf_Curl_Virus": "पत्ती मोड़क विषाणु (सफेद मक्खी)",
        },
        "risk_rule": {
            "Late_Blight": "अधिक: नमी 85% से ऊपर और 10 से 25 °C",
            "Early_Blight": "अधिक: नमी 80% से ऊपर और 24 से 30 °C",
            "Leaf_Curl_Virus": "अधिक: 28 °C या ज़्यादा और नमी 60% या कम",
        },
        "model_error": "मॉडल लोड नहीं हो सका। जांचें कि tomato_model.keras और class_names.json, app.py वाले ही फोल्डर में हैं।",
        "back": "वेबसाइट पर वापस जाएं",
        "footer": "TomatoGuard, एक PBL प्रोजेक्ट। कृषि अधिकारी की सलाह का विकल्प नहीं।",
    },
}

# Names and advice per class. No spray doses on purpose: those must come from
# the product label or a local agriculture officer.
DISEASE = {
    "Early_Blight": {
        "name": {"en": "Early Blight", "hi": "अगेती झुलसा (अर्ली ब्लाइट)"},
        "kind": "bad",
        "color": "#e0a23b",
        "advice": {
            "en": [
                "Pick off the infected lower leaves and destroy them away from the field.",
                "Rotate crops and keep space between plants so air can flow.",
                "Water at the root (drip or channel), not over the leaves.",
                "If spraying is needed, use a fungicide recommended for tomato, exactly as the label says, after asking your local agriculture officer.",
            ],
            "hi": [
                "संक्रमित निचली पत्तियों को तोड़कर खेत से दूर नष्ट कर दें।",
                "फसल चक्र अपनाएं और पौधों के बीच जगह रखें ताकि हवा चल सके।",
                "पानी जड़ में दें (ड्रिप या नाली से), पत्तियों के ऊपर से नहीं।",
                "छिड़काव ज़रूरी हो तो स्थानीय कृषि अधिकारी से पूछकर टमाटर के लिए सुझाया गया फफूंदनाशक लेबल के अनुसार ही इस्तेमाल करें।",
            ],
        },
    },
    "Late_Blight": {
        "name": {"en": "Late Blight", "hi": "पछेती झुलसा (लेट ब्लाइट)"},
        "kind": "bad",
        "color": "#5fb4c9",
        "advice": {
            "en": [
                "Act fast: this disease can spread across a field in days in cool, damp weather.",
                "Uproot badly infected plants and destroy them. Do not put them in compost.",
                "Keep leaves dry, use drip irrigation and avoid watering in the evening.",
                "Check the crop daily in humid weather and ask your local agriculture officer which fungicide to use, following the label.",
            ],
            "hi": [
                "तुरंत कदम उठाएं: ठंडे और नम मौसम में यह रोग कुछ ही दिनों में पूरे खेत में फैल सकता है।",
                "बहुत संक्रमित पौधों को उखाड़कर नष्ट करें। उन्हें खाद में न डालें।",
                "पत्तियों को सूखा रखें, ड्रिप सिंचाई करें और शाम को पानी न दें।",
                "नमी वाले मौसम में रोज़ फसल देखें और कौन सा फफूंदनाशक लेना है, यह स्थानीय कृषि अधिकारी से पूछें और लेबल का पालन करें।",
            ],
        },
    },
    "Leaf_Curl_Virus": {
        "name": {"en": "Leaf Curl Virus", "hi": "पत्ती मोड़क विषाणु (लीफ कर्ल वायरस)"},
        "kind": "bad",
        "color": "#c9d44a",
        "advice": {
            "en": [
                "There is no cure for a virus. Uproot and destroy infected plants so they do not infect others.",
                "The virus is spread by whiteflies. Use yellow sticky traps and ask your agriculture officer about whitefly control, including neem-based options.",
                "Use insect-proof nets in the nursery and plant only healthy seedlings.",
                "Remove weeds around the field, because they can hide whiteflies and the virus.",
            ],
            "hi": [
                "विषाणु का कोई इलाज नहीं है। संक्रमित पौधों को उखाड़कर नष्ट करें ताकि दूसरों में न फैले।",
                "यह रोग सफेद मक्खी से फैलता है। पीले चिपचिपे जाल लगाएं और नीम आधारित उपायों सहित सफेद मक्खी नियंत्रण के बारे में कृषि अधिकारी से पूछें।",
                "नर्सरी में कीट-रोधी जाली लगाएं और केवल स्वस्थ पौध ही रोपें।",
                "खेत के आसपास के खरपतवार हटाएं, क्योंकि उनमें सफेद मक्खी और विषाणु छिपे रह सकते हैं।",
            ],
        },
    },
    "Healthy": {
        "name": {"en": "Healthy leaf", "hi": "स्वस्थ पत्ती"},
        "kind": "good",
        "color": "#4ccf7a",
        "advice": {
            "en": [
                "No disease was detected in this leaf. Keep checking the plants regularly.",
                "Keep leaves dry and water at the root.",
                "Check the weather risk tab, especially in humid weather.",
            ],
            "hi": [
                "इस पत्ती में कोई रोग नहीं मिला। पौधों की नियमित जांच करते रहें।",
                "पत्तियों को सूखा रखें और पानी जड़ में दें।",
                "खासकर नमी वाले मौसम में मौसम जोखिम टैब देखें।",
            ],
        },
    },
}
FALLBACK = {"kind": "bad", "color": "#e0a23b"}

# ---------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;700&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');
:root {
  --leaf-950:#071d11; --leaf-900:#0b2a1a; --leaf-800:#11402a;
  --lime:#c4e57e; --tomato:#d93a28; --tomato-hover:#e5483a;
  --on:#eaf6ec; --soft:#a8c5b2; --card:rgba(255,255,255,.07);
  --display:'Bricolage Grotesque','Noto Sans Devanagari','Arial Narrow',system-ui,sans-serif;
  --body:'Hanken Grotesk','Noto Sans Devanagari',system-ui,-apple-system,'Segoe UI',sans-serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: radial-gradient(58rem 36rem at 88% -6%, #1f7a45 0%, transparent 62%),
              linear-gradient(180deg, #0b2a1a, #071d11) fixed !important;
  color: var(--on);
  font-family: var(--body);
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container { max-width: 52rem; padding-top: 1.6rem; padding-bottom: 4rem; }
.stApp p, .stApp li, .stApp label, .stMarkdown { font-family: var(--body); }
.stApp, .stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p { color: var(--on); }

/* hero */
.tg-top { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:1.4rem; }
.tg-brand { display:inline-flex; align-items:center; gap:.6rem; font:800 1.3rem/1 var(--display); letter-spacing:-.02em; color:var(--on); }
.tg-dot { position:relative; width:1.7rem; height:1.7rem; border-radius:50%; background:var(--tomato); display:inline-block; }
.tg-dot::before { content:""; position:absolute; left:50%; top:-.18rem; width:.95rem; height:.5rem; margin-left:-.475rem; border-radius:.5rem .5rem 0 0; background:#2f9250; }
.tg-back { color:var(--soft) !important; text-decoration:none; font-weight:700; font-size:.95rem; }
.tg-back:hover { color:#fff !important; }
.tg-hero h1 { font:800 clamp(2.1rem,6vw,3.6rem)/1 var(--display); letter-spacing:-.03em; margin:0 0 .8rem; padding:0; color:var(--on); }
.tg-hero p.tg-sub-lead { font-size:1.1rem; color:var(--soft); max-width:36rem; margin:0 0 1.1rem; }
.tg-steps { display:flex; flex-wrap:wrap; gap:.6rem; margin:0 0 1.6rem; padding:0; list-style:none; }
.tg-steps li { display:inline-flex; align-items:center; gap:.5rem; padding:.4rem .9rem .4rem .45rem; border-radius:999px; background:var(--card); font-weight:700; color:var(--on); }
.tg-steps li b { display:grid; place-items:center; width:1.55rem; height:1.55rem; border-radius:50%; background:var(--leaf-800); color:var(--lime); font:800 .85rem/1 var(--display); }

/* language switch + tabs */
[data-testid="stRadio"] { width:fit-content; margin-left:auto; }
[data-testid="stRadio"] [role="radiogroup"] { gap:.4rem; flex-wrap:wrap; }
[data-testid="stRadio"] label { background:var(--card); padding:.25rem .85rem !important; border-radius:999px; }
[data-testid="stRadio"] label:has(input:checked) { background:var(--tomato); }
[data-testid="stRadio"] label:has(input:checked) p { color:#fff !important; }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; background:var(--card) !important; padding:.35rem !important; border-radius:999px !important; width:fit-content; }
.stTabs [data-baseweb="tab"] { border-radius:999px !important; padding:.5rem 1.3rem !important; height:auto !important; background:transparent; }
.stTabs [data-baseweb="tab"] p { color:var(--soft); font-weight:700; }
.stTabs [aria-selected="true"] { background:var(--tomato) !important; }
.stTabs [aria-selected="true"] p { color:#fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.4rem; }

/* inputs */
[data-testid="stFileUploaderDropzone"] { background:var(--card); border:2px dashed rgba(196,229,126,.5); border-radius:22px; padding:1.8rem; }
[data-testid="stFileUploaderDropzone"] * { color:var(--on) !important; }
[data-testid="stFileUploaderDropzone"] button, .stButton > button, [data-testid="stBaseButton-secondary"] {
  background:var(--tomato) !important; color:#fff !important; border:0 !important; border-radius:999px !important;
  padding:.55rem 1.4rem !important; font-weight:700 !important; }
[data-testid="stFileUploaderDropzone"] button *, .stButton > button *, [data-testid="stBaseButton-secondary"] * { color:#fff !important; }
[data-testid="stFileUploaderDropzone"] button:hover, .stButton > button:hover { background:var(--tomato-hover) !important; }
[data-testid="stExpander"] { background:var(--card); border:0 !important; border-radius:18px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:var(--on) !important; }
[data-baseweb="input"], [data-baseweb="base-input"] { background:rgba(255,255,255,.09) !important; border-radius:14px !important; border:0 !important; }
.stTextInput input { color:var(--on) !important; background:transparent !important; -webkit-text-fill-color:var(--on); }
[data-testid="stImage"] img { border-radius:20px; }
.stAlert { border-radius:16px; }

/* cards */
.tg-tip { color:var(--soft); margin:.2rem 0 1rem; }
.tg-card { background:var(--card); border-radius:22px; padding:1.2rem 1.3rem; margin:0 0 1rem; }
.tg-result { display:flex; align-items:center; gap:1.1rem; background:var(--leaf-800); background:linear-gradient(135deg, color-mix(in srgb, var(--c) 30%, #0b2a1a), #0b2a1a); }
.tg-ring { --p:0; width:5.4rem; height:5.4rem; flex:none; border-radius:50%; position:relative; display:grid; place-items:center;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.16) 0); }
.tg-ring::before { content:""; position:absolute; inset:.5rem; border-radius:50%; background:#0b2a1a; }
.tg-ring span { position:relative; font:800 1.15rem/1 var(--display); color:var(--on); font-variant-numeric:tabular-nums; }
.tg-label { font-size:.95rem; color:var(--soft); }
.tg-name { font:800 clamp(1.5rem,4.2vw,2.1rem)/1.1 var(--display); letter-spacing:-.02em; margin:.1rem 0 .2rem; color:var(--on); }
.tg-unsure .tg-name { color:#ffd48a; }
.tg-h { font:800 1.25rem/1.2 var(--display); margin:1.4rem 0 .7rem; color:var(--on); }
.tg-bars { display:grid; gap:.55rem; }
.tg-bar { display:grid; grid-template-columns:9.5rem 1fr 3.2rem; gap:.7rem; align-items:center; font-size:.92rem; color:var(--soft); }
.tg-bar i { display:block; height:10px; border-radius:6px; background:rgba(255,255,255,.14); overflow:hidden; }
.tg-bar b { display:block; height:100%; border-radius:6px; background:var(--c); }
.tg-bar em { font-style:normal; text-align:right; font-variant-numeric:tabular-nums; }
.tg-advice { list-style:none; margin:0 !important; padding:0 !important; display:grid; gap:.65rem; }
.tg-advice li { display:grid; grid-template-columns:1.5rem 1fr; gap:.75rem; background:var(--card); border-radius:14px; padding:.8rem 1rem; color:var(--on); }
.tg-advice li::before { content:"\\2713"; display:grid; place-items:center; width:1.4rem; height:1.4rem; margin-top:.1rem; border-radius:50%; background:var(--lime); color:#0b2a1a; font-weight:800; font-size:.8rem; }
.tg-fine { color:var(--soft); font-size:.88rem; margin-top:1.2rem; }
.tg-stats { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin:1rem 0; }
.tg-stat { background:var(--card); border-radius:18px; padding:1rem 1.2rem; }
.tg-stat b { display:block; font:800 2rem/1.1 var(--display); color:var(--on); font-variant-numeric:tabular-nums; }
.tg-stat span { color:var(--soft); font-size:.9rem; }
.tg-place { font:800 1.5rem/1.2 var(--display); margin:.4rem 0 0; color:var(--on); }
.tg-risk { display:flex; align-items:center; justify-content:space-between; gap:1rem; background:var(--card); border-radius:16px; padding:.9rem 1.1rem; margin-bottom:.6rem; }
.tg-risk strong { display:block; color:var(--on); }
.tg-risk small { color:var(--soft); font-size:.86rem; }
.tg-pill { white-space:nowrap; flex:none; min-width:5.4rem; text-align:center; padding:.45rem 1rem; border-radius:999px; font-weight:800; color:#fff; background:#1f7a45; }
.tg-pill.medium { background:#b26a00; }
.tg-pill.high { background:#c42e1d; }
.tg-foot { margin-top:2.4rem; color:var(--soft); font-size:.88rem; }

@media (max-width: 640px) {
  .tg-bar { grid-template-columns:7rem 1fr 2.8rem; }
  .tg-result { flex-direction:row; }
  .tg-stats { grid-template-columns:1fr 1fr; }
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
"""


def h(text):
    return html.escape(str(text), quote=True)


def md(markup):
    """Render HTML without leading spaces or blank lines (Markdown would turn them into code)."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


# ---------------------------------------------------------------- logic
def blight_risk(temp, hum):
    """Late blight likes cool and very humid weather."""
    if hum >= 85 and 10 <= temp <= 25:
        return "high"
    if hum >= 75 and 10 <= temp <= 28:
        return "medium"
    return "low"


def early_blight_risk(temp, hum):
    """Early blight likes warm and humid weather."""
    if hum >= 80 and 24 <= temp <= 30:
        return "high"
    if hum >= 70 and 20 <= temp <= 32:
        return "medium"
    return "low"


def whitefly_risk(temp, hum):
    """Whiteflies (leaf curl virus) multiply in warm, dry weather."""
    if temp >= 28 and hum <= 60:
        return "high"
    if temp >= 25 and hum <= 70:
        return "medium"
    return "low"


def predict(model, image):
    """Return a numpy array of class probabilities for a PIL image.

    The saved model already contains its own preprocessing, so it takes raw
    0-255 pixel values at 224x224, exactly like in the notebook.
    """
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE)
    batch = np.expand_dims(np.asarray(image, dtype="float32"), 0)
    return np.asarray(model.predict(batch, verbose=0)[0])


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(city):
    """Today's average temperature and humidity from Open-Meteo (free, no key)."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=10,
    ).json()
    if not geo.get("results"):
        return None
    place = geo["results"][0]
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "hourly": "temperature_2m,relative_humidity_2m",
            "forecast_days": 1, "timezone": "auto",
        }, timeout=10,
    ).json()
    hourly = wx["hourly"]
    label = ", ".join(p for p in (place.get("name"), place.get("admin1")) if p)
    return {
        "place": label,
        "temp": mean(hourly["temperature_2m"]),
        "hum": mean(hourly["relative_humidity_2m"]),
    }


# ---------------------------------------------------------------- pages
def hero(t):
    back = f'<a class="tg-back" href="{h(WEBSITE_URL)}" target="_blank" rel="noopener">{h(t["back"])}</a>' if WEBSITE_URL else ""
    steps = "".join(f"<li><b>{i}</b>{h(s)}</li>" for i, s in enumerate(t["steps"], 1))
    md(f"""
    <div class="tg-top"><span class="tg-brand"><span class="tg-dot"></span>TomatoGuard</span>{back}</div>
    <div class="tg-hero"><h1>{h(t["hero_title"])}</h1>
    <p class="tg-sub-lead">{h(t["subtitle"])}</p>
    <ol class="tg-steps">{steps}</ol></div>
    """)


def show_image(image):
    """Full-width image on both new and older Streamlit versions."""
    try:
        st.image(image, width="stretch")
    except Exception:
        st.image(image, use_container_width=True)


def diagnose_tab(t, lang):
    md(f'<p class="tg-tip">{h(t["tip"])}</p>')
    uploaded = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with st.expander(t["camera"]):
        shot = st.camera_input(t["camera"], label_visibility="collapsed")
    source = uploaded or shot
    if source is None:
        return

    image = Image.open(source)

    try:
        model, class_names = load_model()
    except Exception as e:  # missing file or TensorFlow/Keras version mismatch
        st.error(t["model_error"])
        st.caption(f"{type(e).__name__}: {e}")
        return

    with st.spinner(t["analyzing"]):
        probs = predict(model, image)

    best = int(np.argmax(probs))
    cls = class_names[best]
    conf = float(probs[best])
    info = DISEASE.get(cls, FALLBACK)
    name = info.get("name", {}).get(lang, cls)
    color = info["color"]
    pct = round(conf * 100)
    sure = conf >= MIN_CONFIDENCE

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        show_image(image)
    with right:
        if sure:
            md(f"""
            <div class="tg-card tg-result" style="--c:{color}">
            <div class="tg-ring" style="--p:{pct}"><span>{pct}%</span></div>
            <div><div class="tg-label">{h(t["result"])}</div><div class="tg-name">{h(name)}</div>
            <div class="tg-label">{h(t["confidence"])}: {conf * 100:.1f}%</div></div></div>
            """)
        else:
            md(f"""
            <div class="tg-card tg-result tg-unsure" style="--c:#e0a23b">
            <div class="tg-ring" style="--p:{pct}"><span>{pct}%</span></div>
            <div><div class="tg-label">{h(t["result"])}</div><div class="tg-name">{h(t["unsure_title"])}</div>
            <div class="tg-label">{h(t["unsure"])}</div></div></div>
            """)

    order = np.argsort(probs)[::-1]
    rows = ""
    for i in order:
        c = class_names[int(i)]
        lbl = DISEASE.get(c, {}).get("name", {}).get(lang, c)
        col = DISEASE.get(c, FALLBACK)["color"]
        p = float(probs[int(i)]) * 100
        rows += f'<div class="tg-bar" style="--c:{col}"><span>{h(lbl)}</span><i><b style="width:{p:.1f}%"></b></i><em>{p:.0f}%</em></div>'
    md(f'<div class="tg-h">{h(t["all_probs"])}</div><div class="tg-bars">{rows}</div>')

    if sure:
        items = "".join(f"<li><span>{h(a)}</span></li>" for a in info.get("advice", {}).get(lang, []))
        md(f'<div class="tg-h">{h(t["advice"])}</div><ul class="tg-advice">{items}</ul>')
    md(f'<p class="tg-fine">{h(t["disclaimer"])}</p>')


def weather_tab(t):
    md(f'<p class="tg-tip">{h(t["weather_intro"])}</p>')
    city = st.text_input(t["city"], value="Pune")
    if not st.button(t["get_weather"]):
        return
    try:
        data = get_weather(city.strip())
    except Exception:
        st.error(t["weather_error"])
        return
    if data is None:
        st.warning(t["city_not_found"])
        return

    risks = [
        ("Late_Blight", blight_risk(data["temp"], data["hum"])),
        ("Early_Blight", early_blight_risk(data["temp"], data["hum"])),
        ("Leaf_Curl_Virus", whitefly_risk(data["temp"], data["hum"])),
    ]
    rows = "".join(
        f'<div class="tg-risk"><div><strong>{h(t["risk_label"][k])}</strong><small>{h(t["risk_rule"][k])}</small></div>'
        f'<span class="tg-pill {lv}">{h(t["level"][lv])}</span></div>'
        for k, lv in risks
    )
    md(f"""
    <div class="tg-place">{h(data["place"])}</div>
    <div class="tg-stats">
    <div class="tg-stat"><b>{data["temp"]:.1f} °C</b><span>{h(t["temp"])}, {h(t["today"])}</span></div>
    <div class="tg-stat"><b>{data["hum"]:.0f}%</b><span>{h(t["hum"])}, {h(t["today"])}</span></div>
    </div>
    <div class="tg-h">{h(t["risk_for"])}</div>{rows}
    <p class="tg-fine">{h(t["risk_note"])}</p>
    """)


def main():
    st.set_page_config(page_title="TomatoGuard", page_icon="🍅", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    choice = st.radio("Language / भाषा", ["English", "हिन्दी"], horizontal=True, label_visibility="collapsed")
    lang = "en" if choice == "English" else "hi"
    t = UI[lang]

    hero(t)
    tab1, tab2 = st.tabs([t["tab_diag"], t["tab_weather"]])
    with tab1:
        diagnose_tab(t, lang)
    with tab2:
        weather_tab(t)
    md(f'<p class="tg-foot">{h(t["footer"])}</p>')


if __name__ == "__main__":
    main()
