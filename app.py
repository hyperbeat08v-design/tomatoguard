"""TomatoGuard - tomato leaf disease detection with weather early warning.

Needs two files next to this one (both downloaded from the Colab notebook):
    tomato_model.keras
    class_names.json
"""
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

# ---------------------------------------------------------------- texts
UI = {
    "en": {
        "title": "TomatoGuard",
        "subtitle": "Tomato leaf disease detection and weather early warning",
        "language": "Language",
        "tab_diag": "Check a leaf",
        "tab_weather": "Weather risk",
        "upload": "Upload a clear photo of one tomato leaf",
        "camera": "Or take a photo with the camera",
        "analyzing": "Analyzing the leaf...",
        "result": "Result",
        "confidence": "Confidence",
        "unsure": "The model is not sure about this photo. Take a clearer, close-up photo of a single leaf in good light and try again.",
        "all_probs": "Probability for each class",
        "advice": "What to do",
        "disclaimer": "Student project. The model knows only 4 classes (3 diseases + healthy) and is trained on tomato leaves, so a non-tomato or blurry photo can give a wrong answer. Confirm with your local agriculture officer before spraying anything, and follow the product label.",
        "city": "Enter your village or city",
        "get_weather": "Check risk",
        "city_not_found": "Could not find that place. Try the nearest big town.",
        "weather_error": "Could not get weather data right now. Check the internet and try again.",
        "today": "Today's average",
        "temp": "Temperature",
        "hum": "Humidity",
        "risk_for": "Disease risk from today's weather",
        "risk_note": "This is a simple rule-based guide for learning, not a scientific forecast.",
        "level": {"low": "Low", "medium": "Medium", "high": "High"},
        "risk_label": {
            "Early_Blight": "Early Blight",
            "Late_Blight": "Late Blight",
            "Leaf_Curl_Virus": "Leaf Curl Virus (whitefly)",
        },
        "model_error": "The model could not be loaded. Make sure tomato_model.keras and class_names.json are in the same folder as app.py.",
    },
    "hi": {
        "title": "TomatoGuard",
        "subtitle": "टमाटर की पत्ती के रोग की पहचान और मौसम आधारित चेतावनी",
        "language": "भाषा",
        "tab_diag": "पत्ती जांचें",
        "tab_weather": "मौसम जोखिम",
        "upload": "टमाटर की एक पत्ती की साफ़ फोटो अपलोड करें",
        "camera": "या कैमरे से फोटो लें",
        "analyzing": "पत्ती की जांच हो रही है...",
        "result": "नतीजा",
        "confidence": "भरोसा",
        "unsure": "मॉडल इस फोटो के बारे में पक्का नहीं है। अच्छी रोशनी में एक पत्ती की पास से साफ़ फोटो लेकर फिर कोशिश करें।",
        "all_probs": "हर वर्ग की संभावना",
        "advice": "क्या करें",
        "disclaimer": "यह एक छात्र प्रोजेक्ट है। मॉडल सिर्फ 4 वर्ग (3 रोग + स्वस्थ) पहचानता है और टमाटर की पत्तियों पर सिखाया गया है, इसलिए दूसरी फसल या धुंधली फोटो पर गलत नतीजा आ सकता है। कोई भी दवा छिड़कने से पहले स्थानीय कृषि अधिकारी से सलाह लें और उत्पाद के लेबल का पालन करें।",
        "city": "अपना गांव या शहर लिखें",
        "get_weather": "जोखिम देखें",
        "city_not_found": "यह जगह नहीं मिली। नज़दीकी बड़े शहर का नाम लिखें।",
        "weather_error": "अभी मौसम की जानकारी नहीं मिल पाई। इंटरनेट जांचकर फिर कोशिश करें।",
        "today": "आज का औसत",
        "temp": "तापमान",
        "hum": "नमी",
        "risk_for": "आज के मौसम से रोग का जोखिम",
        "risk_note": "यह सीखने के लिए एक सरल नियम-आधारित अनुमान है, वैज्ञानिक पूर्वानुमान नहीं।",
        "level": {"low": "कम", "medium": "मध्यम", "high": "अधिक"},
        "risk_label": {
            "Early_Blight": "अगेती झुलसा (अर्ली ब्लाइट)",
            "Late_Blight": "पछेती झुलसा (लेट ब्लाइट)",
            "Leaf_Curl_Virus": "पत्ती मोड़क विषाणु (सफेद मक्खी)",
        },
        "model_error": "मॉडल लोड नहीं हो सका। जांचें कि tomato_model.keras और class_names.json, app.py वाले ही फोल्डर में हैं।",
    },
}

# Names and advice per class. No spray doses on purpose: those must come from
# the product label or a local agriculture officer.
DISEASE = {
    "Early_Blight": {
        "name": {"en": "Early Blight", "hi": "अगेती झुलसा (अर्ली ब्लाइट)"},
        "kind": "bad",
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


def show_level(level, text):
    if level == "high":
        st.error(text)
    elif level == "medium":
        st.warning(text)
    else:
        st.success(text)


# ---------------------------------------------------------------- pages
def diagnose_tab(t, lang):
    uploaded = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])
    with st.expander(t["camera"]):
        shot = st.camera_input(t["camera"], label_visibility="collapsed")
    source = uploaded or shot
    if source is None:
        return

    image = Image.open(source)
    st.image(image, use_container_width=True)

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
    info = DISEASE.get(cls, {"name": {"en": cls, "hi": cls}, "kind": "bad", "advice": {"en": [], "hi": []}})

    st.subheader(t["result"])
    if conf < MIN_CONFIDENCE:
        st.warning(t["unsure"])
    else:
        text = f"**{info['name'][lang]}**"
        (st.success if info["kind"] == "good" else st.error)(text)
    st.metric(t["confidence"], f"{conf * 100:.1f}%")

    st.write(f"**{t['all_probs']}**")
    labels = [DISEASE.get(c, {"name": {lang: c}})["name"][lang] for c in class_names]
    st.bar_chart({"probability": dict(zip(labels, [float(p) for p in probs]))}, horizontal=True)

    if conf >= MIN_CONFIDENCE:
        st.write(f"**{t['advice']}**")
        for line in info["advice"][lang]:
            st.markdown(f"- {line}")
    st.caption(t["disclaimer"])


def weather_tab(t):
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

    st.subheader(data["place"])
    c1, c2 = st.columns(2)
    c1.metric(f"{t['temp']} ({t['today']})", f"{data['temp']:.1f} °C")
    c2.metric(f"{t['hum']} ({t['today']})", f"{data['hum']:.0f}%")

    st.write(f"**{t['risk_for']}**")
    risks = {
        "Late_Blight": blight_risk(data["temp"], data["hum"]),
        "Early_Blight": early_blight_risk(data["temp"], data["hum"]),
        "Leaf_Curl_Virus": whitefly_risk(data["temp"], data["hum"]),
    }
    for key, level in risks.items():
        show_level(level, f"{t['risk_label'][key]}: {t['level'][level]}")
    st.caption(t["risk_note"])


def main():
    st.set_page_config(page_title="TomatoGuard", page_icon="🍅", layout="centered")
    choice = st.sidebar.radio("Language / भाषा", ["English", "हिन्दी"])
    lang = "en" if choice == "English" else "hi"
    t = UI[lang]

    st.title("🍅 " + t["title"])
    st.write(t["subtitle"])
    tab1, tab2 = st.tabs([t["tab_diag"], t["tab_weather"]])
    with tab1:
        diagnose_tab(t, lang)
    with tab2:
        weather_tab(t)


if __name__ == "__main__":
    main()
