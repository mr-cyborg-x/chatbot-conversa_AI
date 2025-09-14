import streamlit as st
import re, html
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer

# deterministic langdetect
DetectorFactory.seed = 0

# ---------------------------
# Static FAQ Data
# ---------------------------
RESPONSES = {
    "fees": {
        "en": "Semester fees is ₹15000. You can pay online."
    },
    "exam": {
        "en": "Semester exams will begin from December 10th."
    },
    "hostel": {
        "en": "Yes, hostel facilities are available for both boys and girls."
    }
}

# ---------------------------
# Translation Helper
# ---------------------------
def load_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate(text, src="en", tgt="ta"):
    try:
        model, tokenizer = load_model(src, tgt)
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return text  # fallback

# ---------------------------
# Utils
# ---------------------------
def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = s.lower().strip()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language_safe(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

# intent keywords
INTENT_KEYWORDS = {
    "fees": ["fee", "fees", "semester fee", "tuition", "pay"],
    "exam": ["exam", "exams", "test"],
    "hostel": ["hostel", "dorm", "accommodation"]
}

def match_intent(english_text: str) -> str:
    txt = clean_text(english_text)
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                return intent
    return None

def get_response(intent, lang):
    if not intent:
        return "Sorry, I don’t understand your question."

    english_resp = RESPONSES.get(intent, {}).get("en", "Sorry, no info available.")
    if lang == "en":
        return english_resp

    # Try to translate English → User's lang
    try:
        translated = translate(english_resp, src="en", tgt=lang)
        return translated
    except:
        return english_resp

# ---------------------------
# UI (conversa.ai style)
# ---------------------------
st.set_page_config(page_title="Conversa.AI", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #0d0d1a;
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #a855f7;
        margin-bottom: -10px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2em;
        color: #cbd5e1;
        margin-bottom: 30px;
    }
    .chat-box {
        background: linear-gradient(145deg, #1e1e2f, #2b2b40);
        padding: 20px;
        border-radius: 15px;
        width: 80%;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">conversa.ai</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Your Multilingual AI Assistant</div>', unsafe_allow_html=True)

# ---------------------------
# Chat Loop
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to conversa.ai! How can I help you today?"}]

# show history
with st.container():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lang = detect_language_safe(user_input)
    intent = match_intent(user_input if lang == "en" else user_input)

    reply_text = get_response(intent, lang)

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    with st.chat_message("assistant"):
        st.markdown(reply_text)
