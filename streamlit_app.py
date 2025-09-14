import streamlit as st
import json
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
import faiss

st.title("🎓 College Multilingual Chatbot")

# Load FAQs
with open("faq_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions_en = [item["question_en"] for item in faq_data]
answers_en = [item["answer_en"] for item in faq_data]

# Build FAISS index
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model_embed.encode(questions_en)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# MarianMT Models
LANG_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "ta": "Helsinki-NLP/opus-mt-ta-en",
    "te": "Helsinki-NLP/opus-mt-te-en",
    "kn": "Helsinki-NLP/opus-mt-kn-en",
    "ml": "Helsinki-NLP/opus-mt-ml-en",
    "bn": "Helsinki-NLP/opus-mt-bn-en",
    "mr": "Helsinki-NLP/opus-mt-mr-en",
    "gu": "Helsinki-NLP/opus-mt-gu-en",
    "pa": "Helsinki-NLP/opus-mt-pa-en",
    "or": "Helsinki-NLP/opus-mt-or-en",
    "as": "Helsinki-NLP/opus-mt-as-en",
    "ur": "Helsinki-NLP/opus-mt-ur-en"
}

marian_models = {}
for lang, model_name in LANG_MODELS.items():
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    marian_models[lang] = (tokenizer, model)

def translate_to_en(text, lang):
    if lang in marian_models:
        tokenizer, model = marian_models[lang]
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    return text

def translate_from_en(text, lang):
    rev_model_name = f"Helsinki-NLP/opus-mt-en-{lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(rev_model_name)
        model = MarianMTModel.from_pretrained(rev_model_name)
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except:
        return text

# Chat session
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask your question in any Indian language:")

if st.button("Send") and user_input:
    lang = detect(user_input)
    user_en = translate_to_en(user_input, lang)
    query_vec = model_embed.encode([user_en])
    D, I = index.search(query_vec, k=1)
    answer_en = answers_en[I[0][0]]
    answer_lang = translate_from_en(answer_en, lang)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", answer_lang))

for role, msg in st.session_state.chat:
    st.write(f"**{role}:** {msg}")