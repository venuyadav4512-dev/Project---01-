
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    svm = joblib.load("model/spam_svm.pkl")
    tfidf = joblib.load("model/tfidf.pkl")
    return svm, tfidf


svm, tfidf = load_models()

# ============================================================
# STOPWORDS
# ============================================================

@st.cache_resource
def load_stopwords():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english")) - {
        "won", "only", "now"
    }


stop_words = load_stopwords()

# ============================================================
# REGEX PATTERNS
# ============================================================

url_pattern = re.compile(
    r"\b(?:"
    r"(?:https?|ftp)://[^\s<>'\"]+"
    r"|www\.[^\s<>'\"]+"
    r"|(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|in|co\.in|co\.uk|biz|info|xyz|ly|me)"
    r"(?:/[^\s<>'\"]*)?"
    r")",
    re.IGNORECASE
)

email_pattern = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

phone_pattern = re.compile(
    r"""
    (?:
        \+\d{1,3}[\s.-]?\d{6,14}
        |
        [6-9]\d{9}
        |
        [6-9]\d{4}[\s.-]\d{5}
        |
        0[6-9]\d{9}
    )
    """,
    re.VERBOSE
)

percentage_pattern = re.compile(
    r"\b\d+(?:\.\d+)?\s*%"
)

# ============================================================
# TEXT PREPROCESSING
# Must match train.py
# ============================================================

def clean_text(text):
    text = str(text).lower()

    text = url_pattern.sub(" link ", text)
    text = email_pattern.sub(" EMAIL ", text)
    text = phone_pattern.sub(" PHONE_NUMBER ", text)
    text = percentage_pattern.sub(" PERCENTAGE ", text)

    words = re.findall(r"\b[a-z_]+\b", text)

    words = [
        word for word in words
        if word not in stop_words
    ]

    text = " ".join(words)
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ============================================================
# PROFESSIONAL UI
# ============================================================

st.title("🛡️ SpamShield")

st.divider()

st.write("#### Analyze your message:")

message = st.text_area(
    "Message",
    placeholder="Type or paste your message here...",
    height=160,
    label_visibility="collapsed",
    key="message_input"
)

st.caption(
    "Paste a message to check whether it is HAM or SPAM."
)

analyze = st.button(
    "🔍  Analyze Message",
    type="primary",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if analyze:
    if not message.strip():
        st.warning("Please enter a message before analyzing.")

    else:
        with st.spinner("Analyzing message..."):
            cleaned_message = clean_text(message)

            message_vector = tfidf.transform(
                [cleaned_message]
            )

            prediction = int(
                svm.predict(message_vector)[0]
            )

        st.divider()
        st.subheader("Detection Result")

        if prediction == 0:
            with st.container(border=True):
                st.success("✓ HAM — LEGITIMATE MESSAGE")

                st.markdown(
                    "### 🟢 Message appears legitimate"
                )

                st.write(
                    "The message has been classified as HAM."
                )

        else:
            with st.container(border=True):
                st.error("⚠ SPAM — SUSPICIOUS MESSAGE")

                st.markdown(
                    "### 🔴 Potential spam detected"
                )

                st.write(
                    "The message has been classified as SPAM. "
                    "Be cautious with suspicious links and requests."
                )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("SpamShield · Message Detection")
