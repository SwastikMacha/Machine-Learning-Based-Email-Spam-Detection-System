import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Make sure nltk finds data from streamlit cloud
nltk.data.path.append('nltk_data')

# Initialize stemmer
ps = PorterStemmer()

# -----------------------------
# Text preprocessing function
# -----------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [word for word in text if word.isalnum()]
    text = [word for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# -----------------------------
# Load vectorizer and model
# -----------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# Streamlit app layout
# -----------------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Email/SMS Spam Classifier")
st.write(
    """
    This app detects whether an email or SMS message is **Spam** or **Not Spam**.
    Enter your message below and click **Predict**.
    """
)

# Input from user
input_sms = st.text_area("Enter your message here:")

# Prediction
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.markdown("<h3 style='color: red;'>⚠️ Spam Detected!</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>✅ Not Spam</h3>", unsafe_allow_html=True)
