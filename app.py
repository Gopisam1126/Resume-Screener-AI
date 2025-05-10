import streamlit as st
import joblib
import docx
import PyPDF2
import re

pipeline = joblib.load("re_sc_pipeline.pkl")
svc_model = pipeline["model"]
tfidf = pipeline["vectorizer"]
le = pipeline["label_encoder"]

def cleanResume(txt: str) -> str:
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_txt(file) -> str:
    raw = file.read()
    try:
        return raw.decode("utf-8")
    except:
        return raw.decode("latin-1")

def handle_file_upload(uploaded_file) -> str:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == "docx":
        return extract_text_from_docx(uploaded_file)
    elif ext == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")

def pred_category(resume_text: str) -> str:
    cleaned = cleanResume(resume_text)
    vec = tfidf.transform([cleaned])
    pred = svc_model.predict(vec)
    return le.inverse_transform(pred)[0]

# 5. Streamlit UI
def main():
    st.set_page_config(
        page_title="Resume Category Prediction",
        page_icon="📄",
        layout="wide"
    )
    st.title("📄 Resume Category Prediction")
    st.write("Upload a resume (PDF, DOCX, or TXT) to see its predicted category.")

    uploaded_file = st.file_uploader("Choose your resume", type=["pdf", "docx", "txt"])
    if not uploaded_file:
        return

    try:
        text = handle_file_upload(uploaded_file)
        st.success("✅ Text extraction successful!")
    except Exception as e:
        st.error(f"❌ Failed to extract text: {e}")
        return

    if st.checkbox("Show extracted text"):
        st.text_area("Resume Text", text, height=300)

    if st.button("Predict Category"):
        with st.spinner("Analyzing..."):
            try:
                category = pred_category(text)
                st.subheader("Predicted Category")
                st.success(f"**{category}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
