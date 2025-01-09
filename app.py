import streamlit as st
from logic import count_words, preprocess_text, compute_tf, compute_idf, compute_tfidf, cosine_similarity, find_common_words, word_frequencies
from bs4 import BeautifulSoup
import requests

def main():
    st.set_page_config(page_title="Document Similarity Checker", layout="wide")
    display_ui()

def display_ui():
    st.markdown("<h1 style='text-align: center;'>Document Similarity Checker</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Upload two text documents or provide links to calculate their similarity score and explore detailed analysis.</h4>", unsafe_allow_html=True)

    # File upload section
    st.subheader("Step 1: Upload Documents or Provide URLs")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file1 = st.file_uploader("Upload Document 1 (TXT)", type=["txt"], key="file1", label_visibility="collapsed")
        url1 = st.text_input("Or paste URL for Document 1", key="url1")
    with col2:
        uploaded_file2 = st.file_uploader("Upload Document 2 (TXT)", type=["txt"], key="file2", label_visibility="collapsed")
        url2 = st.text_input("Or paste URL for Document 2", key="url2")

    # Handle text extraction from uploaded files or URLs
    text1 = None
    text2 = None

    # Extract text from uploaded files
    if uploaded_file1 is not None:
        text1 = uploaded_file1.read().decode("utf-8")
    elif url1:
        text1 = extract_text_from_url(url1)

    # Extract text from uploaded files
    if uploaded_file2 is not None:
        text2 = uploaded_file2.read().decode("utf-8")
    elif url2:
        text2 = extract_text_from_url(url2)

    # Proceed only if both text sources are available
    if text1 and text2:
        display_document_previews(text1, text2)
        display_word_count(text1, text2)
        
        # Add a spinner while processing similarity
        with st.spinner("Analyzing similarity..."):
            display_similarity_results(text1, text2)

def extract_text_from_url(url):
    """Function to extract text from a webpage using BeautifulSoup."""
    try:
        # Send a GET request to fetch the HTML content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all text from paragraph tags (<p>)
        text = ' '.join([para.get_text() for para in soup.find_all('p')])
        return text
    except Exception as e:
        st.error(f"Error fetching the URL: {e}")
        return ""

def display_document_previews(text1, text2):
    preview_length = 300
    preview_text1 = text1[:preview_length] + "..." if len(text1) > preview_length else text1
    preview_text2 = text2[:preview_length] + "..." if len(text2) > preview_length else text2

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Text 1 Preview")
        st.text_area("", preview_text1, height=150, disabled=True, key="preview1")
    with col2:
        st.subheader("Text 2 Preview")
        st.text_area("", preview_text2, height=150, disabled=True, key="preview2")

def display_word_count(text1, text2):
    total_words_text1 = count_words(text1)
    total_words_text2 = count_words(text2)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Words in Document 1", total_words_text1)
    with col2:
        st.metric("Words in Document 2", total_words_text2)

def display_similarity_results(text1, text2):
    doc1 = preprocess_text(text1)
    doc2 = preprocess_text(text2)

    tf1 = compute_tf(doc1)
    tf2 = compute_tf(doc2)
    idf = compute_idf([doc1, doc2])
    tfidf1 = compute_tfidf(tf1, idf)
    tfidf2 = compute_tfidf(tf2, idf)

    similarity = cosine_similarity(tfidf1, tfidf2)
    word_freq_doc1 = word_frequencies(doc1)
    word_freq_doc2 = word_frequencies(doc2)

    # Display Cosine Similarity Score with improved UI
    st.header("Document Similarity Analysis")
    score_color = "green" if similarity < 0.3 else "orange" if similarity < 0.7 else "red"
    st.markdown(f"<h3 style='color: {score_color}; font-size: 1.5em;'>Similarity Score: {similarity:.2%}</h3>", unsafe_allow_html=True)

    # Collapsible sections for detailed results
    with st.expander("Top 5 Keywords by Term Frequency (TF)"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Document 1:")
            top_tf1 = sorted(tf1.items(), key=lambda x: x[1], reverse=True)[:5]
            for word, freq in top_tf1:
                st.write(f"{word} ({freq:.4f})")
            
        with col2:
            st.write("Document 2:")
            top_tf2 = sorted(tf2.items(), key=lambda x: x[1], reverse=True)[:5]
            for word, freq in top_tf2:
                st.write(f"{word} ({freq:.4f})")

    with st.expander("Top 5 Keywords by Inverse Document Frequency (IDF)"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Document 1:")
            top_idf = sorted([(w, idf[w]) for w in tf1.keys()], key=lambda x: x[1], reverse=True)[:5]
            for word, score in top_idf:
                st.write(f"{word} ({score:.4f})")
            
        with col2:
            st.write("Document 2:")
            top_idf2 = sorted([(w, idf[w]) for w in tf2.keys()], key=lambda x: x[1], reverse=True)[:5]
            for word, score in top_idf2:
                st.write(f"{word} ({score:.4f})")

    with st.expander("Top 5 Common Words"):
        common_words = find_common_words(tfidf1, tfidf2)[:5]
        for word in common_words:
            freq1 = word_freq_doc1.get(word, 0)
            freq2 = word_freq_doc2.get(word, 0)
            st.write(f"{word} (Doc1: {freq1}, Doc2: {freq2})")

if __name__ == "__main__":
    main()
