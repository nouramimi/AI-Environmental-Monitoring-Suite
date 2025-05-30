import streamlit as st
from rag3 import (extract_pdf_text, chunk_text, build_embeddings, store_embeddings_in_faiss, build_and_store_index, build_prompt_with_context, ollama)
import os

st.set_page_config(page_title="AI Chat Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to style the chat
def load_css():
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                color: #333;  /* Text Color */
                font-family: 'Arial', sans-serif;
            }
            .stTextInput > div > div > input {
                color: #666;
            }
            .message {
                position: relative;
                width: fit-content;
                padding: 10px 20px;
                border-radius: 25px;
                margin-bottom: 10px;
                line-height: 24px;
            }
            .userMessage {
                background-color: #0078D4; /* Light blue background for user messages */
                color: white;
                margin-left: auto;
                margin-right: 10px;
            }
            .aiMessage {
                background-color: #f1f1f1; /* Light gray background for AI messages */
                color: black;
                margin-left: 10px;
                margin-right: auto;
            }
            .systemMessage {
                background-color: #ff6347; /* Red background for system messages */
                color: white;
                margin-left: 10px;
                margin-right: auto;
            }
            .chat-container {
                overflow-y: auto;
                padding: 10px;
                max-width: 800px;
                margin: auto;
                background: white;
            }
        </style>
        """, unsafe_allow_html=True)

def main():
    load_css()
    st.title('AI Chat Assistant')
    PDF_PATH = "100-ways-you-can-improve-the-environment.pdf"

    if not os.path.exists(PDF_PATH):
        st.error(f"Error: The mandatory PDF file '{PDF_PATH}' was not found. Please check the file path.")
        return

    index, chunks = build_and_store_index(PDF_PATH)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for image upload
    with st.sidebar:
        st.header("Upload Image for Context")
        image_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
        image_path = None
        if image_file is not None:
            image_path = f"temp_image.{image_file.name.split('.')[-1]}"
            with open(image_path, "wb") as f:
                f.write(image_file.getvalue())
            st.image(image_path, caption='Uploaded Context Image', use_column_width=True)

    # Main chat interface
    with st.form("chat_form", clear_on_submit=True):
        st.text_input("Enter your query here:", key="query_input")
        submitted = st.form_submit_button("Send")

    if submitted:
        query = st.session_state.query_input
        if query:
            user_message = {'author': 'User', 'message': query}
            st.session_state.chat_history.append(user_message)
            prompt = build_prompt_with_context(query, index, chunks, image_path)

            try:
                response = ollama.chat(**prompt)
                ai_message = {'author': 'AI', 'message': response['message']['content']}
                st.session_state.chat_history.append(ai_message)
            except Exception as e:
                error_message = {'author': 'System', 'message': f"Error: {e}"}
                st.session_state.chat_history.append(error_message)

    # Display chat history
    with st.container():
        for chat in st.session_state.chat_history:
            if chat['author'] == 'User':
                st.markdown(f"<div class='message userMessage'>{chat['message']}</div>", unsafe_allow_html=True)
            elif chat['author'] == 'AI':
                st.markdown(f"<div class='message aiMessage'>{chat['message']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='message systemMessage'>{chat['message']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
