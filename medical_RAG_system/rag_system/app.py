import streamlit as st
from med_rag import MedRAG

# --- Khởi tạo mô hình ---
rag = MedRAG(retriever=1, question_type=1)

# --- Cấu hình giao diện ---
st.set_page_config(page_title="MedRAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 MedRAG Medical Chatbot")
st.write("Chatbot dùng mô hình RAG để trả lời các câu hỏi về chấn thương trong thể thao 🧠")

# --- Lưu lịch sử hội thoại ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hiển thị hội thoại ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Nhập câu hỏi mới ---
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Hiển thị câu hỏi người dùng
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Gọi pipeline RAG để lấy câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ... 🤔"):
            answer = rag.get_answer(prompt)
            st.markdown(answer)

    # Lưu phản hồi vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": answer})

