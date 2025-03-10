import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import io
import docx
import pdfplumber
import pandas as pd
import os

# Konfigurasi quantization 8-bit
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True  # Gunakan 8-bit quantization
)

@st.cache_resource
def load_model_and_tokenizer():
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,  # Tambahkan konfigurasi quantization 8-bit
        device_map="auto"
    )
    return model, tokenizer

# Load model dan tokenizer
model, tokenizer = load_model_and_tokenizer()

def extract_text_from_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

# UI
st.title(" Chatbot Qwen2.5 - 1.5B 8-bit quantization")
st.caption("Running locally using Transformers")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are AI assistant with name Louis, You are a helpful assistant. Please type a message to start the conversation and using Bahasa Indonesia."}
    ]

# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# File upload
uploaded_file = st.file_uploader("Unggah file (Word, PDF, Excel)", type=["docx", "pdf", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    file_content = uploaded_file.read()

    # Simpan file ke direktori lokal
    upload_dir = "uploads"  # Direktori untuk menyimpan file
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)

    if file_extension == "docx":
        text_content = extract_text_from_docx(io.BytesIO(file_content))
    elif file_extension == "pdf":
        text_content = extract_text_from_pdf(io.BytesIO(file_content))
    elif file_extension == "xlsx":
        text_content = extract_text_from_excel(io.BytesIO(file_content))
    else:
        text_content = ""

    st.session_state.messages.append({"role": "system", "content": f"File yang diunggah berisi teks:\n{text_content}"})

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare the prompt for the model
    messages = st.session_state.messages  # Get the entire conversation history
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    with st.spinner("Thinking..."):
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    with st.chat_message("assistant"):
        st.markdown(response)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})