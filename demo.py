import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


# UI
st.title("💬 Chatbot Qwen2.5 - 1.5B 8-bit quantization 💬")
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
