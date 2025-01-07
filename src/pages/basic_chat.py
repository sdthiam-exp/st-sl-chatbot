import logging
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st

from utils import get_llm

logger = logging.getLogger(__name__)

# App title
st.set_page_config(page_title="💬 Basic Chatbot Streamlit MC", page_icon="💬")
st.title("💬 Basic Chatbot Streamlit MC")

with st.sidebar:
    deployment = st.radio("**GPT Model**", ["gpt-35-turbo", "gpt-4o", "gpt-4o-mini"], index=2)
    match deployment:
        case "gpt-35-turbo":
            model_version = "0613"
        case "gpt-4o":
            model_version = "2024-05-13"
        case "gpt-4o-mini":
            model_version = "2024-07-18"
    temperature = st.slider("**Temperature**", 0.0, 1.0, 0.2, 0.1)
    prompt_system = st.text_area("**Prompt System**", "You are a helpful assistant. Answer all questions to the best of your ability.")

    if st.button("Clear Chat History", icon="🧹"):
        st.session_state.messages = []

# Azure OpenAI
llm = get_llm(deployment, model_version, temperature)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
chain = prompt | llm | StrOutputParser()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Message MC"):
    with get_openai_callback() as cb:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        response = chain.invoke({"question": prompt, "chat_history": st.session_state.messages})

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Tokens
        st.info(f"{cb.total_tokens} tokens used.")