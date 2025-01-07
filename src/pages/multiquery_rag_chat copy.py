import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.globals import set_verbose, set_debug
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st

from utils import get_llm, get_retriever

logger = logging.getLogger(__name__)
set_debug(True)

# App title
st.set_page_config(page_title="💬 Multi query RAG Chatbot Streamlit MC", page_icon="💬")
st.title("💬 Multi query RAG Chatbot Streamlit MC")

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
    rag_prompt_system = st.text_area("**RAG Prompt System**", """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}""")
    azure_search_index = st.selectbox("**Azure Search Index**", ["cse-index"], index=0)

    if st.button("Clear Chat History", icon="🧹"):
        st.session_state.rag_messages = []

# Azure OpenAI
llm = get_llm(deployment, model_version, temperature)

# Azure AI Search
retriever = get_retriever(azure_search_index)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)

prompt = ChatPromptTemplate.from_messages(
[
    ("system", rag_prompt_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

# Initialize chat history
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

# Display chat messages from history on app rerun
for message in st.session_state.rag_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Message MC"):
    with get_openai_callback() as cb:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        docs = retriever.invoke(prompt)
        unique_docs = retriever_from_llm.invoke(prompt)
        st.write(docs)
        st.write(unique_docs)
        context = "\n\n".join(doc.page_content for doc in unique_docs)
        response = chain.invoke({"question": prompt, "context": context, "chat_history": st.session_state.rag_messages})

        # Add user message to chat history
        st.session_state.rag_messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.rag_messages.append({"role": "assistant", "content": response})

        # Tokens
        st.info(f"{cb.total_tokens} tokens used.")