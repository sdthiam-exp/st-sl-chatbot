import logging
from azure.ai.documentintelligence.models import AnalyzeResult
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st

from utils import get_document_intelligence_client, get_file_retriever, get_file_vector_store, get_llm

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
    rag_prompt_system = st.text_area("**RAG Prompt System**", """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}""")

    if "file_id" not in st.session_state:
        if uploaded_file := st.file_uploader("**Télécharger un fichier**", type=["pdf"]):
            st.session_state.file_id = uploaded_file.file_id
            st.session_state.file_name = uploaded_file.name
            document_intelligence_client = get_document_intelligence_client()
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", analyze_request=uploaded_file.getvalue(), content_type="application/octet-stream", output_content_format="markdown"
            )
            result: AnalyzeResult = poller.result()

            excluded_paragraph_roles = [
                "pageHeader",
                "pageFooter",
                "footnote",
                "pageNumber",
            ]

            relevant_paragraphs = []
            for paragraph in result.paragraphs:
                if "role" in paragraph.keys():
                    if paragraph["role"] not in excluded_paragraph_roles:
                        relevant_paragraphs.append(paragraph)
                else:
                    relevant_paragraphs.append(paragraph)

            paragraphs_by_page = {}
            for paragraph in relevant_paragraphs:
                page_number = paragraph.bounding_regions[0]["pageNumber"]
                is_new_page = page_number not in paragraphs_by_page
                if is_new_page:
                    paragraphs_by_page[page_number] = []
                paragraphs_by_page[page_number].append(paragraph)
            
            texts = []
            metadatas = []
            for page_number, paragraphs in paragraphs_by_page.items():
                content = "\n\n".join([paragraph.content for paragraph in paragraphs])
                texts.append(content)
                metadatas.append({"parent_id": uploaded_file.file_id, "page_number": page_number, "file_name": uploaded_file.name})

            vector_store = get_file_vector_store("sl-index")
            vector_store.add_texts(texts, metadatas)
    else:
        st.info(f"Vous travaillez avec le fichier {st.session_state.file_name}")

    if st.button("Clear Chat History", icon="🧹"):
        st.session_state.rag_messages = []

# Azure OpenAI
llm = get_llm(deployment, model_version, temperature)

# Azure AI Search
retriever = get_file_retriever("sl-index")

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
if prompt := st.chat_input("Message MC", disabled="file_id" not in st.session_state):
    with get_openai_callback() as cb:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        docs = retriever.invoke(prompt, filters=f"parent_id eq '{st.session_state.file_id}'")
        context = "\n\n".join(doc.page_content for doc in docs)
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
    