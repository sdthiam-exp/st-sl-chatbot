import logging
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.globals import set_verbose, set_debug
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import streamlit as st

from utils import get_llm, get_retriever

logger = logging.getLogger(__name__)
set_debug(True)

# App title
st.set_page_config(page_title="💬 RAG Chatbot Streamlit MC", page_icon="💬")
st.title("💬 RAG Chatbot Streamlit MC")

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
grader_llm = get_llm(deployment, model_version, 0)

# Azure AI Search
retriever = get_retriever(azure_search_index)

# Retrieval Grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
structured_llm_document_grader = grader_llm.with_structured_output(GradeDocuments)

retrieval_grader_prompt_system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_prompt_system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = retrieval_grader_prompt | structured_llm_document_grader

# Hallucination Grader
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
# LLM with function call
structured_llm_hallucination_grader = grader_llm.with_structured_output(GradeHallucinations)

# Prompt
hallucination_grader_prompt_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_prompt_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_grader_prompt | structured_llm_hallucination_grader

## Question Re-writer
# Prompt
re_write_prompt_system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", re_write_prompt_system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | grader_llm | StrOutputParser()

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

        # Retrieve
        docs = retriever.invoke(prompt)

        # Grade documents
        filtered_docs = []
        for doc in docs:
            doc_score = retrieval_grader.invoke(
                {"question": prompt, "document": doc.page_content}
            )
            doc_grade = doc_score.binary_score
            if doc_grade == "yes":
                filtered_docs.append(doc)
        
        if not filtered_docs:
            rewritten_prompt = question_rewriter.invoke({"question": prompt})
            docs = retriever.invoke(prompt)
            for doc in docs:
                doc_score = retrieval_grader.invoke(
                    {"question": prompt, "document": doc.page_content}
                )
                doc_grade = doc_score.binary_score
                if doc_grade == "yes":
                    filtered_docs.append(doc)

        # Generate
        context = "\n\n".join(doc.page_content for doc in filtered_docs)
        response = chain.invoke({"question": prompt, "context": context, "chat_history": st.session_state.rag_messages})

        # Grade generation
        generation_score = hallucination_grader.invoke(
            {"documents": filtered_docs, "generation": response}
        )
        generation_grade = generation_score.binary_score

        # Add user message to chat history
        st.session_state.rag_messages.append({"role": "user", "content": prompt})

        if generation_grade == "yes":
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
        else:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(f"**hallucination** {response}")
            
            # Add assistant response to chat history
            st.session_state.rag_messages.append({"role": "assistant", "content": f"**hallucination** {response}"})

        # Tokens
        st.info(f"{cb.total_tokens} tokens used.")
