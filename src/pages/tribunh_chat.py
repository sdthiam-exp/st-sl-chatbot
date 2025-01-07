import logging
from typing import Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_verbose, set_debug
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st

from tools import CseSearchTool, ExcelSearchTool, WordSearchTool
from utils import get_llm, get_retriever

# App title
st.set_page_config(page_title="💬 Tribun Health RFP MC", page_icon="💬")
st.title("💬 Tribun Health RFP MC")

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

# Azure OpenAI
llm = get_llm(deployment, model_version, temperature)

# Tools
def qna_search(query: str):
    """A search engine optimized to find similar questions and the answers given to them based on a question."""
    # Azure AI Search
    retriever = get_retriever("tribunh-rfp-index")
    docs = retriever.invoke(query)
    qna = [{"question" : doc.page_content, "answer": doc.metadata["answer"], "source": doc.metadata["source"], "sheet": doc.metadata["sheet"]} for doc in docs]
    return qna

def qna_grader(query: str, question: str):
    """An evaluator that assesses the relevance of a retrieved question to a user’s query. It should be used after each qna search to validate the relevance of the results. Return a binary score 'yes' or 'no' to indicate whether the training is relevant to the query."""
    class GradeTrainings(BaseModel):
        """Binary score for relevance check on retrieved trainings."""

        binary_score: str = Field(
            description="Trainings are relevant to the query, 'yes' or 'no'"
        )
    structured_llm_training_grader = llm.with_structured_output(GradeTrainings)
    qna_grader_prompt_system = """You are a grader assessing relevance of a retrieved question to a user query. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the training contains keyword(s) or semantic meaning related to the user query, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the training is relevant to the query."""
    qna_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qna_grader_prompt_system),
            ("human", "Retrieved question: \n\n {question} \n\n User query: {query}"),
        ]
    )
    qna_grader = qna_grader_prompt | structured_llm_training_grader
    qna_score = qna_grader.invoke(
        {"query": query, "question": question}
    )
    qna_grade = qna_score.binary_score
    print (f"{question}: {qna_grade}")
    return qna_grade

def knowledge_search(query: str):
    """A search engine optimized to find knowledge."""
    # Azure AI Search
    retriever = get_retriever("tribunh-knowledge-index")
    docs = retriever.invoke(query)
    return [{"content": doc.page_content, "source": doc.metadata["source"], "page_number": doc.metadata["page_number"]} for doc in docs]

def knowledge_grader(query: str, knowledge: str):
    """An evaluator that assesses the relevance of a retrieved knowledge to a user’s query. It should be used after each knowledge search to validate the relevance of the results. Return a binary score 'yes' or 'no' to indicate whether the knowledge is relevant to the query."""
    class GradeTrainings(BaseModel):
        """Binary score for relevance check on retrieved knowledge."""

        binary_score: str = Field(
            description="Knowledge are relevant to the query, 'yes' or 'no'"
        )
    structured_llm_training_grader = llm.with_structured_output(GradeTrainings)
    training_grader_prompt_system = """You are a grader assessing relevance of a retrieved knowledge to a user query. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the training contains keyword(s) or semantic meaning related to the user query, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the training is relevant to the query."""
    training_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", training_grader_prompt_system),
            ("human", "Retrieved knowledge: \n\n {knowledge} \n\n User query: {query}"),
        ]
    )
    training_grader = training_grader_prompt | structured_llm_training_grader
    training_score = training_grader.invoke(
        {"query": query, "knowledge": knowledge}
    )
    training_grade = training_score.binary_score
    return training_grade

# Graph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    qna: list[dict]
    filtered_qna: list[dict]
    knowledge: list[dict]
    filtered_knowledge: list[dict]

# Graph nodes
def qna_retriever(state: State):
    message = state["messages"][-1]

    qna = qna_search(message.content)

    filtered_qna = []
    for item in qna:
        grade = qna_grader(message.content, item["question"])
        if grade == "yes":
            filtered_qna.append(item)
    
    return {"qna": qna, "filtered_qna": filtered_qna}

def knowledge_retriever(state: State):
    message = state["messages"][-1]

    knowledge = knowledge_search(message.content)

    filtered_knowledge = []
    for item in knowledge:
        grade = knowledge_grader(message.content, item["content"])
        if grade == "yes":
            filtered_knowledge.append(item)
    
    return {"knowledge": knowledge, "filtered_knowledge": filtered_knowledge}

def assistant_qna(state: State):
    message = state["messages"][-1]
    assistant_prompt_system = """You are a specialized assistant intended to help users answer questions asked during calls for tender.
Using only the answers given to similar questions provided and their response, propose an answer to the user's question.

# Similar questions and their answer
{qna}
"""
    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", assistant_prompt_system),
            ("human", "{message}"),
        ]
    )
    assistant_chain = assistant_prompt | llm
    return {"messages": assistant_chain.invoke({"message": message.content, "qna": state["filtered_qna"]})}

def assistant_knowledge(state: State):
    message = state["messages"][-1]
    assistant_prompt_system = """You are a specialized assistant intended to help users answer questions asked during calls for tender.
Using only the knowledge provided, propose an answer to the user's question.
If you can't provide an answer, just say you can't answer.

# Knowledge
{knowledge}
"""
    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", assistant_prompt_system),
            ("human", "{message}"),
        ]
    )
    assistant_chain = assistant_prompt | llm
    return {"messages": assistant_chain.invoke({"message": message.content, "knowledge": state["filtered_knowledge"]})}

# Graph edges
def qna_condition(state: State):
    if state["filtered_qna"]:
        return "assistant_qna"
    else:
        return "knowledge_retriever"
    

@st.cache_resource
def get_graph_memory():
    return MemorySaver()


# Graph
graph_builder = StateGraph(State)

graph_builder.add_node("qna_retriever", qna_retriever)
graph_builder.add_node("assistant_qna", assistant_qna)
graph_builder.add_node("knowledge_retriever", knowledge_retriever)
graph_builder.add_node("assistant_knowledge", assistant_knowledge)

graph_builder.add_conditional_edges("qna_retriever", qna_condition)
graph_builder.add_edge("assistant_qna", "__end__")
graph_builder.add_edge("knowledge_retriever", "assistant_knowledge")
graph_builder.add_edge("assistant_knowledge", "__end__")

# graph_builder.add_node("chatbot", assistant)
# graph_builder.add_node("tools", ToolNode(tools=tools))

# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_conditional_edges("chatbot", tools_condition)

# graph_builder.set_entry_point("chatbot")
# graph = graph_builder.compile(checkpointer=get_graph_memory())

graph_builder.set_entry_point("qna_retriever")
graph = graph_builder.compile()

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

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get assistant response
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": "1"}}
            result_state = graph.invoke({"messages": {"role": "user", "content": prompt}}, config)
            # st.info(result_state)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(result_state["messages"][-1].content)

        if "filtered_qna" in result_state and len(result_state["filtered_qna"]) > 0:
            st.markdown("**Questions and answers used to build the response**")
            st.table(result_state["filtered_qna"])

        if "filtered_knowledge" in result_state and len(result_state["filtered_knowledge"]) > 0:
            st.markdown("**Knowledge used to build the response**")
            st.table(result_state["filtered_knowledge"])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result_state["messages"][-1].content})

        # Tokens
        st.info(f"{cb.total_tokens} tokens used.")