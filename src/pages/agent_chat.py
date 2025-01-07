import logging
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_verbose, set_debug
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import streamlit as st

from tools import CseSearchTool, ExcelSearchTool, WordSearchTool
from utils import get_llm

logger = logging.getLogger(__name__)
set_debug(True)

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

# Tools
word_tool = WordSearchTool(llm=llm)
excel_tool = ExcelSearchTool(llm=llm)
cse_tool = CseSearchTool()
tools = [word_tool, excel_tool, cse_tool]
llm_with_tools = llm.bind_tools(tools)

# Graph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Graph nodes
def chatbot(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

# Graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.set_entry_point("chatbot")
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
        with st.spinner('Thinking...'):
            result_state = graph.invoke({"messages": {"role": "user", "content": prompt}})
            st.info(result_state)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(result_state["messages"][-1].content)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result_state["messages"][-1].content})

        # Tokens
        st.info(f"{cb.total_tokens} tokens used.")