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
from utils_db import get_training_programs, get_training_sessions

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

# Azure OpenAI
llm = get_llm(deployment, model_version, temperature)

# Tools
@tool
def training_search(query: str):
    """
    A search engine optimized for finding training programs in the transport and logistics professions.
    Generates Cosmos DB queries based on user message analysis.
    """
       
    class SearchCriteria(BaseModel):
        """Structured search criteria extracted from user query."""
        title_terms: list[str] = Field(
            description="Terms that should be searched in the title field",
            default_factory=list
        )
        level_terms: list[str] = Field(
            description="Terms that indicate the training level",
            default_factory=list
        )
        type_terms: list[str] = Field(
            description="Terms that indicate the training type",
            default_factory=list
        )
        description_terms: list[str] = Field(
            description="Additional terms to search in description",
            default_factory=list
        )

        def add_conditions(self):
            conditions = []

            # Add title conditions
            if self.title_terms:
                title_conditions = []
                for term in self.title_terms:
                    words = term.split()  # Split the term into individual words
                    word_conditions = [f"CONTAINS(LOWER(c.title), LOWER('{word}'))" for word in words]
                    title_conditions.append(f"({' AND '.join(word_conditions)})")  # Combine words with AND
                conditions.extend(title_conditions)

            # Add level conditions
            if self.level_terms:
                level_conditions = []
                for term in search_criteria.level_terms:
                    words = term.split()
                    word_conditions = [f"CONTAINS(LOWER(c.level), LOWER('{word}'))" for word in words]
                    level_conditions.append(f"({' AND '.join(word_conditions)})")
                conditions.extend(level_conditions)
            
            # Add type conditions
            if self.type_terms:
                type_conditions = []
                for term in self.type_terms:
                    words = term.split()
                    word_conditions = [f"CONTAINS(LOWER(c.type), LOWER('{word}'))" for word in words]
                    type_conditions.append(f"({' AND '.join(word_conditions)})")
                conditions.extend(type_conditions)

                # Add description conditions
            if self.description_terms:
                desc_conditions = []
                for term in self.description_terms:
                    words = term.split()
                    word_conditions = [f"CONTAINS(LOWER(c.description), LOWER('{word}'))" for word in words]
                    desc_conditions.append(f"({' AND '.join(word_conditions)})")
                conditions.extend(desc_conditions)
            return conditions
             

    structured_llm_criteria = llm.with_structured_output(SearchCriteria)
    criteria_extractor_system = """Analyze the user's search request and categorize search terms by field:
        - title_terms: Words that describe the training name/title (e.g., "supply chain")
        - level_terms: Words indicating level (e.g., "débutant", "avancé", "avant bac", "expert")
        - type_terms: Words indicating training type (e.g., "formation courte", "certification")
        - description_terms: Additional relevant search terms

        Example:
        For "Je cherche une formation cariste pour débutant":
        - title_terms should include ["formation cariste"]
        - level_terms should include ["débutant"]
        
        Separate concepts appropriately - don't mix level terms into title terms."""
    
    criteria_extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", criteria_extractor_system),
        ("human", "Extract search criteria from this request: {query}")
    ])
    
    # Extract structured search criteria
    criteria_chain = criteria_extractor_prompt | structured_llm_criteria
    search_criteria = criteria_chain.invoke({"query": query})
    
    # Build Cosmos DB query conditions
    conditions = []
    
    # Add title conditions
    conditions.extend(search_criteria.add_conditions())
    
    # Combine all conditions
    if conditions:
        final_query = f"SELECT * FROM c WHERE {' OR '.join(conditions)}"
    else:
        final_query = "SELECT * FROM c"
    
    # Execute the query
    return get_training_programs(final_query)

@tool
def training_grader(query: str, training_title: str, training_description: str):
    """An evaluator that assesses the relevance of a retrieved training course to a user’s query. It should be used after each training search to validate the relevance of the results. Return a binary score 'yes' or 'no' to indicate whether the training is relevant to the query."""
    class GradeTrainings(BaseModel):
        """Binary score for relevance check on retrieved trainings."""

        binary_score: str = Field(
            description="Trainings are relevant to the query, 'yes' or 'no'"
        )
    structured_llm_training_grader = llm.with_structured_output(GradeTrainings)
    training_grader_prompt_system = """You are a grader assessing relevance of a retrieved training to a user query. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the training contains keyword(s) or semantic meaning related to the user query, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the training is relevant to the query."""
    training_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", training_grader_prompt_system),
            ("human", "Retrieved training: \n\n {training} \n\n User query: {query}"),
        ]
    )
    training_grader = training_grader_prompt | structured_llm_training_grader
    training_score = training_grader.invoke(
        {"query": query, "training": f"**{training_title}**\n{training_description}"}
    )
    training_grade = training_score.binary_score
    return training_grade

@tool
def training_session_search(training_id: str):
    """A search engine optimized for finding training sessions by training_id."""
    training_sessions = get_training_sessions(training_id) 
    return [session for session in training_sessions if session["training_id"] == training_id]

@tool
def training_session_register(training_id: str, session_id: str, user_name: str, user_email: str):
    """Registers the user for a training session. Returns True if the registration was successful."""
    st.toast(f"Merci {user_name}, votre demande d'inscription pour la session {session_id} a bien été reçue. Vous serez bientôt contacté par email à l'adresse {user_email}.")
    return True

tools = [training_search, training_grader, training_session_search, training_session_register]
llm_with_tools = llm.bind_tools(tools)

# Graph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Graph nodes
def assistant(state: State):
    assistant_prompt_system = """You are a specialized assistant designed to help users find training programs in the fields of transportation and logistics.
You should only offer training that is relevant to the user’s request.
If you don't have any relevant training to offer, ask for details.
Once the user confirms that the correct training has been found, you will offer to find a session for this training to register for."""
    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", assistant_prompt_system),
            ("placeholder", "{messages}"),
        ]
    )
    assistant_chain = assistant_prompt | llm_with_tools
    return {"messages": assistant_chain.invoke({"messages": state["messages"]})}

@st.cache_resource
def get_graph_memory():
    return MemorySaver()


# Graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", assistant)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=get_graph_memory())

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
            config = {"configurable": {"thread_id": "1"}}
            result_state = graph.invoke({"messages": {"role": "user", "content": prompt}}, config)
            # st.info(result_state)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(result_state["messages"][-1].content)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result_state["messages"][-1].content})

        # Tokens
        st.info(f"{cb.total_tokens} tokens used.")