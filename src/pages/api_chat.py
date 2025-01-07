import logging
import streamlit as st
from msal_streamlit_authentication import msal_authentication
from fast_api_client import AuthenticatedClient
from fast_api_client.models import Conversation
from fast_api_client.api.conversations import get_conversations_api_conversations_get, chat_api_conversations_conversation_id_chat_post, add_conversation_api_conversations_post, delete_conversation_api_conversations_conversation_id_delete
from fast_api_client.models.conversation_chat_request import ConversationChatRequest
from fast_api_client.models.conversation_request import ConversationRequest

logger = logging.getLogger(__name__)

def get_chat_client():
    return AuthenticatedClient(base_url=st.secrets.api.endpoint, token=st.session_state.login_token["accessToken"])

def select_conversation(conversation: Conversation):
    st.session_state.selected_conversation = conversation

def is_conversation_selected(conversation: Conversation):
    if "selected_conversation" in st.session_state.keys():
        return conversation.id == st.session_state.selected_conversation.id
    return False

def add_conversation(conversation_name: str):
    st.session_state.conversation_name = conversation_name

def delete_conversation(conversation: Conversation):
    st.session_state.delete_conversation = conversation

# App title
st.set_page_config(page_title="🤗💬 Chatbot Streamlit MC")
st.title("🤗💬 Chatbot Streamlit MC")

with st.sidebar:
    st.header("Login to chatbot API")
    login_token = msal_authentication(
        auth={
            "clientId": st.secrets.azure_ad.client_id,
            "authority": f"https://login.microsoftonline.com/{st.secrets.azure_ad.tenant_id}",
            "redirectUri": "/auth",
            "postLogoutRedirectUri": "/auth"
        }, # Corresponds to the 'auth' configuration for an MSAL Instance
        cache={
            "cacheLocation": "sessionStorage",
            "storeAuthStateInCookie": False
        }, # Corresponds to the 'cache' configuration for an MSAL Instance
        login_request={
            "scopes": [st.secrets.api.scope]
        }, # Optional
        logout_request={}, # Optional
        login_button_text="Login", # Optional, defaults to "Login"
        logout_button_text="Logout", # Optional, defaults to "Logout"
        class_name="css_button_class_selector", # Optional, defaults to None. Corresponds to HTML class.
        html_id="html_id_for_button", # Optional, defaults to None. Corresponds to HTML id.
        #key="1" # Optional if only a single instance is needed
    )
    st.session_state.login_token = login_token

if st.session_state.login_token is not None:
    with st.sidebar:
        if "conversation_name" in st.session_state.keys():
            client = get_chat_client()
            with client as client:
                request = ConversationRequest(name=st.session_state.conversation_name)
                with st.spinner("Creating new conversation..."):
                    conversation = add_conversation_api_conversations_post.sync(client=client, body=request)
                st.session_state.conversations.append(conversation)
                st.session_state.selected_conversation = conversation
                del st.session_state.conversation_name

        if "delete_conversation" in st.session_state.keys():
            client = get_chat_client()
            with client as client:
                with st.spinner("Deleting conversation..."):
                    delete_conversation_api_conversations_conversation_id_delete.sync(client=client, conversation_id=st.session_state.delete_conversation.id)
                st.session_state.conversations = [c for c in st.session_state.conversations if c.id != st.session_state.delete_conversation.id]
                if st.session_state.selected_conversation.id == st.session_state.delete_conversation.id:
                    del st.session_state.selected_conversation
                del st.session_state.delete_conversation

        if "conversations" not in st.session_state.keys():
            with st.sidebar:
                client = get_chat_client()
                with client as client:
                    with st.spinner("Loading conversations..."):
                        st.session_state.conversations = get_conversations_api_conversations_get.sync(client=client)

        st.header("Conversations")
        for conversation in st.session_state.conversations:
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.button(
                    conversation.name,
                    key=conversation.id,
                    on_click=select_conversation,
                    args=(conversation,),
                    type="primary" if is_conversation_selected(conversation) else "secondary",
                    use_container_width=True)
            with col2:
                st.button("🗑️", key=f"{conversation}_delete", on_click=delete_conversation, args=(conversation,))
        
        st.header("Add conversation")
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            conversation_name = st.text_input("Conversation name", placeholder="Conversation name", label_visibility="collapsed")
        with col2:
            st.button("Add", disabled=not conversation_name, on_click=add_conversation, args=(conversation_name,))

    if "selected_conversation" in st.session_state.keys():
        for message in st.session_state.selected_conversation.messages:
            with st.chat_message(message.type):
                st.markdown(message.content)
        
        if prompt := st.chat_input("Say something"):
            with st.chat_message("human"):
                st.markdown(prompt)
            client = get_chat_client()
            with client as client:
                request = ConversationChatRequest(prompt)
                with st.spinner("Chatbot at work! Grab a virtual cup of coffee while you wait."):
                    conversation = chat_api_conversations_conversation_id_chat_post.sync(client=client, conversation_id=st.session_state.selected_conversation.id, body=request)
                st.session_state.conversations = [c if c.id != conversation.id else conversation for c in st.session_state.conversations]
                st.session_state.selected_conversation = conversation
                with st.chat_message(conversation.messages[-1].type):
                    st.markdown(conversation.messages[-1].content)