from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import SearchableField, SearchField, SearchFieldDataType, SimpleField
from langchain_community.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import streamlit as st

# Azure OpenAI
@st.cache_resource
def get_llm(deployment, model_version, temperature) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=st.secrets.azure_openai.endpoint,
        openai_api_key=st.secrets.azure_openai.api_key,
        openai_api_version=st.secrets.azure_openai.api_version,
        azure_deployment=deployment,
        model_version=model_version,
        temperature=temperature
    )

@st.cache_resource
def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=st.secrets.azure_openai.endpoint,
        openai_api_key=st.secrets.azure_openai.api_key,
        openai_api_version=st.secrets.azure_openai.api_version,
        azure_deployment=st.secrets.azure_openai.embedding_deployment,
    )

# Azure AI Search
@st.cache_resource
def get_vector_store(azure_search_index) -> AzureSearch:
    return AzureSearch(
        azure_search_endpoint=st.secrets.azure_search.endpoint,
        azure_search_key=st.secrets.azure_search.key,
        index_name=azure_search_index,
        embedding_function=get_embeddings().embed_query
    )

@st.cache_resource
def get_retriever(azure_search_index, k=3) -> AzureSearchVectorStoreRetriever:
    return AzureSearchVectorStoreRetriever(vectorstore=get_vector_store(azure_search_index), k=k)

def get_file_vector_store(azure_search_index) -> AzureSearch:
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile",
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
        ),
        SearchableField(
            name="parent_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
        ),
        SearchableField(
            name="file_name",
            type=SearchFieldDataType.String,
        ),
    ]
    return AzureSearch(
        azure_search_endpoint=st.secrets.azure_search.endpoint,
        azure_search_key=st.secrets.azure_search.key,
        index_name=azure_search_index,
        embedding_function=get_embeddings().embed_query,
        fields=fields
    )

@st.cache_resource
def get_file_retriever(azure_search_index, k=3) -> AzureSearchVectorStoreRetriever:
    return AzureSearchVectorStoreRetriever(vectorstore=get_file_vector_store(azure_search_index), k=k)

# Azure document intelligence
@st.cache_resource
def get_document_intelligence_client() -> DocumentIntelligenceClient:
    document_intelligence_credential = AzureKeyCredential(st.secrets.azure_document_intelligence.key)
    return DocumentIntelligenceClient(st.secrets.azure_document_intelligence.endpoint, document_intelligence_credential)