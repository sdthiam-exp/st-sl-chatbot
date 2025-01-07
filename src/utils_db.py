from azure.cosmos import CosmosClient, PartitionKey
import streamlit as st


# Initialize the client
client = CosmosClient(url = st.secrets.azure_cosmosdb.endpoint, credential = st.secrets.azure_cosmosdb.key)

# Give access to the database and containers
database = client.get_database_client(st.secrets.azure_cosmosdb.database)
sessions_container = database.get_container_client(st.secrets.azure_cosmosdb.sessions_container)
programs_container = database.get_container_client(st.secrets.azure_cosmosdb.programs_container)

@st.cache_data
def get_training_sessions(training_id):
    """
    Retrieve training sessions with the given training id.
    """
    query = "SELECT * FROM c WHERE c.training_id = @training_id"
    parameters = [{"name": "@training_id", "value": training_id}]
        
    items = list(sessions_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
    if not items:
        return None
    return items

@st.cache_data
def get_training_programs(query):
    """ 
    Retrieve a list of trainings based on the given query.
    """
    items = list(programs_container.query_items(query=query, enable_cross_partition_query=True))

    if not items:
        return None
    return items


#for later
def add_training_session(session_data: dict):
    """
    Add a new training session to the database or updated an existing one.
    """
    sessions_container.upsert_item(session_data)
    st.cache_data.clear()

#for later
def add_training_program(program_data: dict):
    """
    Add a new training to the data or updated an existing one.
    """
    programs_container.upsert_item(program_data)
    st.cache_data.clear()