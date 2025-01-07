from typing import Optional
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils import get_retriever

class WordSearchTool(BaseTool):
    """Tool to interact with for questions regarding Word software"""

    name: str = "wordsoftwaresearch"
    description : str = "Useful when questions concern Word software."

    llm : AzureChatOpenAI

    def __init__(self, **base):
        super().__init__(**base)

    def _run(self, query: str, return_redirect = True, run_manager: Optional[CallbackManagerForToolRun] = None) -> str :
        try:
            prompt_system = """# Instruction
                - Act as a specialized user support on Word software
                - Your job is to answer users' questions about Word software."""
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_system),
                    ("human", "{question}"),
                ]
            )
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"question": query})
            return answer
        except Exception as e:
            print(e)
            return str(e)  # Return an empty string or some error indicator

class ExcelSearchTool(BaseTool):
    """Tool to interact with for questions regarding Excel software"""
    
    name: str = "excelsoftwaresearch"
    description: str  = "Useful when questions concern Excel software."

    llm: AzureChatOpenAI
    
    def __init__(self, **data):
        super().__init__(**data)
    
    def _run(self, query: str,  return_direct = False, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            prompt_system = """# Instruction
                - Act as a specialized user support on Excel software
                - Your job is to answer users' questions about Excel software."""
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_system),
                    ("human", "{question}"),
                ]
            )
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"question": query})
            return answer
        except Exception as e:
            print(e)
            return str(e)  # Return an empty string or some error indicator
        

class CseSearchTool(BaseTool):
     """Tool that queries the CSE documentation"""

     
     name: str = "cse_search"
     description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results about CSE. "
        "Useful for when you need to answer questions about CSE. "
        "Input should be a search query."
     )
    
     def __init__(self, **data):
        super().__init__(**data)
    
     def _run(self, query: str,  return_direct = False, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            # Azure AI Search
            retriever = get_retriever("cse-index")
            docs = retriever.invoke(query)
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            print(e)
            return str(e)  # Return an empty string or some error indicator