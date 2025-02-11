from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI 

PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

--------

Answer the question based on the following query:

{question}
"""

def generate_response_using_llm(question, context, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    llm = ChatOpenAI(model_name="gpt-4")
    response = llm.predict(prompt)
    
    return response
