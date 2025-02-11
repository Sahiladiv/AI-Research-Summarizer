import streamlit as st
from create_database import CreateChromaDatabase
from query_data import query_the_data
from generate_response import generate_response_using_llm

st.title("AI-Powered Research Summarizer")

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chroma_object = CreateChromaDatabase(openai_api_key)
chroma_object.generate_data_stores()

user_query = st.text_input("Enter your query:")
query_text, context_text = query_the_data(user_query, openai_api_key)
    
if user_query:
    response = generate_response_using_llm(query_text, context_text, openai_api_key)
    st.write("### Response:")
    st.write(response)