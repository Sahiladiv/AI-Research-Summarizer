import argparse
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_community.vectorstores.chroma import Chroma



def query_the_data(query_text, openai_api_key):
    chroma_path = 'chroma_db/'

    # Prepare the database
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Search the database
    results = db.similarity_search_with_relevance_scores(query_text)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    return query_text, context_text
