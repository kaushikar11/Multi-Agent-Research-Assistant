import argparse
from dataclasses import dataclass
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()


CHROMA_PATH = "chroma1"

PROMPT_TEMPLATE= """ 
Answer the question based only on the foloowing context:
{context}

---

Answer the question basedon the above context : {question}

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type = str, help = "The query text")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results")
        return
    
    contex_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=contex_text, question=query_text)
    print(f"Prompt: {prompt}")

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    response_text = model.invoke(prompt).content

    sources = [doc.metadata["source"] for doc, _score in results]
    formatted_response = f"Answer: {response_text}\n\nSources:\n" + "\n".join(sources)
    print(formatted_response)

if __name__ == "__main__":
    main()