from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import RetrievalQA
import json

# -------------------------------------------------------------
# LOAD JSON
# -------------------------------------------------------------
def load_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] load_json_file: {e}")
        return None

# -------------------------------------------------------------
# TEAM 3 RAG PROCESS
# -------------------------------------------------------------
def run_team3_rag(input_json_path, output_json_path):
    
    try:
        data = load_json_file(input_json_path)
        if not data:
            print("No data found in input JSON.")
            return

        # Extract query and retrieved chunks 
        question = data.get("query", "")
        docs = data.get("retrieved_chunks", [])

        if not question:
            print("No query found in input JSON.")
            return

        # Extract text from retrieved chunks
        texts = [chunk["text"] for chunk in docs]

        # 1) Embedding model
        emb = OllamaEmbeddings(model="nomic-embed-text")

        # 2) Build FAISS vector store
        vectorstore = FAISS.from_texts(texts, emb)
        retriever = vectorstore.as_retriever()

        # 3) LLM model
        llm = OllamaLLM(model="tinyllama")

        # 4) QA system
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        # 5) Use invoke() instead of deprecated run()
        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "")

        # 6) Save output
        output = {"result": answer}
        with open(output_json_path, "w") as g:
            json.dump(output, g, indent=4)

        print(" LLM Output Saved:", output_json_path)

    except Exception as e:
        print(f"[ERROR] run_team3_rag: {e}")
