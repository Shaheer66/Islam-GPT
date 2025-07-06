# symptom_predictor.py
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

class SymptomPredictor:
    def __init__(self):
        # Load Hugging Face model
        self.HF_TOKEN = os.environ.get("HF_TOKEN")
        self.HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

        self.llm = HuggingFaceEndpoint(
            repo_id=self.HUGGINGFACE_REPO_ID,
            temperature=1.5,
            model_kwargs={"token": self.HF_TOKEN, "max_length": "512"}
        )

        # Load FAISS database
        self.DB_FAISS_PATH = "vectorstore/db_faiss"
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.load_local(self.DB_FAISS_PATH, self.embedding_model, allow_dangerous_deserialization=True)

        # Set custom prompt
        self.CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        prompt = PromptTemplate(template=self.CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )

    def predict(self, user_query):
        """Predicts the disease based on user symptoms"""
        response = self.qa_chain.invoke({'query': user_query})
        return {
            "result": response["result"],
           # "source_documents": response["source_documents"]
        }
