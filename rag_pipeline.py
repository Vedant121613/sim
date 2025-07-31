# import os
# import hashlib
# import pickle
# from functools import lru_cache
# from typing import List, Dict, Any
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, RetrievalQA

# # Load API Key from file
# with open("API_KEY.txt", "r") as f:
#     GEMINI_API_KEY = f.read().strip()

# genai.configure(api_key=GEMINI_API_KEY)

# class RAGPipeline:
#     def __init__(self, pdf_path, use_cache=True):
#         self.pdf_path = pdf_path
#         self.use_cache = use_cache
#         self.cache_dir = "rag_cache"
        
#         # Create cache directory
#         os.makedirs(self.cache_dir, exist_ok=True)
        
#         # Initialize components with caching
#         self._initialize_components()
    
#     def _get_cache_path(self, suffix: str) -> str:
#         """Generate cache file path based on PDF file hash"""
#         file_hash = hashlib.md5(self.pdf_path.encode()).hexdigest()
#         return os.path.join(self.cache_dir, f"{file_hash}_{suffix}.pkl")
    
#     def _initialize_components(self):
#         """Initialize or load cached components"""
#         # Check if components are cached
#         vectorstore_cache = self._get_cache_path("vectorstore")
#         chunks_cache = self._get_cache_path("chunks")
        
#         if self.use_cache and os.path.exists(vectorstore_cache) and os.path.exists(chunks_cache):
#             print(f"Loading cached components for {self.pdf_path}")
#             self._load_from_cache()
#         else:
#             print(f"Creating new components for {self.pdf_path}")
#             self._create_components()
#             if self.use_cache:
#                 self._save_to_cache()
        
#         # Initialize LLM and chains (these are lightweight)
#         self._initialize_llm_chains()
    
#     def _load_from_cache(self):
#         """Load components from cache"""
#         try:
#             # Load text chunks
#             with open(self._get_cache_path("chunks"), 'rb') as f:
#                 self.text_chunks = pickle.load(f)
            
#             # Load vectorstore
#             vectorstore_path = self._get_cache_path("vectorstore")
#             self.vectorstore = FAISS.load_local(
#                 vectorstore_path, 
#                 HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#                 allow_dangerous_deserialization=True
#             )
            
#             self.retriever = self.vectorstore.as_retriever(search_type="mmr")
#             print("Successfully loaded components from cache")
            
#         except Exception as e:
#             print(f"Failed to load from cache: {e}. Creating new components.")
#             self._create_components()
    
#     def _save_to_cache(self):
#         """Save components to cache"""
#         try:
#             # Save text chunks
#             with open(self._get_cache_path("chunks"), 'wb') as f:
#                 pickle.dump(self.text_chunks, f)
            
#             # Save vectorstore
#             vectorstore_path = self._get_cache_path("vectorstore")
#             self.vectorstore.save_local(vectorstore_path)
            
#             print("Successfully saved components to cache")
            
#         except Exception as e:
#             print(f"Failed to save to cache: {e}")
    
#     def _create_components(self):
#         """Create new components"""
#         # Load documents
#         self.documents = PyMuPDFLoader(self.pdf_path).load()
        
#         # Split documents
#         self.text_chunks = RecursiveCharacterTextSplitter(
#             chunk_size=400, 
#             chunk_overlap=50
#         ).split_documents(self.documents)
        
#         # Create embeddings (reuse the same instance)
#         if not hasattr(self, 'embeddings'):
#             self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
#         # Create vectorstore
#         self.vectorstore = FAISS.from_documents(self.text_chunks, self.embeddings)
#         self.retriever = self.vectorstore.as_retriever(search_type="mmr")
    
#     def _initialize_llm_chains(self):
#         """Initialize LLM and chains"""
#         # Initialize LLM model
#         self.llm_model = ChatGoogleGenerativeAI(
#             model="gemini-2.0-flash",
#             google_api_key=GEMINI_API_KEY,
#             temperature=0.0
#         )
        
#         # Custom prompt
#         custom_prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
# You are a professional insurance document analyst. Your task is to answer user queries based strictly on the provided insurance policy document context.

# Guidelines for generating answers:

# - Use ONLY the information present in the context provided. Do not use outside knowledge or assumptions.
# - Be accurate and formal in tone.
# - Include exact details (e.g., number of days/months, specific conditions, monetary values, policy terms, legal references) when present.
# - Begin with "Yes" or "No" when relevant, followed by a precise explanation.
# - If the information is not available in the context, clearly state: "Information not available in the provided document."
# - Limit your answer to **ONE single sentence**, no matter the complexity.
# - Avoid vague language. Prefer concrete facts over generalizations.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#         )
        
#         # Initialize chains
#         self.qa_llm_chain = LLMChain(llm=self.llm_model, prompt=custom_prompt)
        
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm_model,
#             chain_type="stuff",
#             retriever=self.retriever,
#             chain_type_kwargs={"prompt": custom_prompt},
#             return_source_documents=False,
#         )
    
#     @lru_cache(maxsize=500)
#     def _cached_ask(self, question_hash: str, question: str) -> str:
#         """Cache question answers"""
#         return self.qa_chain.run(question)
    
#     def ask(self, question: str) -> str:
#         """Answer a question with caching"""
#         if self.use_cache:
#             # Create hash for caching
#             question_hash = hashlib.md5(question.encode()).hexdigest()
#             return self._cached_ask(question_hash, question)
#         else:
#             return self.qa_chain.run(question)
    
#     def batch_ask(self, questions: List[str]) -> List[str]:
#         """
#         Process multiple questions efficiently
#         For now, processes them individually but can be optimized further
#         """
#         answers = []
#         for question in questions:
#             answer = self.ask(question)
#             answers.append(answer)
#         return answers
    
#     def clear_cache(self):
#         """Clear the LRU cache for questions"""
#         self._cached_ask.cache_clear()
        
#     def get_cache_info(self):
#         """Get cache statistics"""
#         return {
#             "question_cache_info": self._cached_ask.cache_info()._asdict(),
#             "vectorstore_cached": os.path.exists(self._get_cache_path("vectorstore")),
#             "chunks_cached": os.path.exists(self._get_cache_path("chunks"))
#         }
    
#     def preload_similar_questions(self, sample_questions: List[str]):
#         """Preload cache with sample questions for faster future responses"""
#         print(f"Preloading {len(sample_questions)} sample questions...")
#         for question in sample_questions:
#             self.ask(question)
#         print("Preloading completed")

# # Global embeddings instance to reuse across multiple RAG instances
# _embeddings_instance = None

# def get_embeddings_instance():
#     """Get or create a global embeddings instance for reuse"""
#     global _embeddings_instance
#     if _embeddings_instance is None:
#         _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     return _embeddings_instance

import os
from dotenv import load_dotenv
import hashlib
import pickle
from functools import lru_cache
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient

load_dotenv()
# Load Gemini API Key from file
# with open("API_KEY.txt", "r") as f:
#     GEMINI_API_KEY = f.read().strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

# Hugging Face config
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ Custom LangChain-compatible wrapper
class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model: str, token: str):
        self.client = InferenceClient(model=model, token=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.client.feature_extraction(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text)


class RAGPipeline:
    def __init__(self, pdf_path, use_cache=True):
        self.pdf_path = pdf_path
        self.use_cache = use_cache
        self.cache_dir = "rag_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # ✅ Initialize wrapped embedding model
        self.embedding_model = HFInferenceEmbeddings(model=HF_MODEL, token=HF_TOKEN)

        self._initialize_components()

    def _get_cache_path(self, suffix: str) -> str:
        file_hash = hashlib.md5(self.pdf_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{file_hash}_{suffix}.pkl")

    def _initialize_components(self):
        vectorstore_cache = self._get_cache_path("vectorstore")
        chunks_cache = self._get_cache_path("chunks")

        if self.use_cache and os.path.exists(vectorstore_cache) and os.path.exists(chunks_cache):
            self._load_from_cache()
        else:
            self._create_components()
            if self.use_cache:
                self._save_to_cache()

        self._initialize_llm_chains()

    def _load_from_cache(self):
        with open(self._get_cache_path("chunks"), 'rb') as f:
            self.text_chunks = pickle.load(f)

        vectorstore_path = self._get_cache_path("vectorstore")
        self.vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_type="mmr")

    def _save_to_cache(self):
        with open(self._get_cache_path("chunks"), 'wb') as f:
            pickle.dump(self.text_chunks, f)
        self.vectorstore.save_local(self._get_cache_path("vectorstore"))

    def _create_components(self):
        self.documents = PyMuPDFLoader(self.pdf_path).load()
        self.text_chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(self.documents)

        self.vectorstore = FAISS.from_documents(self.text_chunks, self.embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_type="mmr")

    def _initialize_llm_chains(self):
        self.llm_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a professional insurance document analyst. Your task is to answer user queries based strictly on the provided insurance policy document context.

Guidelines:
- Use ONLY the information in the context.
- Be accurate and formal.
- Include policy details (terms, monetary values, legal references).
- Start with "Yes" or "No" if applicable.
- If info not present, say: "Information not available in the provided document."
- Limit to ONE sentence.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_model,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=False
        )

    @lru_cache(maxsize=500)
    def _cached_ask(self, question_hash: str, question: str) -> str:
        return self.qa_chain.run(question)

    def ask(self, question: str) -> str:
        question_hash = hashlib.md5(question.encode()).hexdigest()
        return self._cached_ask(question_hash, question)

    def batch_ask(self, questions: List[str]) -> List[str]:
        return [self.ask(q) for q in questions]

    def clear_cache(self):
        self._cached_ask.cache_clear()

    def get_cache_info(self):
        return {
            "question_cache_info": self._cached_ask.cache_info()._asdict(),
            "vectorstore_cached": os.path.exists(self._get_cache_path("vectorstore")),
            "chunks_cached": os.path.exists(self._get_cache_path("chunks"))
        }
