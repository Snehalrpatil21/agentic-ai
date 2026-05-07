from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import requests
import os

CHROMA_PATH = "chroma"

class ChromaRetrieverTool(BaseTool):
    name = "chroma_retriever"
    description = "Retrieve relevant document chunks from the Chroma vector store based on a query."

    def _run(self, query: str) -> List[Dict[str, Any]]:
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_relevance_scores(query, k=10)
        filtered = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
            if score > 0.7
        ]
        return filtered[:5]

class WebLookupTool(BaseTool):
    name = "web_lookup"
    description = "Perform a web search or SEC lookup for fresh information."

    def _run(self, query: str) -> str:
        # Placeholder for web search; integrate with a real API like SerpAPI or SEC EDGAR
        # For now, return a mock response
        return f"Mock web search result for '{query}': No real-time data available in this demo."

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform financial calculations or numeric analysis."

    def _run(self, expression: str) -> str:
        try:
            # Use eval for simple calculations; in production, use a safer library
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

class SummarizerTool(BaseTool):
    name = "summarizer"
    description = "Summarize long text or responses."

    def _run(self, text: str) -> str:
        model = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt = f"Summarize the following text concisely:\n\n{text}"
        return model.predict(prompt)