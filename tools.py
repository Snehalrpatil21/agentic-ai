from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import requests
import os
from sympy import sympify, SympifyError
from serpapi import GoogleSearch

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
    description = "Search Google for real-time web information."

    def _run(self, query: str) -> str:
        api_key = os.environ.get('SERPAPI_KEY')
        if not api_key:
            return "Error: SERPAPI_KEY not configured in .env file. Please add your SerpAPI key."
        
        try:
            params = {
                "q": query,
                "api_key": api_key,
                "num": 3
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "organic_results" in results:
                output = f"Web search results for '{query}':\n\n"
                for i, result in enumerate(results["organic_results"][:3], 1):
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', 'No snippet')
                    link = result.get('link', 'No link')
                    output += f"{i}. {title}\nSnippet: {snippet}\nLink: {link}\n\n"
                return output
            else:
                return "No search results found."
        except Exception as e:
            return f"Error in web search: {str(e)}"

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform financial calculations or numeric analysis."

    def _run(self, expression: str) -> str:
        try:
            result = sympify(expression)
            return str(result)
        except SympifyError as e:
            return f"Error in calculation: {str(e)}"

class SummarizerTool(BaseTool):
    name = "summarizer"
    description = "Summarize long text or responses."

    def _run(self, text: str) -> str:
        model = ChatOpenAI(model_name="gpt-4", temperature=0)
        prompt = f"Summarize the following text concisely:\n\n{text}"
        return model.predict(prompt)