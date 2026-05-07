from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tools import ChromaRetrieverTool, WebLookupTool, CalculatorTool, SummarizerTool
from typing import TypedDict, List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key is None:
    raise EnvironmentError("OPENAI_API_KEY is not set. Add it to .env or environment.")

# Define the state for the graph
class AgentState(TypedDict):
    query: str
    history: List[Dict[str, str]]
    intent: str
    retrieved_docs: List[Dict[str, Any]]
    tool_outputs: Dict[str, Any]
    answer: str
    sources: List[Dict[str, Any]]
    action_trace: List[str]
    confidence: float

# Initialize tools
retriever_tool = ChromaRetrieverTool()
web_tool = WebLookupTool()
calc_tool = CalculatorTool()
summarizer_tool = SummarizerTool()

model = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

# Node functions
def query_intent_node(state: AgentState) -> AgentState:
    query = state["query"]
    history = state["history"]
    prompt = f"Classify the intent of this query: '{query}'. History: {history}. Options: document_lookup, numeric_analysis, external_search, clarification_needed."
    intent = model.predict(prompt).strip().lower()
    state["intent"] = intent
    state["action_trace"].append(f"Classified intent as: {intent}")
    return state

def retriever_node(state: AgentState) -> AgentState:
    query = state["query"]
    docs = retriever_tool._run(query)
    state["retrieved_docs"] = docs
    state["action_trace"].append(f"Retrieved {len(docs)} document chunks")
    return state

def tool_selector_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    if intent == "numeric_analysis":
        calc_result = calc_tool._run("extract and compute from docs")  # Placeholder logic
        state["tool_outputs"]["calculator"] = calc_result
        state["action_trace"].append("Used calculator tool")
    elif intent == "external_search":
        web_result = web_tool._run(state["query"])
        state["tool_outputs"]["web"] = web_result
        state["action_trace"].append("Used web lookup tool")
    return state

def synthesis_node(state: AgentState) -> AgentState:
    docs = state["retrieved_docs"]
    tool_outputs = state["tool_outputs"]
    context = "\n".join([doc["content"] for doc in docs])
    prompt = f"Synthesize an answer based on context: {context} and tools: {tool_outputs}. Query: {state['query']}"
    answer = model.predict(prompt)
    state["answer"] = answer
    state["sources"] = [doc["metadata"] for doc in docs]
    state["action_trace"].append("Synthesized final answer")
    return state

def validator_node(state: AgentState) -> AgentState:
    # Simple validation: check if answer mentions sources
    if "source" in state["answer"].lower():
        state["confidence"] = 0.9
    else:
        state["confidence"] = 0.5
    state["action_trace"].append(f"Validated with confidence: {state['confidence']}")
    return state

# Build the graph
graph = StateGraph(AgentState)

graph.add_node("intent_step", query_intent_node)
graph.add_node("retrieve_step", retriever_node)
graph.add_node("select_tools_step", tool_selector_node)
graph.add_node("synthesize_step", synthesis_node)
graph.add_node("validate_step", validator_node)

graph.set_entry_point("intent_step")
graph.add_edge("intent_step", "retrieve_step")
graph.add_edge("retrieve_step", "select_tools_step")
graph.add_edge("select_tools_step", "synthesize_step")
graph.add_edge("synthesize_step", "validate_step")
graph.add_edge("validate_step", END)

agent = graph.compile()

def run_agent(query: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    initial_state = {
        "query": query,
        "history": history,
        "intent": "",
        "retrieved_docs": [],
        "tool_outputs": {},
        "answer": "",
        "sources": [],
        "action_trace": [],
        "confidence": 0.0
    }
    result = agent.invoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "action_trace": result["action_trace"],
        "confidence": result["confidence"]
    }