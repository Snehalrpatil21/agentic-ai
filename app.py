from flask import Flask, request, render_template, redirect, url_for, session
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Use the previous conversation history to answer this follow-up question.

{history}

Answer the question based on the above context and history: {question}. if you don't get the answer or need more context please ask for more information
"""

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

@app.route('/')
def index():
    response = session.pop('response', None)
    sources = session.pop('sources', None)
    current_question = session.pop('current_question', None)
    history = session.get('history', [])
    return render_template('index.html', response=response, sources=sources, current_question=current_question, history=history)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form['query']
    
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    history = session.get('history', [])
    search_query = query_text
    if history:
        recent_history = history[-3:]
        history_context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in recent_history])
        rewrite_prompt = f"Given the conversation history:\n{history_context}\n\nRewrite this follow-up question into a standalone query that can be searched in a document database: '{query_text}'"
        rewrite_model = ChatOpenAI(model_name="gpt-4", temperature=0)
        search_query = rewrite_model.predict(rewrite_prompt).strip()

    results = db.similarity_search_with_relevance_scores(search_query, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        response_text = "Unable to find matching results."
        sources = []
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        history = session.get('history', [])
        history_text = "\n\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in history]) if history else "No prior conversation history."

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, history=history_text, question=query_text)

        model = ChatOpenAI(model_name="gpt-4", temperature=0)
        response_text = model.predict(prompt)
        sources = [{"source": doc.metadata.get("source"), "page": doc.metadata.get("page"), "chunk_index": doc.metadata.get("start_index")} for doc, _score in results]
    
    # Update history: append new Q&A and keep only last 5
    history = session.get('history', [])
    history.append({'question': query_text, 'answer': response_text, 'sources': sources})
    session['history'] = history[-5:]  # Keep only the last 5
    
    session['response'] = response_text
    session['sources'] = sources
    session['current_question'] = query_text
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)