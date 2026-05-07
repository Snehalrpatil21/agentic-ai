import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, render_template, redirect, url_for, session
from agents import run_agent

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

@app.route('/')
def index():
    response = session.pop('response', None)
    sources = session.pop('sources', None)
    current_question = session.pop('current_question', None)
    history = session.get('history', [])
    action_trace = session.pop('action_trace', None)
    confidence = session.pop('confidence', None)
    return render_template('index.html', response=response, sources=sources, current_question=current_question, history=history, action_trace=action_trace, confidence=confidence)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form['query']
    history = session.get('history', [])
    
    # Run the agent
    result = run_agent(query_text, history)
    response_text = result["answer"]
    sources = result["sources"]
    action_trace = result["action_trace"]
    confidence = result["confidence"]
    
    # Update history
    history.append({'question': query_text, 'answer': response_text, 'sources': sources, 'action_trace': action_trace, 'confidence': confidence})
    session['history'] = history[-5:]  # Keep only the last 5
    
    session['response'] = response_text
    session['sources'] = sources
    session['current_question'] = query_text
    session['action_trace'] = action_trace
    session['confidence'] = confidence
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)