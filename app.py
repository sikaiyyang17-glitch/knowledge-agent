from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from google import genai
from groq import Groq
import bcrypt
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///knowledge_agent.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), default='New Conversation')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.checkpw(password, user.password_hash):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        confirm_password = request.form['confirm_password'].encode('utf-8')
        if password != confirm_password:
            error = 'Passwords do not match'
        else:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                error = 'Username already taken — please choose another'
            else:
                password_hash = bcrypt.hashpw(password, bcrypt.gensalt())
                new_user = User(username=username, password_hash=password_hash)
                db.session.add(new_user)
                db.session.commit()
                return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/dashboard')
@app.route('/dashboard/<int:conversation_id>')
@login_required
def dashboard(conversation_id=None):
    conversations = Conversation.query.filter_by(user_id=current_user.id).all()
    active_conversation = None
    messages = []
    if conversation_id:
        active_conversation = Conversation.query.get(conversation_id)
    elif conversations:
        active_conversation = conversations[-1]
    if active_conversation:
        messages = Message.query.filter_by(conversation_id=active_conversation.id).all()
    return render_template('dashboard.html',
        conversations=conversations,
        active_conversation=active_conversation,
        messages=messages)

@app.route('/conversation/new')
@login_required
def new_conversation():
    conv = Conversation(title='New Conversation', user_id=current_user.id)
    db.session.add(conv)
    db.session.commit()
    return redirect(url_for('dashboard', conversation_id=conv.id))

@app.route('/conversation/<int:conversation_id>/delete')
@login_required
def delete_conversation(conversation_id):
    conv = Conversation.query.get(conversation_id)
    if conv and conv.user_id == current_user.id:
        db.session.delete(conv)
        db.session.commit()
    return redirect(url_for('dashboard'))

def call_gemini(message):
    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=message
    )
    return response.text

def call_groq(message):
    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': message}]
    )
    return response.choices[0].message.content

def call_ollama(message):
    import requests
    response = requests.post('http://localhost:11434/api/chat', json={
        'model': 'llama3.2',
        'messages': [{'role': 'user', 'content': message}],
        'stream': False
    })
    return response.json()['message']['content']

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    conversation_id = data.get('conversation_id')
    model = data.get('model', 'gemini')

    conv = Conversation.query.get(conversation_id)
    if not conv or conv.user_id != current_user.id:
        return jsonify({'error': 'Invalid conversation'}), 400

    if conv.title == 'New Conversation':
        conv.title = user_message[:50]
        db.session.commit()

    user_msg = Message(role='user', content=user_message, conversation_id=conv.id)
    db.session.add(user_msg)
    db.session.commit()

    try:
        if model == 'groq':
            answer = call_groq(user_message)
        elif model == 'ollama':
            answer = call_ollama(user_message)
        else:
            answer = call_gemini(user_message)
    except Exception as e:
        # Auto-fallback: if Gemini fails, try Groq
        try:
            answer = call_groq(user_message)
            answer = '⚡ (Gemini unavailable, using Groq as backup)\n\n' + answer
        except:
            return jsonify({'error': 'All models unavailable. Please try again.'}), 503

    agent_msg = Message(role='agent', content=answer, conversation_id=conv.id)
    db.session.add(agent_msg)
    db.session.commit()

    return jsonify({'response': answer, 'conversation_id': conv.id})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)