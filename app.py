from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from dotenv import load_dotenv
from google import genai
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

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

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
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=user_message
    )
    return jsonify({'response': response.text})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)