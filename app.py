from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
from google import genai
from groq import Groq
import bcrypt
import os
import zipfile
import io
from datetime import datetime

load_dotenv()

def extract_text(filepath, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    try:
        if ext in ['txt', 'md']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext in ['html', 'htm']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                from markdownify import markdownify
                return markdownify(f.read())
        elif ext in ['json']:
            import json
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                return f"```json\n{json.dumps(data, indent=2)}\n```"
        elif ext in ['xml', 'csv']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == 'pdf':
            import fitz
            doc = fitz.open(filepath)
            lines = []
            for page in doc:
                blocks = page.get_text('dict')['blocks']
                for block in blocks:
                    if block['type'] == 0:
                        for line in block['lines']:
                            text = ' '.join([s['text'] for s in line['spans']]).strip()
                            if not text:
                                continue
                            size = line['spans'][0]['size'] if line['spans'] else 0
                            if size >= 16:
                                lines.append(f"\n# {text}")
                            elif size >= 13:
                                lines.append(f"\n## {text}")
                            else:
                                lines.append(text)
                lines.append('\n')
            return '\n'.join(lines)
        elif ext == 'docx':
            from docx import Document
            doc = Document(filepath)
            lines = []
            for para in doc.paragraphs:
                if not para.text.strip():
                    continue
                style = para.style.name.lower()
                if 'heading 1' in style:
                    lines.append(f"\n# {para.text}")
                elif 'heading 2' in style:
                    lines.append(f"\n## {para.text}")
                elif 'heading 3' in style:
                    lines.append(f"\n### {para.text}")
                elif 'list' in style:
                    lines.append(f"- {para.text}")
                else:
                    lines.append(para.text)
            return '\n'.join(lines)
        elif ext in ['xlsx', 'xls']:
            import openpyxl
            wb = openpyxl.load_workbook(filepath)
            lines = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                lines.append(f"\n## Sheet: {sheet_name}\n")
                rows = list(ws.iter_rows(values_only=True))
                if not rows:
                    continue
                headers = [str(c) if c is not None else '' for c in rows[0]]
                lines.append('| ' + ' | '.join(headers) + ' |')
                lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
                for row in rows[1:]:
                    cells = [str(c) if c is not None else '' for c in row]
                    lines.append('| ' + ' | '.join(cells) + ' |')
            return '\n'.join(lines)
        elif ext == 'pptx':
            from pptx import Presentation
            prs = Presentation(filepath)
            lines = []
            for i, slide in enumerate(prs.slides):
                lines.append(f"\n## Slide {i + 1}\n")
                for shape in slide.shapes:
                    if hasattr(shape, 'text') and shape.text.strip():
                        if shape.shape_type == 13:
                            continue
                        lines.append(f"- {shape.text.strip()}")
            return '\n'.join(lines)
        elif ext in ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff']:
            return f'[Image file: {filename}]'
        elif ext in ['mp3', 'wav', 'mp4']:
            return f'[Media file: {filename}]'
        else:
            return f'[Unsupported file type: {filename}]'
    except Exception as e:
        return f'[Could not extract text: {str(e)}]'

def get_all_files(kb):
    """Recursively get all files from a KB and all its children."""
    files = list(kb.files)
    for child in kb.children:
        files.extend(get_all_files(child))
    return files

def get_kb_stats(kb):
    total_size = 0
    last_modified = None
    all_files = get_all_files(kb)
    for f in all_files:
        if os.path.exists(f.filepath):
            total_size += os.path.getsize(f.filepath)
            mtime = os.path.getmtime(f.filepath)
            if last_modified is None or mtime > last_modified:
                last_modified = mtime
    if total_size < 1024:
        size_str = f"{total_size} B"
    elif total_size < 1024 * 1024:
        size_str = f"{total_size / 1024:.1f} KB"
    else:
        size_str = f"{total_size / (1024*1024):.1f} MB"
    date_str = datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M') if last_modified else 'No files yet'
    return size_str, date_str, len(all_files)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///knowledge_agent.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

@app.template_filter('filesize')
def filesize_filter(filepath):
    try:
        size = os.path.getsize(filepath)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f} KB"
        else:
            return f"{size/(1024*1024):.1f} MB"
    except:
        return ''

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

class KnowledgeBase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True, default='')
    instructions = db.Column(db.Text, nullable=True, default='')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('knowledge_base.id'), nullable=True)
    children = db.relationship('KnowledgeBase', backref=db.backref('parent', remote_side='KnowledgeBase.id'), lazy=True, cascade='all, delete-orphan')
    files = db.relationship('KBFile', backref='knowledge_base', lazy=True, cascade='all, delete-orphan')

class KBFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=True)
    kb_id = db.Column(db.Integer, db.ForeignKey('knowledge_base.id'), nullable=False)

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
    kbs = KnowledgeBase.query.filter_by(user_id=current_user.id).all()
    kb_stats = {kb.id: get_kb_stats(kb) for kb in kbs}
    return render_template('dashboard.html',
        conversations=conversations,
        active_conversation=active_conversation,
        messages=messages,
        kbs=kbs,
        kb_stats=kb_stats)

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
    response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=message)
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

@app.route('/kb')
@login_required
def kb_manager():
    return redirect(url_for('dashboard') + '?mode=kb')

@app.route('/kb/new', methods=['POST'])
@login_required
def kb_new():
    name = request.form.get('name', '').strip()
    parent_id = request.form.get('parent_id', None)
    if parent_id:
        parent_id = int(parent_id)
    if name:
        kb = KnowledgeBase(name=name, user_id=current_user.id, parent_id=parent_id)
        db.session.add(kb)
        db.session.commit()
        os.makedirs(f'uploads/user_{current_user.id}/kb_{kb.id}', exist_ok=True)
    if parent_id:
        return redirect(url_for('kb_detail', kb_id=parent_id))
    return redirect(url_for('dashboard') + '?mode=kb&view=manage')

@app.route('/kb/import', methods=['POST'])
@login_required
def kb_import_global():
    from werkzeug.utils import secure_filename
    file = request.files.get('zip_file')
    if not file or not file.filename.endswith('.zip'):
        return redirect(url_for('dashboard') + '?mode=kb')
    zip_name = file.filename.replace('.zip', '')
    new_kb = KnowledgeBase(name=zip_name, user_id=current_user.id, parent_id=None)
    db.session.add(new_kb)
    db.session.commit()
    folder = f'uploads/user_{current_user.id}/kb_{new_kb.id}'
    os.makedirs(folder, exist_ok=True)
    zip_buffer = io.BytesIO(file.read())
    with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
        for zip_entry in zip_file.namelist():
            if zip_entry.endswith('/'):
                continue
            filename = secure_filename(os.path.basename(zip_entry))
            if not filename:
                continue
            filepath = os.path.join(folder, filename)
            with open(filepath, 'wb') as f:
                f.write(zip_file.read(zip_entry))
            content = extract_text(filepath, filename)
            kb_file = KBFile(filename=filename, filepath=filepath, content=content, kb_id=new_kb.id)
            db.session.add(kb_file)
    db.session.commit()
    return redirect(url_for('kb_detail', kb_id=new_kb.id))

@app.route('/kb/<int:kb_id>')
@login_required
def kb_detail(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    breadcrumb = []
    parent = kb.parent
    while parent:
        breadcrumb.insert(0, parent)
        parent = parent.parent
    return render_template('kb_detail.html', kb=kb, breadcrumb=breadcrumb)

@app.route('/kb/<int:kb_id>/explore')
@login_required
def kb_explore(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    breadcrumb = []
    parent = kb.parent
    while parent:
        breadcrumb.insert(0, parent)
        parent = parent.parent
    return render_template('kb_explore.html', kb=kb, breadcrumb=breadcrumb)

@app.route('/kb/<int:kb_id>/delete')
@login_required
def kb_delete(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    parent_id = kb.parent_id if kb else None
    if kb and kb.user_id == current_user.id:
        db.session.delete(kb)
        db.session.commit()
    if parent_id:
        return redirect(url_for('kb_detail', kb_id=parent_id))
    return redirect(url_for('dashboard') + '?mode=kb&view=manage')

@app.route('/kb/<int:kb_id>/rename', methods=['POST'])
@login_required
def kb_rename(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if kb and kb.user_id == current_user.id:
        new_name = request.form.get('name', '').strip()
        if new_name:
            kb.name = new_name
            db.session.commit()
    if kb and kb.parent_id:
        return redirect(url_for('kb_detail', kb_id=kb.parent_id))
    return redirect(url_for('kb_detail', kb_id=kb_id))

@app.route('/kb/<int:kb_id>/upload', methods=['POST'])
@login_required
def kb_upload(kb_id):
    from werkzeug.utils import secure_filename
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return redirect(url_for('dashboard') + '?mode=kb')
    file = request.files.get('file')
    if file and file.filename:
        filename = secure_filename(file.filename)
        folder = f'uploads/user_{current_user.id}/kb_{kb_id}'
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        file.save(filepath)
        content = extract_text(filepath, filename)
        kb_file = KBFile(filename=filename, filepath=filepath, content=content, kb_id=kb_id)
        db.session.add(kb_file)
        db.session.commit()
    return redirect(url_for('kb_detail', kb_id=kb_id))

@app.route('/kb/<int:kb_id>/file/<int:file_id>/delete')
@login_required
def kb_file_delete(kb_id, file_id):
    f = KBFile.query.get(file_id)
    if f and f.kb_id == kb_id:
        if os.path.exists(f.filepath):
            os.remove(f.filepath)
        db.session.delete(f)
        db.session.commit()
    return redirect(url_for('kb_detail', kb_id=kb_id))

@app.route('/kb/<int:kb_id>/file/<int:file_id>/rename', methods=['POST'])
@login_required
def kb_file_rename(kb_id, file_id):
    f = KBFile.query.get(file_id)
    if not f or f.kb_id != kb_id:
        return jsonify({'error': 'Invalid file'}), 400
    new_name = request.form.get('new_name', '').strip()
    if not new_name:
        return jsonify({'error': 'Invalid name'}), 400
    old_path = f.filepath
    folder = os.path.dirname(old_path)
    new_path = os.path.join(folder, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    f.filename = new_name
    f.filepath = new_path
    db.session.commit()
    return jsonify({'success': True})

@app.route('/kb/<int:kb_id>/description/save', methods=['POST'])
@login_required
def kb_description_save(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return jsonify({'success': False}), 400
    description = request.form.get('description', '').strip()
    kb.description = description
    db.session.commit()
    return redirect(url_for('kb_detail', kb_id=kb_id))

@app.route('/kb/<int:kb_id>/instructions/save', methods=['POST'])
@login_required
def kb_instructions_save(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return jsonify({'success': False}), 400
    instructions = request.form.get('instructions', '').strip()
    kb.instructions = instructions
    db.session.commit()
    return redirect(url_for('kb_detail', kb_id=kb_id))

@app.route('/kb/<int:kb_id>/export/selected', methods=['POST'])
@login_required
def kb_export_selected(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return jsonify({'error': 'Invalid KB'}), 400
    file_ids = request.json.get('file_ids', [])
    if not file_ids:
        return jsonify({'error': 'No files selected'}), 400
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_id in file_ids:
            kb_file = KBFile.query.get(int(file_id))
            if kb_file and kb_file.kb_id == kb_id and os.path.exists(kb_file.filepath):
                zip_file.write(kb_file.filepath, arcname=kb_file.filename)
    zip_buffer.seek(0)
    now = datetime.now().strftime('%Y-%m-%d')
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'{kb.name}_{now}.zip')

@app.route('/kb/<int:kb_id>/export/all')
@login_required
def kb_export_all(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return redirect(url_for('dashboard') + '?mode=kb')
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for kb_file in kb.files:
            if os.path.exists(kb_file.filepath):
                zip_file.write(kb_file.filepath, arcname=kb_file.filename)
    zip_buffer.seek(0)
    now = datetime.now().strftime('%Y-%m-%d')
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f'{kb.name}_{now}.zip')

@app.route('/kb/<int:kb_id>/files/delete-selected', methods=['POST'])
@login_required
def kb_files_delete_selected(kb_id):
    kb = KnowledgeBase.query.get(kb_id)
    if not kb or kb.user_id != current_user.id:
        return redirect(url_for('dashboard') + '?mode=kb')
    file_ids = request.json.get('file_ids', [])
    for file_id in file_ids:
        f = KBFile.query.get(int(file_id))
        if f and f.kb_id == kb_id:
            if os.path.exists(f.filepath):
                os.remove(f.filepath)
            db.session.delete(f)
    db.session.commit()
    return jsonify({'success': True})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)