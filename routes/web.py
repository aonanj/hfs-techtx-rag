from flask import Blueprint, render_template

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    return render_template('index.html')

@web_bp.route('/upload')
def upload():
    return render_template('upload.html')

@web_bp.route('/manifest')
def manifest_status():
    return render_template('manifest.html')

@web_bp.route('/chunks')
def chunk_status():
    return render_template('chunks.html')

@web_bp.route('/db-viewer')
def db_viewer():
    return render_template('dbviewer.html')
