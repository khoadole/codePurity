import os
import subprocess
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
ALLOWED_EXTENSIONS = {'py'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if OpenAI API key was provided
    openai_api_key = request.form.get('openai_api_key', '').strip()
    if not openai_api_key:
        flash('OpenAI API key is required', 'error')
        return redirect(url_for('index'))
    
    # Check if paper name was provided
    paper_name = request.form.get('paper_name', '').strip()
    if not paper_name:
        flash('Paper name is required', 'error')
        return redirect(url_for('index'))
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Create a unique directory for this run
        run_id = str(uuid.uuid4())
        run_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], run_id)
        run_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], run_id)
        
        os.makedirs(run_upload_dir, exist_ok=True)
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_file_path = os.path.join(run_upload_dir, filename)
        file.save(input_file_path)
        
        # Save run info to session
        session['run_id'] = run_id
        session['paper_name'] = paper_name
        session['filename'] = filename
        
        # Create a cleaned file path
        cleaned_file_path = os.path.join(run_upload_dir, f"{os.path.splitext(filename)[0]}_cleaned.py")
        
        try:
            # Step 1: Preprocess
            preprocess_cmd = [
                'python', 'code_process.py',
                '--input_file', input_file_path,
                '--output_file', cleaned_file_path
            ]
            subprocess.run(preprocess_cmd, check=True, env={**os.environ, 'OPENAI_API_KEY': openai_api_key})
            
            # Step 2: Planning
            planning_cmd = [
                'python', 'planning.py',
                '--paper_name', paper_name,
                '--gpt_version', 'gpt-3.5-turbo',
                '--input_python', cleaned_file_path,
                '--output_dir', run_output_dir
            ]
            subprocess.run(planning_cmd, check=True, env={**os.environ, 'OPENAI_API_KEY': openai_api_key})
            
            # Step 3: Analyzing
            analyzing_cmd = [
                'python', 'analyzing.py',
                '--input_file', cleaned_file_path,
                '--output_file', os.path.join(run_output_dir, 'analysis_result.json')
            ]
            subprocess.run(analyzing_cmd, check=True, env={**os.environ, 'OPENAI_API_KEY': openai_api_key})
            
            # Step 4: Make Paper
            makepaper_cmd = [
                'python', 'makepaper.py',
                '--output_dir', run_output_dir,
                '--gpt_version', 'gpt-3.5-turbo'
            ]
            subprocess.run(makepaper_cmd, check=True, env={**os.environ, 'OPENAI_API_KEY': openai_api_key})
            
            flash('Processing completed successfully!', 'success')
            return redirect(url_for('results', run_id=run_id))
            
        except subprocess.CalledProcessError as e:
            flash(f'Error during processing: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Only Python (.py) files are allowed.', 'error')
    return redirect(url_for('index'))

@app.route('/results/<run_id>')
def results(run_id):
    run_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], run_id)
    
    # Check if the output directory exists
    if not os.path.exists(run_output_dir):
        flash('Output directory not found', 'error')
        return redirect(url_for('index'))
    
    # Get list of generated files
    generated_files = []
    
    # Check for markdown file
    markdown_path = os.path.join(run_output_dir, 'paper.md')
    if os.path.exists(markdown_path):
        generated_files.append({
            'name': 'Markdown Paper',
            'path': f'download/{run_id}/paper.md',
            'type': 'markdown'
        })
    
    # Check for LaTeX file
    latex_path = os.path.join(run_output_dir, 'paper.tex')
    if os.path.exists(latex_path):
        generated_files.append({
            'name': 'LaTeX Paper',
            'path': f'download/{run_id}/paper.tex',
            'type': 'latex'
        })
    
    # Check for PDF file
    pdf_path = os.path.join(run_output_dir, 'paper.pdf')
    if os.path.exists(pdf_path):
        generated_files.append({
            'name': 'PDF Paper',
            'path': f'download/{run_id}/paper.pdf',
            'type': 'pdf'
        })
    
    # Check for analysis result
    analysis_path = os.path.join(run_output_dir, 'analysis_result.json')
    if os.path.exists(analysis_path):
        generated_files.append({
            'name': 'Analysis Result (JSON)',
            'path': f'download/{run_id}/analysis_result.json',
            'type': 'json'
        })
    
    # Check for paper plan
    plan_path = os.path.join(run_output_dir, 'paper_plan.json')
    if os.path.exists(plan_path):
        generated_files.append({
            'name': 'Paper Plan (JSON)',
            'path': f'download/{run_id}/paper_plan.json',
            'type': 'json'
        })
    
    # Check for figures
    figures_dir = os.path.join(run_output_dir, 'figures')
    if os.path.exists(figures_dir):
        for figure_file in os.listdir(figures_dir):
            if figure_file.endswith('.png'):
                generated_files.append({
                    'name': f'Figure: {figure_file}',
                    'path': f'download/{run_id}/figures/{figure_file}',
                    'type': 'image'
                })
    
    # Try to read markdown content for preview
    markdown_content = None
    if os.path.exists(markdown_path):
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except Exception as e:
            print(f"Error reading markdown: {e}")
    
    return render_template('results.html', 
                          generated_files=generated_files, 
                          markdown_content=markdown_content,
                          run_id=run_id)

@app.route('/download/<run_id>/<path:filename>')
def download_file(run_id, filename):
    # Ensure the run_id is valid and not trying to escape the directory
    if '..' in run_id or '/' in run_id:
        flash('Invalid run ID', 'error')
        return redirect(url_for('index'))
    
    # Determine if we're looking for a figure or a main output file
    if filename.startswith('figures/'):
        # This is a figure file
        figure_name = filename.split('/')[-1]
        return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], run_id, 'figures'), 
                                   figure_name)
    else:
        # This is a main output file
        return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], run_id), 
                                   filename)

@app.route('/view_markdown/<run_id>')
def view_markdown(run_id):
    markdown_path = os.path.join(app.config['OUTPUT_FOLDER'], run_id, 'paper.md')
    
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        return render_template('markdown_view.html', markdown_content=markdown_content)
    except Exception as e:
        flash(f'Error reading markdown file: {str(e)}', 'error')
        return redirect(url_for('results', run_id=run_id))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)