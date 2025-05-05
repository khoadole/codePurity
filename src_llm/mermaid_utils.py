#!/usr/bin/env python3
"""
Mermaid Diagram Generation Utilities

This module generates Mermaid diagrams using an open-source LLM (Mistral) for architecture, class, and component flow diagrams.
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
import re
from PIL import Image
import math
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil
from PIL import ImageFont, ImageDraw
def generate_mermaid_architecture_diagram(classes: Dict[str, Any], model, tokenizer) -> str:
    """
    Generate a Mermaid architecture diagram using the LLM.
    
    Args:
        classes: Dictionary of class information with methods and properties
        model: Loaded LLM model
        tokenizer: Tokenizer for the LLM
        
    Returns:
        Mermaid diagram code
    """
    class_info = []
    for class_name, info in classes.items():
        methods = [m.get("name", "") for m in info.get("methods", [])]
        attributes = [a.get("name", "") for a in info.get("attributes", [])]
        inherits = info.get("inherits_from", [])
        
        class_info.append({
            "name": class_name,
            "methods": methods[:5],
            "attributes": attributes[:5],
            "inherits_from": inherits
        })
    
    prompt = f"""
    Generate a Mermaid class diagram that represents the architecture of a system with the following classes.
    For each class, I'll provide its name, key methods, attributes, and inheritance relationships.
    
    Class Information:
    {json.dumps(class_info, indent=2)}
    
    Please create a Mermaid class diagram using the classDiagram syntax. Include:
    1. All classes with their methods and attributes
    2. Inheritance relationships using <|-- notation
    3. Reasonable assumptions about relationships between classes based on naming patterns
    
    The diagram should be well-organized and easy to read. Use appropriate Mermaid syntax to represent the architecture clearly.
    Only return the Mermaid code without any explanation or additional text.
    Begin your response with "```mermaid" and end with "```".
    """
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=2000,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        mermaid_code = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        mermaid_code = extract_mermaid_code(mermaid_code)
        return mermaid_code
    except Exception as e:
        print(f"Error generating architecture diagram: {e}")
        return """classDiagram
            class MainClass {
                +process()
                +initialize()
            }
            class Helper {
                +support()
            }
            MainClass --> Helper"""

def generate_mermaid_class_diagram(classes: Dict[str, Any], dependencies: Dict[str, Any], model, tokenizer) -> str:
    """
    Generate a more detailed Mermaid class diagram using the LLM.
    
    Args:
        classes: Dictionary of class information
        dependencies: Dictionary of dependency information
        model: Loaded LLM model
        tokenizer: Tokenizer for the LLM
        
    Returns:
        Mermaid diagram code
    """
    class_dependencies = []
    for name, dep_info in dependencies.items():
        if dep_info.get("type") == "class" and name in classes:
            depends_on = [dep for dep in dep_info.get("depends_on", []) if dep in classes]
            if depends_on:
                class_dependencies.append({
                    "class": name,
                    "depends_on": depends_on
                })
    
    prompt = f"""
    Generate a detailed Mermaid class diagram that shows both class structures and their relationships.
    
    Class Information:
    {json.dumps([{"name": name, "methods": [m.get("name") for m in info.get("methods", [])[:5]], 
                 "attributes": [a.get("name") for a in info.get("attributes", [])[:3]],
                 "inherits_from": info.get("inherits_from", [])} 
                for name, info in classes.items()], indent=2)}
    
    Dependencies:
    {json.dumps(class_dependencies, indent=2)}
    
    Please create a comprehensive Mermaid class diagram that shows:
    1. Classes with their key methods and attributes
    2. Inheritance relationships with <|-- notation
    3. Dependencies between classes with --> notation
    4. Use appropriate relationship types (aggregation, composition, etc.) when they can be inferred
    
    Make the diagram as clear and organized as possible. Group related classes together visually.
    Only return the Mermaid code without any explanation or additional text.
    Begin your response with "```mermaid" and end with "```".
    """
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=2500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        mermaid_code = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        mermaid_code = extract_mermaid_code(mermaid_code)
        return mermaid_code
    except Exception as e:
        print(f"Error generating class diagram: {e}")
        return """classDiagram
            class Class1 {
                +method1()
                +method2()
            }
            class Class2 {
                +method1()
            }
            Class1 --> Class2"""

def generate_mermaid_component_flow_diagram(data_flow: Dict[str, Any], model, tokenizer) -> str:
    """
    Generate a Mermaid component flow diagram using the LLM.
    
    Args:
        data_flow: Dictionary of data flow information
        model: Loaded LLM model
        tokenizer: Tokenizer for the LLM
        
    Returns:
        Mermaid diagram code
    """
    data_paths = data_flow.get("data_paths", [])
    
    prompt = f"""
    Generate a Mermaid flowchart diagram that shows the flow of data between components.
    
    Data Flow Paths:
    {json.dumps(data_paths, indent=2)}
    
    Please create a Mermaid flowchart diagram that:
    1. Shows each component as a node
    2. Shows data flow between components with directional arrows
    3. Labels the arrows with data types when available
    4. Uses appropriate Mermaid node shapes to represent different types of components
    5. Organizes the flow logically from inputs to outputs
    
    The diagram should clearly illustrate the processing pipeline and component interactions.
    Only return the Mermaid code without any explanation or additional text.
    Begin your response with "```mermaid" and end with "```".
    """
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=1500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        mermaid_code = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        mermaid_code = extract_mermaid_code(mermaid_code)
        return mermaid_code
    except Exception as e:
        print(f"Error generating component flow diagram: {e}")
        return """flowchart TD
            A[Input] --> B[Process]
            B --> C[Output]"""

def extract_mermaid_code(text: str) -> str:
    """
    Extract Mermaid code from text that might contain markdown formatting.
    
    Args:
        text: Text that might contain Mermaid code in markdown format
        
    Returns:
        Clean Mermaid code
    """
    mermaid_pattern = r"```mermaid\s*([\s\S]*?)\s*```"
    match = re.search(mermaid_pattern, text)
    
    if match:
        return match.group(1).strip()
    else:
        if "classDiagram" in text or "flowchart" in text:
            lines = text.split('\n')
            mermaid_lines = []
            in_diagram = False
            
            for line in lines:
                if "classDiagram" in line or "flowchart" in line:
                    in_diagram = True
                    mermaid_lines.append(line)
                elif in_diagram and line.strip() and not line.startswith("```"):
                    mermaid_lines.append(line)
                elif in_diagram and line.startswith("```"):
                    in_diagram = False
            
            return "\n".join(mermaid_lines)
        else:
            return text.replace("```mermaid", "").replace("```", "").strip()

def render_mermaid_to_png(mermaid_code: str, output_file: str) -> bool:
    """
    Render Mermaid diagram to PNG using the Mermaid CLI (if installed) or Mermaid.ink API.
    
    Args:
        mermaid_code: Mermaid diagram code
        output_file: Path to save the PNG file
        
    Returns:
        Boolean indicating success
    """
    if shutil.which('mmdc'):
        try:
            with tempfile.NamedTemporaryFile(suffix='.mmd', mode='w', delete=False) as temp_file:
                temp_file.write(mermaid_code)
                temp_file_path = temp_file.name
            
            result = subprocess.run(
                ['mmdc', '-i', temp_file_path, '-o', output_file, '-b', 'transparent'],
                capture_output=True,
                text=True,
                check=False
            )
            
            os.unlink(temp_file_path)
            
            if result.returncode == 0:
                print(f"Successfully generated PNG using Mermaid CLI: {output_file}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            pass
    
    try:
        encoded_mermaid = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        mermaid_url = f"https://mermaid.ink/img/{encoded_mermaid}"
        
        response = requests.get(mermaid_url, timeout=15)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Successfully generated PNG using Mermaid.ink: {output_file}")
            return True
        else:
            print(f"Mermaid.ink API request failed: {response.status_code}")
    except Exception as e:
        print(f"Failed to use Mermaid.ink API: {e}")
    
    if shutil.which('puppeteer-mermaid'):
        try:
            with tempfile.NamedTemporaryFile(suffix='.mmd', mode='w', delete=False) as temp_file:
                temp_file.write(mermaid_code)
                temp_file_path = temp_file.name
            
            result = subprocess.run(
                ['puppeteer-mermaid', '-i', temp_file_path, '-o', output_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            os.unlink(temp_file_path)
            
            if result.returncode == 0:
                print(f"Successfully generated PNG using puppeteer-mermaid: {output_file}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            pass
    
    try:
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        d.text((10, 10), "Mermaid rendering failed. Diagram code:", fill=(0, 0, 0), font=font)
        wrapped_text = "\n".join([mermaid_code[i:i+80] for i in range(0, len(mermaid_code), 80)])
        d.text((10, 50), wrapped_text, fill=(0, 0, 0), font=font)
        
        img.save(output_file)
        print(f"Created fallback image with code: {output_file}")
        return True
    except Exception as e:
        print(f"Failed to create fallback image: {e}")
        return False

def generate_architecture_diagram(classes: Dict[str, Any], output_file: str, 
                                 model, tokenizer) -> None:
    """
    Generate an architecture diagram and save it to a file.
    
    Args:
        classes: Dictionary of class information
        output_file: Path to save the output diagram
        model: Loaded LLM model
        tokenizer: Tokenizer for the LLM
    """
    mermaid_code = generate_mermaid_architecture_diagram(classes, model, tokenizer)
    mmd_file = output_file.replace('.png', '.mmd')
    with open(mmd_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print(f"Architecture diagram mermaid code saved to {mmd_file}")
    
    txt_file = output_file.replace('.png', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"```mermaid\n{mermaid_code}\n```")
    print(f"Architecture diagram description saved to {txt_file}")
    
    if render_mermaid_to_png(mermaid_code, output_file):
        print(f"Architecture diagram saved as PNG: {output_file}")
    else:
        print(f"Failed to render architecture diagram as PNG: {output_file}")

def generate_class_diagram(classes: Dict[str, Any], dependencies: Dict[str, Any],
                          output_file: str, model, tokenizer) -> None:
    """
    Generate a class diagram and save it to a file.
    
    Args:
        classes: Dictionary of class information
        dependencies: Dictionary of dependency information
        output_file: Path to save the output diagram
        model: Loaded LLM model
        tokenizer: Tokenizer for the LLM
    """
    mermaid_code = generate_mermaid_class_diagram(classes, dependencies, model, tokenizer)
    mmd_file = output_file.replace('.png', '.mmd')
    with open(mmd_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print(f"Class diagram mermaid code saved to {mmd_file}")
    
    txt_file = output_file.replace('.png', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"```mermaid\n{mermaid_code}\n```")
    print(f"Class diagram description saved to {txt_file}")
    
    if render_mermaid_to_png(mermaid_code, output_file):
        print(f"Class diagram saved as PNG: {output_file}")
    else:
        print(f"Failed to render class diagram as PNG: {output_file}")

def generate_component_flow_diagram(data_flow: Dict[str, Any], output_file: str,
                                   model, tokenizer) -> None:
    """
    Generate a component flow diagram and save it to a file.
    
    Args:
        data_flow: Dictionary of data flow information
        output_file: Path to save the output diagram
        model: Loaded LLM model
        tokenizer: Tokenizer for the LLM
    """
    mermaid_code = generate_mermaid_component_flow_diagram(data_flow, model, tokenizer)
    mmd_file = output_file.replace('.png', '.mmd')
    with open(mmd_file, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print(f"Component flow diagram mermaid code saved to {mmd_file}")
    
    txt_file = output_file.replace('.png', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"```mermaid\n{mermaid_code}\n```")
    print(f"Component flow diagram description saved to {txt_file}")
    
    if render_mermaid_to_png(mermaid_code, output_file):
        print(f"Component flow diagram saved as PNG: {output_file}")
    else:
        print(f"Failed to render component flow diagram as PNG: {output_file}")