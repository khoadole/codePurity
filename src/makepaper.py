#!/usr/bin/env python3
"""
Paper Generation Module for Paper with Mermaid Diagrams

This script creates a complete research paper document based on code analysis results,
generating text content, Mermaid diagrams, and formatting the final document.
"""

import argparse
import json
import os
import math
import re
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import networkx as nx
import openai
from utils import (
    load_json, save_json, create_directory,
    format_markdown, generate_tex_preamble, generate_tex_closing,
    extract_metrics_summary, extract_complexity_summary
)
from mermaid_utils import (
    generate_architecture_diagram, generate_class_diagram, generate_component_flow_diagram
)

class PaperGenerator:
    """Generates a complete research paper from code analysis."""
    
    def __init__(self, output_dir: str, gpt_version: str = "gpt-3.5-turbo"):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        create_directory(self.figures_dir)
        
        self.gpt_version = gpt_version
        self.openai_client = openai.OpenAI(api_key = os.environ["OPENAI_API_KEY"])
        
        # Load analysis results
        self.analysis_result = load_json(os.path.join(output_dir, "analysis_result.json"))
        self.paper_plan = load_json(os.path.join(output_dir, "paper_plan.json"))
        
    def generate_figures(self) -> Dict[str, str]:
        """
        Generate all figures for the paper using Mermaid diagrams.
        """
        figure_paths = {}
        
        # Generate architecture diagram
        architecture_path = os.path.join(self.figures_dir, "architecture_diagram")
        generate_architecture_diagram(
            self.analysis_result["complexity"]["classes"],
            architecture_path + ".png",  # Still pass .png for compatibility
            self.openai_client,
            self.gpt_version
        )
        figure_paths["architecture"] = architecture_path + ".mmd"
        
        # Generate class diagram
        class_diagram_path = os.path.join(self.figures_dir, "class_diagram")
        generate_class_diagram(
            self.analysis_result["complexity"]["classes"],
            self.analysis_result["dependencies"],
            class_diagram_path + ".png",  # Still pass .png for compatibility
            self.openai_client,
            self.gpt_version
        )
        figure_paths["class_diagram"] = class_diagram_path + ".mmd"
        
        # Generate component flow diagram
        component_flow_path = os.path.join(self.figures_dir, "component_flow")
        generate_component_flow_diagram(
            self.analysis_result["data_flow"],
            component_flow_path + ".png",  # Still pass .png for compatibility
            self.openai_client,
            self.gpt_version
        )
        figure_paths["component_flow"] = component_flow_path + ".mmd"
        
        return figure_paths
    
    def generate_abstract(self) -> str:
        """
        Generate paper abstract using GPT.
        """
        metrics = extract_metrics_summary(self.analysis_result["metrics"])
        complexity = extract_complexity_summary(self.analysis_result["complexity"])
        paper_name = self.paper_plan["paper_name"]
        prompt = f"""
        Write an abstract for a research paper analyzing the implementation of {paper_name}.
        The code has the following characteristics:
        
        {metrics}
        {complexity}
        
        The paper analyzes the architecture, implementation details, and code quality.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "You are an expert AI researcher who writes clear, concise academic abstracts."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            abstract = response.choices[0].message.content.strip()
            return abstract
            
        except Exception as e:
            print(f"Error generating abstract: {e}")
            return "Abstract generation failed. Please check your code analysis results and try again."
    
    def generate_introduction(self) -> str:
        """
        Generate introduction section using GPT.
        """
        paper_name = self.paper_plan["paper_name"]
        
        prompt = f"""
        Write an introduction section for a research paper analyzing the implementation of {paper_name}.
        
        Include:
        1. Background and importance of {paper_name}
        2. Motivation for analyzing this particular implementation
        3. Overview of the paper structure
        4. Main contributions
        
        Keep it academic, concise, and focused on code analysis rather than the model's performance.
        
        IMPORTANT: Do not use any markdown formatting like **bold** or *italic* in your response as this will be directly inserted into a LaTeX document. Do not conclude this part.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "You are an expert AI researcher who writes clear, academic papers."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            introduction = response.choices[0].message.content.strip()
            return introduction
            
        except Exception as e:
            print(f"Error generating introduction: {e}")
            return "Introduction generation failed. Please check your code analysis results and try again."
    
    def generate_architecture_section(self) -> str:
        """
        Generate architecture and implementation details section using GPT.
        """
        # Extract key information from analysis
        classes = list(self.analysis_result["complexity"]["classes"].keys())
        neural_network_info = self.analysis_result["algorithms"]["neural_network"]
        attention_info = self.analysis_result["algorithms"]["attention_mechanism"]
        paper_name = self.paper_plan["paper_name"]
        
        # Read Mermaid diagrams if available
        architecture_diagram_path = os.path.join(self.figures_dir, "architecture_diagram.mmd")
        class_diagram_path = os.path.join(self.figures_dir, "class_diagram.mmd")
        component_flow_path = os.path.join(self.figures_dir, "component_flow.mmd")
        
        architecture_diagram = ""
        class_diagram = ""
        component_flow = ""
        
        try:
            if os.path.exists(architecture_diagram_path):
                with open(architecture_diagram_path, 'r', encoding='utf-8') as f:
                    architecture_diagram = f.read()
        except Exception as e:
            print(f"Error reading architecture diagram: {e}")
            
        try:
            if os.path.exists(class_diagram_path):
                with open(class_diagram_path, 'r', encoding='utf-8') as f:
                    class_diagram = f.read()
        except Exception as e:
            print(f"Error reading class diagram: {e}")
            
        try:
            if os.path.exists(component_flow_path):
                with open(component_flow_path, 'r', encoding='utf-8') as f:
                    component_flow = f.read()
        except Exception as e:
            print(f"Error reading component flow diagram: {e}")
        
        diagram_info = f"""
        Architecture Diagram Overview:
        Contains classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}
        
        Class Diagram Overview: 
        Shows relationships between classes including inheritance and dependencies.
        
        Component Flow Overview:
        Illustrates the data processing pipeline from input to output.
        """
        
        prompt = f"""
        Write a detailed architecture and implementation section for a research paper analyzing the implementation of {paper_name}.
        
        Key components include:
        - Classes: {classes}
        - Neural network elements: {neural_network_info}
        - Attention mechanism: {attention_info}
        
        {diagram_info}
        
        Focus on:
        1. Overall architecture design
        2. Implementation of the multi-head attention mechanism if it exists (else don't write about it)
        3. Feed-forward networks and layer normalization if it exists (else don't write about it)
        4. Data flow through the model
        
        Include references to the figures (Figure 1: Architecture Diagram, Figure 2: Class Diagram, Figure 3: Component Flow).
        Write in an academic style with technical details.
        
        IMPORTANT: Do not use any markdown formatting like **bold** or *italic* in your response as this will be directly inserted into a LaTeX document. Do not write conclusion here.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "You are an expert AI researcher who writes technical architecture descriptions."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            architecture_section = response.choices[0].message.content.strip()
            return architecture_section
            
        except Exception as e:
            print(f"Error generating architecture section: {e}")
            return "Architecture section generation failed. Please check your code analysis results and try again."
    
    def generate_code_quality_section(self) -> str:
        """
        Generate code quality analysis section using GPT.
        """
        code_quality = self.analysis_result["code_quality"]
        paper_name = self.paper_plan["paper_name"]
        
        # Format the quality metrics
        quality_metrics = "\n".join([
            f"- Docstring coverage: {code_quality.get('docstring_coverage', 0):.2f}",
            f"- Naming consistency: {code_quality.get('naming_consistency', 0):.2f}",
            f"- Average function length: {code_quality.get('average_function_length', 0):.1f} lines",
            f"- Complexity ratio: {code_quality.get('complexity_ratio', 0):.1f}",
            f"- Overall quality score: {code_quality.get('overall_quality', 0):.2f}"
        ])
        
        prompt = f"""
        Write a code quality analysis section for a research paper evaluating implementation of {paper_name}.
        The code quality metrics are:
        
        {quality_metrics}
        
        The dominant naming convention is {code_quality.get('dominant_naming_convention', 'unknown')}.
        
        Focus on:
        1. Code readability and maintainability
        2. Documentation quality
        3. Adherence to Python best practices
        4. Areas for potential improvement
        
        Be analytical and provide concrete suggestions for improvement.
        
        IMPORTANT: Do not use any markdown formatting like **bold** or *italic* in your response as this will be directly inserted into a LaTeX document. Do not write conclusion for this part.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "You are an expert software engineer who specializes in code quality analysis."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            code_quality_section = response.choices[0].message.content.strip()
            return code_quality_section
            
        except Exception as e:
            print(f"Error generating code quality section: {e}")
            return "Code quality section generation failed. Please check your code analysis results and try again."
    
    def generate_conclusion(self) -> str:
        """
        Generate conclusion section using GPT.
        """
        # Extract key metrics for the conclusion
        metrics = self.analysis_result["metrics"]
        code_quality = self.analysis_result["code_quality"]
        paper_name = self.paper_plan["paper_name"]
        
        prompt = f"""
        Write a conclusion section for a research paper analyzing the implementation of {paper_name}.
        
        The codebase has:
        - {metrics.get('class_count', 0)} classes
        - {metrics.get('function_count', 0)} functions
        - Overall quality score of {code_quality.get('overall_quality', 0):.2f}/1.0
        
        Include:
        1. Summary of key findings from the architecture and code quality analysis
        2. Strengths and weaknesses of the implementation
        3. Recommendations for improvement
        4. Final thoughts on the implementation's suitability for production or research
        
        Keep it concise, balanced, and provide meaningful insights.
        
        IMPORTANT: Do not use any markdown formatting like **bold** or *italic* in your response as this will be directly inserted into a LaTeX document.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_version,
                messages=[
                    {"role": "system", "content": "You are an expert AI researcher who writes impactful paper conclusions."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            conclusion = response.choices[0].message.content.strip()
            return conclusion
            
        except Exception as e:
            print(f"Error generating conclusion: {e}")
            return "Conclusion generation failed. Please check your code analysis results and try again."
    
    def generate_paper(self) -> Dict[str, str]:
        """
        Generate the complete paper with all sections.
        """
        print("Generating diagrams...")
        figure_paths = self.generate_figures()
        
        print("Generating abstract...")
        abstract = self.generate_abstract()
        
        print("Generating introduction...")
        introduction = self.generate_introduction()
        
        print("Generating architecture section...")
        architecture_section = self.generate_architecture_section()
        
        print("Generating code quality section...")
        code_quality_section = self.generate_code_quality_section()
        
        print("Generating conclusion...")
        conclusion = self.generate_conclusion()
        
        # Compile the paper
        paper = {
            "title": f"Analysis of {self.paper_plan['paper_name']} Implementation",
            "abstract": abstract,
            "introduction": introduction,
            "architecture": architecture_section,
            "code_quality": code_quality_section,
            "conclusion": conclusion,
            "figures": figure_paths
        }
        
        return paper
    
    def save_paper_markdown(self, paper: Dict[str, str]) -> str:
        """
        Save the paper in Markdown format with Mermaid diagrams.
        """
        markdown_path = os.path.join(self.output_dir, "paper.md")
        
        # Read mermaid code from files for diagrams
        architecture_mermaid = ""
        class_diagram_mermaid = ""
        component_flow_mermaid = ""
        
        try:
            with open(paper['figures']['architecture'], 'r', encoding='utf-8') as f:
                architecture_mermaid = f.read()
        except:
            architecture_mermaid = "classDiagram\n"
        
        try:
            with open(paper['figures']['class_diagram'], 'r', encoding='utf-8') as f:
                class_diagram_mermaid = f.read()
        except:
            class_diagram_mermaid = "classDiagram\n"
        
        try:
            with open(paper['figures']['component_flow'], 'r', encoding='utf-8') as f:
                component_flow_mermaid = f.read()
        except:
            component_flow_mermaid = "flowchart TD\n    A-->B"
        
        # Create markdown content with mermaid diagrams
        markdown_content = f"""# {paper['title']}

## Abstract

{paper['abstract']}

## 1. Introduction

{paper['introduction']}

## 2. Architecture and Implementation

{paper['architecture']}

### Figure 1: Architecture Diagram

```mermaid
{architecture_mermaid}
```
*Figure 1: Architecture diagram of the {self.paper_plan['paper_name']} implementation*

### Figure 2: Class Diagram

```mermaid
{class_diagram_mermaid}
```
*Figure 2: Class diagram showing relationships between components*

### Figure 3: Component Flow Diagram

```mermaid
{component_flow_mermaid}
```
*Figure 3: Component flow diagram illustrating data processing pipeline*

## 3. Code Quality Analysis

{paper['code_quality']}

## 4. Conclusion

{paper['conclusion']}
"""
        
        # Save markdown file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Paper saved as markdown at {markdown_path}")
        return markdown_path
    
    def clean_markdown_for_latex(self, text: str) -> str:
        """
        Cleans markdown formatting and converts to LaTeX formatting.
        """
        # Replace Markdown bold with LaTeX bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
        
        # Replace Markdown italic with LaTeX italic
        text = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', text)
        
        # Replace Markdown headers with LaTeX sections/subsections
        text = re.sub(r'^#\s+(.+)$', r'\\section{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'\\subsection{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^###\s+(.+)$', r'\\subsubsection{\1}', text, flags=re.MULTILINE)
        
        # Replace Markdown lists with LaTeX lists
        text = re.sub(r'^-\s+(.+)$', r'\\item \1', text, flags=re.MULTILINE)
        
        # Escape special LaTeX characters
        text = text.replace('%', r'\%')
        text = text.replace('&', r'\&')
        text = text.replace('#', r'\#')
        text = text.replace('_', r'\_')
        
        return text
    
    def save_paper_tex(self, paper: Dict[str, str]) -> str:
        """
        Save the paper in LaTeX format.
        """
        tex_path = os.path.join(self.output_dir, "paper.tex")
        
        # Clean markdown formatting for LaTeX
        clean_introduction = self.clean_markdown_for_latex(paper['introduction'])
        clean_architecture = self.clean_markdown_for_latex(paper['architecture'])
        clean_code_quality = self.clean_markdown_for_latex(paper['code_quality'])
        clean_conclusion = self.clean_markdown_for_latex(paper['conclusion'])
        clean_abstract = self.clean_markdown_for_latex(paper['abstract'])
        
        # Create LaTeX content
        tex_content = generate_tex_preamble(paper['title'])
        
        # Abstract
        tex_content += """
\\begin{abstract}
""" + clean_abstract + """
\\end{abstract}

\\section{Introduction}
""" + clean_introduction + """

\\section{Architecture and Implementation}
""" + clean_architecture + """

\\begin{figure}[h]
\\centering
\\includegraphics[width=0.95\\columnwidth,keepaspectratio]{figures/architecture_diagram.png}
\\caption{Architecture diagram of the """ + self.paper_plan['paper_name'] + """ implementation}
\\label{fig:architecture}
\\end{figure}

\\begin{figure}[h]
\\centering
\\includegraphics[width=0.95\\columnwidth,keepaspectratio]{figures/class_diagram.png}
\\caption{Class diagram showing relationships between components}
\\label{fig:class_diagram}
\\end{figure}

\\begin{figure}[h]
\\centering
\\includegraphics[width=0.95\\columnwidth,keepaspectratio]{figures/component_flow.png}
\\caption{Component flow diagram illustrating data processing pipeline}
\\label{fig:component_flow}
\\end{figure}

\\section{Code Quality Analysis}
""" + clean_code_quality + """

\\section{Conclusion}
""" + clean_conclusion + """

""" + generate_tex_closing()
        
        # Save LaTeX file
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(tex_content)
        
        print(f"Paper saved as LaTeX at {tex_path}")
        return tex_path
    
    def save_paper_pdf(self, tex_path: str) -> str:
        """
        Convert LaTeX to PDF using pdflatex (if available).
        """
        pdf_path = os.path.join(self.output_dir, "paper.pdf")
        
        try:
            # Change to output directory to ensure figure paths are correct
            original_dir = os.getcwd()
            os.chdir(self.output_dir)
            
            # Run pdflatex twice to ensure references are resolved
            os.system(f"pdflatex -interaction=nonstopmode {os.path.basename(tex_path)}")
            os.system(f"pdflatex -interaction=nonstopmode {os.path.basename(tex_path)}")
            
            # Return to original directory
            os.chdir(original_dir)
            
            print(f"Paper saved as PDF at {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            print("PDF generation failed. Please compile the LaTeX file manually.")
            return None

def main():
    parser = argparse.ArgumentParser(description="Generate a research paper from code analysis results.")
    parser.add_argument("--output_dir", required=True, help="Directory with analysis results and for output")
    parser.add_argument("--gpt_version", default="gpt-3.5-turbo", help="GPT model version to use")
    args = parser.parse_args()
    
    # Create paper generator
    generator = PaperGenerator(
        output_dir=args.output_dir,
        gpt_version=args.gpt_version
    )
    
    # Generate paper content
    paper = generator.generate_paper()
    
    # Save paper in different formats
    markdown_path = generator.save_paper_markdown(paper)
    tex_path = generator.save_paper_tex(paper)
    
    # Try to generate PDF if LaTeX is installed
    generator.save_paper_pdf(tex_path)
    
    print("Paper generation completed successfully!")

if __name__ == "__main__":
    main()