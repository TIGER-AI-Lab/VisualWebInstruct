"""
Q&A Markdown Parser and Formatter - A utility for parsing and formatting question-answer content with image support

INPUT:
- Markdown text containing Q&A content with image references
- LLM response objects containing text to be extracted
- Output path for saving parsed Q&A data

OUTPUT:
- Structured QAItem objects with separate handling for question and solution images
- JSON files containing parsed Q&A data
- Formatted markdown text from structured Q&A data

The parser intelligently handles image URLs, filters out placeholder or "Not found" content,
and maintains the structure of questions and solutions with their associated images.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
import re
import json

@dataclass
class QAItem:
    """Data class for storing question-answer items with enhanced image support"""
    question: str
    question_images: List[str]  # Images associated with the question
    solution: str
    solution_images: List[str]  # Images within the solution

def extract_content_from_response(response) -> str:
    """
    Extract the actual content from LLM response structure.
    
    Args:
        response: LLM response object
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If text extraction fails
    """
    try:
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(len(response.candidates[0].content.parts))
        raise ValueError(f"Failed to extract text from response: {str(e)}")

def parse_markdown_qa(markdown_text: str) -> List[QAItem]:
    """
    Parse markdown formatted Q&A text into structured data.
    Handles both question images and solution images separately.
    Skips items where question is "Not found" and doesn't store "Not found" URLs.
    
    Args:
        markdown_text: Markdown formatted text containing Q&A content
        
    Returns:
        List of QAItem objects
    """
    # Split into individual questions
    pattern = r'\*\*Question \d+:\*\*'
    sections = re.split(pattern, markdown_text)
    
    # Remove empty first section if exists
    if sections and not sections[0].strip():
        sections = sections[1:]
    
    qa_items = []
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        question = ""
        question_images = []
        solution = ""
        solution_images = []
        
        current_section = None
        current_text = []
        
        lines = section.split('\n')
        for line in lines:
            line_strip = line.strip()
            
            # Section detection
            if '**Images:**' in line_strip:
                if current_section == 'question':
                    # Save accumulated question text
                    question = '\n'.join(current_text).strip()
                    current_text = []
                current_section = 'question_images'
            elif '**Solution:**' in line_strip:
                current_section = 'solution'
                current_text = []
            elif '**Images in Solution:**' in line_strip:
                if current_section == 'solution':
                    # Save accumulated solution text
                    solution = '\n'.join(current_text).strip()
                    current_text = []
                current_section = 'solution_images'
            elif not current_section and line_strip:
                current_section = 'question'
                current_text.append(line_strip)
            else:
                # Process content based on current section
                if current_section == 'question':
                    current_text.append(line_strip)
                elif current_section in ['question_images', 'solution_images']:
                    if line_strip.startswith('*'):
                        image_url = line_strip.strip('* []')
                        if (image_url and 
                            image_url.lower() != 'first image url if available' and 
                            image_url.lower() != 'second image url if available' and
                            image_url.lower() != 'not found'):  # Filter out "not found"
                            if current_section == 'question_images':
                                question_images.append(image_url)
                            else:
                                solution_images.append(image_url)
                elif current_section == 'solution':
                    if line_strip:
                        current_text.append(line)
        
        # Save final solution text if not already saved
        if current_text and current_section == 'solution':
            solution = '\n'.join(current_text).strip()
        
        # Create QA item if we have a valid question (not "Not found")
        if question and question.lower() != "not found":
            qa_items.append(QAItem(
                question=question,
                question_images=question_images,  # No longer adding default "Not found"
                solution=solution if solution else "Not found",
                solution_images=solution_images  # No longer adding default "Not found"
            ))
    
    return qa_items

def format_qa_output(qa_items: List[QAItem]) -> str:
    """
    Format QA items back into markdown format with separate image sections.
    
    Args:
        qa_items: List of QAItem objects
        
    Returns:
        Formatted markdown text
    """
    output = []
    
    for i, item in enumerate(qa_items, 1):
        output.append(f"**Question {i}:**")
        output.append(item.question)
        
        output.append("**Images:**")
        if item.question_images:  # Only add when there are images
            for img in item.question_images:
                output.append(f"* {img}")
        else:
            output.append("* No images available")
            
        output.append("\n**Solution:**")
        output.append(item.solution)
        
        output.append("\n**Images in Solution:**")
        if item.solution_images:  # Only add when there are images
            for img in item.solution_images:
                output.append(f"* {img}")
        else:
            output.append("* No images available")
            
        output.append("\n")  # Add extra newline between QA sets

    return "\n".join(output)

def save_qa_data(qa_items: List[QAItem], output_path: Union[str, Path]) -> None:
    """
    Save QA items to JSON file with enhanced image support.
    Only saves items where question is not "Not found".
    
    Args:
        qa_items: List of QAItem objects
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter out entries where question is "Not found"
    valid_items = [item for item in qa_items if item.question.lower() != "not found"]
    
    if not valid_items:
        return  # If there are no valid entries, don't create the file
    
    qa_data = [
        {
            'question': item.question,
            'question_images': item.question_images,  # List may now be empty
            'solution': item.solution,
            'solution_images': item.solution_images  # List may now be empty
        }
        for item in valid_items
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)