"""
HTML Content Extractor - A utility for extracting meaningful content from HTML documents

INPUT:
- HTML content: Raw HTML or HTML embedded in JSON
- Current URL: Optional URL of the page being processed (used for resolving relative URLs)

OUTPUT:
- Content tree: A hierarchical structure of text and image nodes
- Image links: Extracted image URLs from the content
- Base64 images: Extracted base64-encoded images
- Formatted content: Text representation suitable for language model processing

The extractor intelligently identifies and filters out non-content elements like navigation,
footers, ads, and scripts, focusing on the main content of the page.
"""

from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse

class ContentNode:
    def __init__(self, type, content):
        self.type = type  # 'text' or 'image'
        self.content = content
        self.children = []
        
    def add_child(self, child):
        if (child.type == 'text' and self.children and 
            self.children[-1].type == 'text'):
            # Merge with previous text node
            self.children[-1].content += ' ' + child.content
        else:
            self.children.append(child)
    def __str__(self, level=0):
        indent = "  " * level
        result = f"{indent}{self.type}: {self.content}\n"
        for child in self.children:
            result += child.__str__(level + 1)
        return result

def detect_base_url(soup, current_url=None):
    """
    Detect the base URL for resolving relative URLs.
    
    Args:
        soup: BeautifulSoup object
        current_url: Current page URL
        
    Returns:
        Base URL as a string
    """
    if current_url:
        parsed = urlparse(current_url)
        # Only take the protocol and domain part as the base URL
        return f"{parsed.scheme}://{parsed.netloc}"
    # Then try to find base tag in HTML
    base_tag = soup.find('base')
    if base_tag and base_tag.get('href'):
        return base_tag['href']
    return None

def should_keep_element(element):
    """
    Determine if an element should be kept in the content tree.
    
    Args:
        element: HTML element
        
    Returns:
        Boolean indicating whether to keep the element
    """
    if not element:
        return False
        
    skip_tags = {
        'script', 'style', 'meta', 'link', 'noscript', 'iframe', 
        'header', 'footer', 'nav', 'button', 'input'
    }
    if hasattr(element, 'name') and element.name in skip_tags:
        return False
        
    skip_classes = {
        'navigation', 'menu', 'sidebar', 'footer', 'header', 'cookie', 
        'ad', 'banner', 'modal', 'popup'
    }
    
    if hasattr(element, 'attrs'):
        classes = element.get('class', [])
        if isinstance(classes, str):
            classes = [classes]
        if any(cls.lower() in skip_classes for cls in classes):
            return False
            
        element_id = element.get('id', '').lower()
        if any(term in element_id for term in skip_classes):
            return False
    
    return True

def extract_content(element, base_url=None):
    """
    Extract meaningful content from HTML elements.
    
    Args:
        element: HTML element
        base_url: Base URL for resolving relative URLs
        
    Returns:
        List of ContentNode objects
    """
    content = []
    
    if not should_keep_element(element):
        return content

    # Handle images
    if element.name == 'img':
        src = element.get('src', '')
        alt = element.get('alt', '')
        if src and base_url:
            # Convert relative URL to absolute URL
            if not src.startswith(('http://', 'https://', '//')):
                src = urljoin(base_url, src)
            content.append(ContentNode('image', f"[Image: {alt}] Source: {src}"))
        elif src:
            # If there's no base_url, still retain the original src
            content.append(ContentNode('image', f"[Image: {alt}] Source: {src}"))
        return content

    # Get text content
    if isinstance(element, str):
        text = element.strip()
        if text:
            content.append(ContentNode('text', text))
        return content

    # Process children
    if hasattr(element, 'children'):
        for child in element.children:
            if hasattr(child, 'name') or (isinstance(child, str) and child.strip()):
                content.extend(extract_content(child, base_url))

    return content

def merge_text_nodes(nodes):
    """
    Merge consecutive text nodes into single nodes.
    
    Args:
        nodes: List of ContentNode objects
        
    Returns:
        List of merged ContentNode objects
    """
    merged = []
    current_text = []
    
    for node in nodes:
        if node.type == 'text':
            current_text.append(node.content)
        else:
            if current_text:
                merged.append(ContentNode('text', ' '.join(current_text)))
                current_text = []
            merged.append(node)
            
    if current_text:
        merged.append(ContentNode('text', ' '.join(current_text)))
    
    return merged

def process_json_file(json_content):
    """
    Process HTML embedded in JSON file and extract meaningful content.
    
    Args:
        json_content: JSON string containing HTML content
        
    Returns:
        ContentNode: Root node of the processed content tree or None if processing fails
    """
    try:
        # Parse JSON content
        data = json.loads(json_content)

        current_url = data.get('url')
        
        # Get HTML content
        html_content = data.get('html')
        
        if not html_content:
            raise ValueError("No HTML content found in the JSON structure")
            return None
            
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to detect base URL from HTML
        base_url = detect_base_url(soup, current_url)

        
        # Extract all content
        content_nodes = extract_content(soup.body if soup.body else soup, base_url)
        
        # Merge consecutive text nodes
        merged_nodes = merge_text_nodes(content_nodes)
        
        # Create root node and add merged content
        root = ContentNode('root', 'Document Content')
        for node in merged_nodes:
            root.add_child(node)
        
        return root
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
    
def process_html_file(html_content, current_url=None):
    """
    Process HTML content and extract meaningful content.
    
    Args:
        html_content: Raw HTML content
        current_url: URL of the page being processed
        
    Returns:
        ContentNode: Root node of the processed content tree
    """
    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Detect base URL
        base_url = detect_base_url(soup, current_url)
        
        # Extract all content
        content_nodes = extract_content(soup.body if soup.body else soup, base_url)
        
        # Merge consecutive text nodes
        merged_nodes = merge_text_nodes(content_nodes)
        
        # Create root node and add merged content
        root = ContentNode('root', 'Document Content')
        for node in merged_nodes:
            root.add_child(node)
        
        return root
        
    except Exception as e:
        print(f"Error processing HTML: {str(e)}")
        return None

def extract_image_links(root_node):
    """
    Extract image URLs from the content tree.
    
    Args:
        root_node: Root ContentNode of the content tree
        
    Returns:
        List of image URLs
    """
    urls = []
    
    def _extract_from_node(node):
        if node is None:
            return
        if node.type == 'image':
            # Directly extract URL, avoid string parsing
            src_match = re.search(r'Source:\s*(\S+)', node.content)
            if src_match:
                url = src_match.group(1)
                if url.startswith(('http://', 'https://', '//')):
                    urls.append(url)
        
        for child in node.children:
            _extract_from_node(child)
    
    _extract_from_node(root_node)
    return urls

def format_for_llm(root_node):
    """
    Format content for language model processing.
    
    Args:
        root_node: Root ContentNode of the content tree
        
    Returns:
        Formatted string suitable for LLM input
    """
    def _recursive_format(node, level=0):
        output = []
        
        if level > 0 and node.content != 'Document Content':
            if node.type == 'text':
                output.append(node.content)
            elif node.type == 'image':
                if '[Image:' in node.content:
                    alt_start = node.content.find('[Image:') + 7
                    alt_end = node.content.find(']', alt_start)
                    alt_text = node.content[alt_start:alt_end].strip()
                    
                    src_start = node.content.find('Source:') + 7
                    src = node.content[src_start:].strip()
                    
                    if alt_text:
                        output.append(f"<image alt='{alt_text}' src='{src}'></image>")
                    else:
                        output.append(f"<image src='{src}'></image>")
        
        for child in node.children:
            child_content = _recursive_format(child, level + 1)
            if child_content:
                output.extend(child_content)
        
        return output

    content_parts = _recursive_format(root_node)
    formatted_text = "\n\n".join(filter(None, content_parts))
    
    return f"<document>\n{formatted_text}\n</document>"

def extract_base64(root_node):
    """
    Extract base64-encoded images from the content tree.
    
    Args:
        root_node: Root ContentNode of the content tree
        
    Returns:
        List of base64 image data strings
    """
    base64_images = []
    
    def traverse_node(node):
        if node is None:
            return
            
        # Check if the current node is an image node
        if node.type == 'image':
            # Extract src from content
            src = None
            if 'Source:' in node.content:
                src = node.content.split('Source:')[1].strip()
            
            # If it's a base64 image, add to the list
            if src and src.startswith('data:image'):
                base64_images.append(src)
        
        # Recursively process all child nodes
        for child in node.children:
            traverse_node(child)
    
    # Start traversing the node tree
    traverse_node(root_node)
    
    return base64_images