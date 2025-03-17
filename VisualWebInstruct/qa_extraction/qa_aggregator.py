"""
QA Data Aggregator - A tool for aggregating and processing question-answer data with image handling

INPUT:
- source_dir: Directory containing QA data files (JSON format)
- images_dir: Directory containing downloaded images
- url_mapping_file: JSON file mapping file IDs to source URLs
- min_image_size: Minimum dimensions for valid images
- remove_animated_gif: Boolean flag to control removal of animated GIFs

OUTPUT:
- Aggregated JSON file containing processed QA data
- Converted image files (SVG to PNG, static GIF to PNG)
- Detailed logs of processing operations and statistics
- Summary statistics of processing results

The aggregator processes each QA item, handles images (converting formats as needed),
verifies image dimensions, manages base64 encoded images, and consolidates all data
into a single structured JSON file for further use.
"""

from pathlib import Path
import json
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging
from datetime import datetime
from PIL import Image
import os
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler
from cairosvg import svg2png
import shutil
import hashlib
from urllib.parse import unquote
import uuid
import base64
from io import BytesIO

class QADataAggregator:
    def __init__(self, 
                 source_dir: str = "./downloaded_qa",
                 images_dir: str = "downloaded_html/images",
                 output_file: str = "aggregated_qa_data.json",
                 log_file: str = "aggregator.log",
                 min_image_size: Tuple[int, int] = (100, 100),
                 remove_animated_gif: bool = True,
                 url_mapping_file: str = "merged_urls.json"):
        """
        Initialize the QA data aggregator.
        
        Args:
            source_dir: Path to directory containing QA data files
            images_dir: Path to directory containing downloaded images
            output_file: Path to save the aggregated data
            log_file: Path for log files
            min_image_size: Minimum dimensions for valid images
            remove_animated_gif: Whether to remove animated GIFs
            url_mapping_file: Path to JSON file mapping file IDs to source URLs
        """
        self.source_dir = Path(source_dir)
        self.images_dir = Path(images_dir)
        self.output_file = Path(output_file)
        self.min_image_size = min_image_size
        self.remove_animated_gif = remove_animated_gif
        self.url_mapping_file = Path(url_mapping_file)
        self.setup_logging(log_file)


        # Load URL mapping
        self.url_map = self.load_url_mapping()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'original_qa_count': 0,
            'final_qa_count': 0,
            'qa_with_local_images': 0,
            'total_images': 0,
            'missing_images': 0,
            'small_images': 0,
            'valid_images': 0,
            'gif_images': 0,
            'animated_gifs': 0,
            'animated_gifs_removed': 0,
            'static_gifs': 0,
            'static_gif_converted': 0,
            'static_gif_conversion_failed': 0,
            'svg_images': 0,
            'svg_converted': 0,
            'svg_conversion_failed': 0,
            'qa_with_images': 0,
            'base64_images': 0,      
            'latex_converted': 0,
            'urls_mapped': 0,  
            'urls_not_found': 0,
        }
    
    def setup_logging(self, log_file: str):
        """Set up the logging system"""
        self.logger = logging.getLogger('QADataAggregator')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create error log handler
        error_file = log_file.replace('.log', '_error.log')
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.WARNING)
        self.logger.addHandler(error_handler)
        
        # Create success log handler
        success_file = log_file.replace('.log', '_success.log')
        success_handler = RotatingFileHandler(
            success_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        success_handler.setFormatter(formatter)
        success_handler.setLevel(logging.INFO)
        success_handler.addFilter(lambda record: record.levelno == logging.INFO)
        self.logger.addHandler(success_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

    def load_url_mapping(self) -> Dict[str, str]:
        """Load URL mapping file"""
        url_map = {}
        try:
            with open(self.url_mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    file_name = item['file_name']
                    url_map[file_name] = item['url']
            self.logger.info(f"Successfully loaded {len(url_map)} URL mappings")
            return url_map
        except Exception as e:
            self.logger.error(f"Failed to load URL mapping file: {str(e)}")
            return {}
    

    def get_image_filename(self, url: str, max_length: int = 100) -> str:
        """
        Extract image filename from URL and handle non-ASCII characters
        
        Args:
            url: Image URL
            max_length: Maximum allowed length for filenames
            
        Returns:
            Processed filename
        """
        try:
            # URL decode
            decoded_url = unquote(url)
            
            # Get base filename
            filename = os.path.basename(urlparse(decoded_url).path)
            
            # Handle non-ASCII characters
            import unicodedata
            # Convert filename to ASCII characters, remove unsupported characters
            safe_filename = "".join(
                char for char in unicodedata.normalize('NFKD', filename)
                if char.isascii() and (char.isalnum() or char in '._-')
            )
            
            # If conversion results in empty string, use URL hash as filename
            if not safe_filename:
                hash_str = hashlib.md5(url.encode()).hexdigest()[:16]
                ext = os.path.splitext(filename)[1] or '.unknown'
                safe_filename = f"img_{hash_str}{ext}"
            
            # Handle long filenames
            if len(safe_filename) > max_length:
                name, ext = os.path.splitext(safe_filename)
                hash_str = hashlib.md5(safe_filename.encode()).hexdigest()[:8]
                truncated_name = name[:32] + "_" + hash_str
                safe_filename = truncated_name + ext
            
            return safe_filename
            
        except Exception as e:
            # If processing fails, return a hash-based filename
            self.logger.error(f"Failed to process filename {url}: {str(e)}")
            hash_str = hashlib.md5(url.encode()).hexdigest()[:16]
            return f"img_{hash_str}.unknown"
        
    def find_image_folder(self, json_path: Path) -> Path:
        """Find folder containing images"""
        try:
            relative_path = json_path.relative_to(self.source_dir).parent
            return self.images_dir / relative_path
        except Exception as e:
            self.logger.error(f"Failed to find image folder: {str(e)}")
            return self.images_dir

    def find_image_in_folder(self, image_filename: str, json_path: Path) -> Optional[Path]:
        """
        Find image in corresponding folder with flexible matching
        
        Args:
            image_filename: Image filename to find
            json_path: Path to JSON file
            
        Returns:
            Path to found image or None if not found
        """
        try:
            relative_path = json_path.relative_to(self.source_dir).parent
            image_folder = self.images_dir / relative_path
            
            if not image_filename:
                return None
                
            # 1. Direct match
            direct_match = image_folder / image_filename
            if direct_match.exists():
                return direct_match
                
            # 2. Get original filename (without parameters)
            original_filename = self.get_original_filename(image_filename)
            if original_filename:
                original_match = image_folder / original_filename
                if original_match.exists():
                    return original_match
            
            # 3. Fuzzy matching
            try:
                base_name = os.path.splitext(original_filename or image_filename)[0]
                # Remove all non-alphanumeric characters
                clean_base = ''.join(c for c in base_name if c.isalnum())
                
                for file_path in image_folder.glob('*'):
                    current_base = os.path.splitext(file_path.name)[0]
                    clean_current = ''.join(c for c in current_base if c.isalnum())
                    
                    # Check if they contain the same key parts
                    if clean_base and clean_current and (
                        clean_base in clean_current or 
                        clean_current in clean_base
                    ):
                        self.logger.info(f"Fuzzy match successful: {image_filename} -> {file_path.name}")
                        return file_path
                        
            except Exception as e:
                self.logger.debug(f"Fuzzy matching failed: {str(e)}")
            
            self.logger.warning(f"Image not found: {image_filename} in directory {image_folder}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find image: {str(e)}")
            return None

    def is_animated_gif(self, image_path: Path) -> bool:
        """
        Check if a GIF is animated
        
        Args:
            image_path: Path to GIF file
            
        Returns:
            True if animated, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                try:
                    img.seek(1)
                    return True
                except EOFError:
                    return False
        except Exception as e:
            self.logger.error(f"Failed to check GIF animation {image_path}: {str(e)}")
            return False

    def convert_svg_to_png(self, svg_path: Path) -> Optional[Path]:
        """
        Convert SVG to PNG format
        
        Args:
            svg_path: Path to SVG file
            
        Returns:
            Path to converted PNG file or None if conversion fails
        """
        try:
            png_path = svg_path.with_suffix('.png')
            
            if png_path.exists():
                return png_path

            png_path.parent.mkdir(parents=True, exist_ok=True)
            
            svg2png(
                url=str(svg_path),
                write_to=str(png_path),
                output_width=1024,
                output_height=1024
            )
            
            # Delete original SVG file
            svg_path.unlink()
            
            self.logger.info(f"Successfully converted SVG to PNG: {svg_path} -> {png_path}")
            self.stats['svg_converted'] += 1
            return png_path
            
        except Exception as e:
            self.logger.error(f"SVG conversion failed {svg_path}: {str(e)}")
            self.stats['svg_conversion_failed'] += 1
            return None
    

    def find_most_similar_base64(self, target_str: str, json_path: Path) -> Optional[str]:
        """
        Find the most similar complete base64 string
        
        Args:
            target_str: Target base64 string
            json_path: JSON file path
            
        Returns:
            Most similar base64 string found, or None if not found
        """
        try:
            # Construct base64.json path (in same directory as json_path)
            base64_json = json_path / 'base64.json'
            if not base64_json.exists():
                self.logger.warning(f"base64.json doesn't exist: {base64_json}")
                return None
                
            # Read base64.json
            with open(base64_json, 'r', encoding='utf-8') as f:
                base64_list = json.load(f)
                
            if not base64_list:
                return None
                
            # Get data part of target string
            target_data = target_str.split(',')[1] if ',' in target_str else target_str
            
            # Calculate similarity and find best match
            best_match = None
            best_similarity = 0
            
            for base64_str in base64_list:
                # Get data part of comparison string
                compare_data = base64_str.split(',')[1] if ',' in base64_str else base64_str
                
                # Calculate similarity (using length of longest common substring as metric)
                similarity = self._compute_similarity(target_data, compare_data)
                
                # Update best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = base64_str
                
            return best_match
            
        except Exception as e:
            self.logger.error(f"Failed to find base64 string: {str(e)}")
            return None

    def _compute_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Use length of shorter string as denominator
            min_len = min(len(str1), len(str2))
            if min_len == 0:
                return 0
                
            # Find longest common prefix
            common_len = 0
            for i in range(min_len):
                if str1[i] == str2[i]:
                    common_len += 1
                else:
                    break
                    
            return common_len / min_len
            
        except Exception:
            return 0

    def convert_static_gif_to_png(self, gif_path: Path) -> Optional[Path]:
        """
        Convert static GIF to PNG format
        
        Args:
            gif_path: Path to GIF file
            
        Returns:
            Path to converted PNG file or None if conversion fails
        """
        try:
            png_path = gif_path.with_suffix('.png')
            with Image.open(gif_path) as gif:
                if self.is_animated_gif(gif_path):
                    return None
                    
                if gif.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', gif.size, (255, 255, 255))
                    background.paste(gif, mask=gif.split()[-1])
                    background.save(png_path, 'PNG')
                else:
                    gif.convert('RGB').save(png_path, 'PNG')
                
                # Delete original GIF file
                gif_path.unlink()
                
                self.logger.info(f"Successfully converted static GIF to PNG: {gif_path} -> {png_path}")
                self.stats['static_gif_converted'] += 1
                return png_path
                
        except Exception as e:
            self.logger.error(f"Failed to convert static GIF {gif_path}: {str(e)}")
            self.stats['static_gif_conversion_failed'] += 1
            return None

    def check_image_size(self, image_path: Path) -> bool:
        """
        Check if image resolution meets requirements
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image meets size requirements, False otherwise
        """
        try:
            if image_path.suffix.lower() == '.svg':
                return True
                
            with Image.open(image_path) as img:
                width, height = img.size
                return width >= self.min_image_size[0] or height >= self.min_image_size[1]
        except Exception as e:
            self.logger.error(f"Failed to read image {image_path}: {str(e)}")
            return False

    def process_image_list(self, images: List[str], json_path: Path) -> List[str]:
        """
        Process a list of images
        
        Args:
            images: List of image URLs
            json_path: Path to JSON file
            
        Returns:
            List of processed local image paths
        """
        if not images:
            return []
                
        processed_images = []
        
        for img_url in images:
            self.stats['total_images'] += 1
            
            try:
                img_filename = self.get_image_filename(img_url)
                img_path = self.find_image_in_folder(img_filename, json_path)

                if img_url.startswith('data:image'):
                    self.stats['base64_images'] = self.stats.get('base64_images', 0) + 1
                    result = self._process_base64_image(img_url, json_path)
                    if result:
                        img_path, local_url = result
                        processed_images.append(local_url)
                        self.stats['valid_images'] += 1
                    continue

                

                if img_filename == 'latex.php':
                    # Directly look for converted png file
                    img_folder = self.find_image_folder(json_path=json_path)
                    img_path = img_folder / 'latex.png'
                    if img_path.exists():
                        relative_path = img_path.relative_to(self.images_dir)
                        local_url = f"./downloaded_html/images/{relative_path}"
                        if not self.check_image_size(img_path):
                            self.stats['small_images'] += 1
                            self.logger.info(f"Image resolution too small: {img_filename}")
                            continue
                        self.stats['valid_images'] += 1
                        processed_images.append(local_url)
                        self.stats['latex_converted'] = self.stats.get('latex_converted', 0) + 1
                        continue
                    else:
                        # If png doesn't exist, try to find and convert php file
                        php_path = img_folder / 'latex.php'
                        if php_path.exists():
                            try:
                                # Rename to png
                                php_path.rename(img_path)
                                relative_path = img_path.relative_to(self.images_dir)
                                local_url = f"./downloaded_html/images/{relative_path}"
                                self.stats['latex_converted'] = self.stats.get('latex_converted', 0) + 1
                                self.logger.info(f"Renamed LaTeX PHP file to PNG: {img_path}")
                                if not self.check_image_size(img_path):
                                    self.stats['small_images'] += 1
                                    self.logger.info(f"Image resolution too small: {img_filename}")
                                    continue
                                self.stats['valid_images'] += 1
                                processed_images.append(local_url)
                                continue
                            except Exception as e:
                                self.logger.error(f"Failed to rename LaTeX PHP file: {str(e)}")
                                continue
                        else:
                            self.stats['missing_images'] += 1
                            self.logger.warning(f"LaTeX file not found: {img_filename} for {json_path}")
                            continue

                

                if img_path and ('svg+xml' in img_path.name):
                    try:
                        # Get base filename (without URL encoding)
                        base_name = img_path.name.split(',%')[0]  # Get part before first comma
                        if base_name.endswith('.svg+xml'):
                            base_name = base_name[:-8] + '.svg'  # Replace .svg+xml with .svg
                        
                        # Create new path
                        new_path = img_path.parent / base_name
                        img_path.rename(new_path)
                        img_path = new_path
                        self.logger.info(f"Fixed SVG filename: {new_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to rename SVG file: {str(e)}")
                        continue


                # Handle .image extension files
                if img_path and img_path.name.endswith('.image'):
                    try:
                        # Create new png path
                        new_path = img_path.with_suffix('.png')
                        # Check if png file already exists
                        if new_path.exists():
                            # If png already exists, use png path
                            img_path = new_path
                            self.logger.info(f"Using existing PNG file: {new_path}")
                        else:
                            # If png doesn't exist, rename file
                            img_path.rename(new_path)
                            img_path = new_path
                            self.logger.info(f"Renamed .image file to .png: {new_path}")
                        
                        # Add to processed results
                        relative_path = img_path.relative_to(self.images_dir)
                        local_url = f"./downloaded_html/images/{relative_path}"
                        if not self.check_image_size(img_path):
                            self.stats['small_images'] += 1
                            self.logger.info(f"Image resolution too small: {img_filename}")
                            continue
                        self.stats['valid_images'] += 1
                        processed_images.append(local_url)
                        continue
                    except Exception as e:
                        self.logger.error(f"Failed to rename .image file: {str(e)}")
                        continue

                
                
                if not img_path or len(str(img_path)) >= 255:
                    self.stats['missing_images'] += 1
                    self.logger.warning(f"Image not found or path too long: {img_filename} for {json_path}")
                    continue
                
                
                # Process SVG
                if img_path.suffix.lower() == '.svg':
                    self.stats['svg_images'] += 1
                    png_path = self.convert_svg_to_png(img_path)
                    if png_path:
                        relative_path = png_path.relative_to(self.images_dir)
                        local_url = f"./downloaded_html/images/{relative_path}"
                        self.stats['valid_images'] += 1
                        if not self.check_image_size(png_path):
                            self.stats['small_images'] += 1
                            self.logger.info(f"Image resolution too small: {img_filename}")
                            continue
                        processed_images.append(local_url)
                    continue
                
                # Process GIF
                if img_path.suffix.lower() == '.gif':
                    self.stats['gif_images'] += 1
                    if self.is_animated_gif(img_path):
                        self.stats['animated_gifs'] += 1
                        if self.remove_animated_gif:
                            try:
                                img_path.unlink()
                                self.stats['animated_gifs_removed'] += 1
                                self.logger.info(f"Removed animated GIF: {img_path}")
                            except Exception as e:
                                self.logger.error(f"Failed to remove animated GIF {img_path}: {str(e)}")
                        
                        continue
                    else:
                        self.stats['static_gifs'] += 1
                        png_path = self.convert_static_gif_to_png(img_path)
                        if png_path:
                            relative_path = png_path.relative_to(self.images_dir)
                            local_url = f"./downloaded_html/images/{relative_path}"
                            if not self.check_image_size(png_path):
                                self.stats['small_images'] += 1
                                self.logger.info(f"Image resolution too small: {img_filename}")
                                continue
                            self.stats['valid_images'] += 1
                            processed_images.append(local_url)
                        
                        continue
                
                # Process other image types
                if not self.check_image_size(img_path):
                    self.stats['small_images'] += 1
                    self.logger.info(f"Image resolution too small: {img_filename}")
                    continue
                
                relative_path = img_path.relative_to(self.images_dir)
                local_url = f"./downloaded_html/images/{relative_path}"
                
                self.stats['valid_images'] += 1
                processed_images.append(local_url)
                
            except Exception as e:
                self.stats['processing_errors'] = self.stats.get('processing_errors', 0) + 1
                self.logger.error(f"Error processing image: {str(e)}")
                
        return list(dict.fromkeys(processed_images))

    def process_qa_item(self, qa_item: Dict[str, Any], json_path: Path) -> Dict[str, Any]:
        """
        Process a single QA item's images
        
        Args:
            qa_item: QA item dictionary
            json_path: Path to JSON file
            
        Returns:
            Processed QA item dictionary
        """
        # First check if it contains images and update statistics
        has_question_images = bool(qa_item.get('question_images', []))
        
        # If there are any images in the question or answer, increment count
        if has_question_images:
            self.stats['qa_with_images'] = self.stats.get('qa_with_images', 0) + 1
        
        processed_item = qa_item.copy()
        
        # Process question images
        processed_item['question_images'] = self.process_image_list(
            qa_item.get('question_images', []), 
            json_path,
        )
        
        # Process solution images
        processed_item['solution_images'] = self.process_image_list(
            qa_item.get('solution_images', []), 
            json_path,
        )
        
        return processed_item
    

    def _process_base64_image(self, img_url: str, json_path: Path) -> Optional[Tuple[Path, str]]:
        """
        Process base64 image and return saved path and local URL
        
        Args:
            img_url: Base64 image URL
            json_path: Path to JSON file
            
        Returns:
            Tuple of (image_path, local_url) or None if processing fails
        """
        try:

            # First find matching complete base64 string
            complete_base64 = self.find_most_similar_base64(img_url, self.find_image_folder(json_path=json_path))
            if not complete_base64:
                self.logger.warning(f"No matching base64 string found")
                return None
                
            # Continue processing with complete base64 string
            img_url = complete_base64
            
            # Parse content type
            content_type = img_url.split(';')[0]
            if '/' in content_type:
                img_format = content_type.split('/')[1].lower()
            else:
                img_format = 'png'
                
            # Format mapping
            format_mapping = {
                'svg+xml': 'svg',
                'jpeg': 'jpg',
                'image': 'png',
                'svg': 'svg',
                'png': 'png',
                'gif': 'gif',
                'webp': 'webp',
                'bmp': 'png'
            }
            img_format = format_mapping.get(img_format, 'png')
            
            # Extract and process base64 data
            try:
                header, base64_data = img_url.split(',', 1)
            except ValueError:
                self.logger.error("Base64 data format error")
                return None
                
            # Clean data
            base64_data = base64_data.strip()
            
            # Add padding
            padding_length = len(base64_data) % 4
            if padding_length:
                base64_data += '=' * (4 - padding_length)
            
            try:
                # Decode data
                decoded_data = base64.b64decode(base64_data)
                
                # Validate data size
                if len(decoded_data) < 100:
                    self.logger.warning(f"Base64 decoded data too small: {len(decoded_data)} bytes")
                    return None
                    
                # Validate image data
                with Image.open(BytesIO(decoded_data)) as img:
                    img.verify()
                    
                    # Check image dimensions
                    img.seek(0)
                    width, height = img.size
                    if width < self.min_image_size[0] or height < self.min_image_size[1]:
                        self.logger.warning(f"Base64 image dimensions too small: {width}x{height}")
                        return None
                
                # Generate output path
                filename = f"base64_{uuid.uuid4().hex[:8]}.{img_format}"
                img_path = self.find_image_folder(json_path=json_path) / filename
                
                # Ensure directory exists
                img_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save image
                with open(img_path, 'wb') as f:
                    f.write(decoded_data)
                    
                # Verify saved file
                if not img_path.exists():
                    raise FileNotFoundError("File not successfully created")
                    
                if img_path.stat().st_size == 0:
                    raise ValueError("Saved file size is 0")
                    
                # Verify saved image
                with Image.open(img_path) as img:
                    img.verify()
                    
                # Generate local URL
                relative_path = img_path.relative_to(self.images_dir)
                local_url = f"./downloaded_html/images/{relative_path}"
                
                self.logger.info(f"Successfully saved base64 image: {img_path}")
                return img_path, local_url
                
            except base64.binascii.Error:
                self.logger.error("Base64 decoding failed")
                return None
            except Exception as e:
                self.logger.error(f"Failed to process base64 image: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Base64 image processing failed: {str(e)}")
            return None