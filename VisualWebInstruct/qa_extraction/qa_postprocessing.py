"""
QA Filter Pipeline - A tool for validating and filtering question-answer pairs using Google's Gemini API

INPUT:
- Input JSON file containing QA pairs with images
- API keys for Google Gemini
- Base path for resolving image paths
- Processing parameters (batch size, workers)

OUTPUT:
- Valid QA pairs JSON file with filtered images
- Invalid QA pairs JSON file
- Processing statistics and logs

The pipeline analyzes each QA pair to determine if the question is meaningful and valid,
and identifies which images are relevant and helpful. It handles image processing to
improve visibility, manages multiple API keys, and provides detailed statistics about
the filtering process.
"""

import json
import os
import threading
import time
import base64
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import logging
from itertools import cycle
from typing import Any, List
import google.generativeai as genai
from tqdm import tqdm
from PIL import Image
import io

class APIKeyManager:
    def __init__(self, api_keys: List[str], workers: int = 5):
        """
        Initialize with API keys and number of workers
        
        Args:
            api_keys: List of API keys
            workers: Number of worker threads
        """
        self.api_keys = api_keys
        self.num_workers = workers
        self.worker_models = {}  # Store models for each worker
        self.lock = Lock()
        self.next_key_index = 0
        
    def get_model_for_worker(self) -> Any:
        """
        Get or create model for current worker thread
        
        Returns:
            Gemini model instance for current thread
        """
        worker_id = threading.get_ident()
        
        # If this worker already has a model, return it
        if worker_id in self.worker_models:
            return self.worker_models[worker_id]
            
        # Otherwise create a new model for this worker
        with self.lock:
            # Select API key in round-robin fashion
            api_key = self.api_keys[self.next_key_index]
            self.next_key_index = (self.next_key_index + 1) % len(self.api_keys)
            
            # Configure model
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Save model
            self.worker_models[worker_id] = model
            return model

class Statistics:
    def __init__(self):
        """Initialize statistics tracking object"""
        self.total_pairs = 0
        self.valid_pairs = 0
        self.invalid_pairs = 0
        self.processed_images = 0
        self.valid_images = 0
        self.lock = Lock()
    
    def update(self, is_valid, num_original_images, num_valid_images):
        """
        Update statistics with processing results
        
        Args:
            is_valid: Whether the QA pair is valid
            num_original_images: Number of original images in the pair
            num_valid_images: Number of valid images after filtering
        """
        with self.lock:
            self.total_pairs += 1
            if is_valid:
                self.valid_pairs += 1
            else:
                self.invalid_pairs += 1
            self.processed_images += num_original_images
            self.valid_images += num_valid_images
    
    def to_dict(self):
        """
        Convert statistics to dictionary format
        
        Returns:
            Dictionary containing statistics
        """
        return {
            "total_pairs_processed": self.total_pairs,
            "valid_pairs": self.valid_pairs,
            "invalid_pairs": self.invalid_pairs,
            "valid_ratio": f"{(self.valid_pairs/self.total_pairs*100):.2f}%" if self.total_pairs > 0 else "0%",
            "total_images_processed": self.processed_images,
            "valid_images": self.valid_images,
            "image_valid_ratio": f"{(self.valid_images/self.processed_images*100):.2f}%" if self.processed_images > 0 else "0%"
        }

class QAFilterPipeline:
    def __init__(self, api_key_manager, debug=False):
        """
        Initialize the QA filter pipeline
        
        Args:
            api_key_manager: Manager for API keys
            debug: Whether to enable debug logging
        """
        self.api_key_manager = api_key_manager
        self.debug = debug
        self.max_retries = 3
        self.retry_delay = 5
        self.stats = Statistics()
        
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/qa_filter_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _enhance_image(self, image):
        """
        Enhance image clarity
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced image or original if enhancement fails
        """
        try:
            from PIL import ImageEnhance
            
            # Convert to grayscale
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
                
            # Check if image is all black or nearly all black
            pixels = list(gray.getdata())
            dark_pixels = sum(1 for p in pixels if p < 50)  # Count dark pixels
            if dark_pixels / len(pixels) > 0.9:  # If more than 90% are dark pixels
                # Adaptively adjust brightness and contrast
                enhancer = ImageEnhance.Brightness(gray)
                enhanced = enhancer.enhance(2.0)  # Increase brightness
                
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(2.0)  # Increase contrast
                
                # Convert back to RGB mode
                if image.mode == 'RGB':
                    enhanced = enhanced.convert('RGB')
                return enhanced
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {str(e)}")
            return image
    
    def _process_image(self, image_path):
        """
        Process image: add white background if transparent and enhance visibility
        
        Args:
            image_path: Path to image file
            
        Returns:
            Binary image data or None if processing fails
        """
        try:
            # Disable PIL debug output
            Image.logger.setLevel(logging.WARNING)
            
            # Open image
            with Image.open(image_path) as image:
                # If image has alpha channel (transparency)
                if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                    # Create white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    
                    # If P mode, convert to RGBA first
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                        
                    # Paste image onto white background
                    if image.mode in ('RGBA', 'LA'):
                        background.paste(image, mask=image.split()[-1])
                    else:
                        background.paste(image)
                        
                    image = background
                
                # If not RGB mode, convert to RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Enhance image clarity
                enhanced_image = self._enhance_image(image)
                
                # Save to byte stream
                img_byte_arr = io.BytesIO()
                enhanced_image.save(img_byte_arr, format='JPEG', quality=95)
                img_byte_arr = img_byte_arr.getvalue()
                
                return img_byte_arr
                
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def check_qa_pair(self, qa_pair, base_path=""):
        """
        Check QA pair validity and identify meaningful images
        
        Args:
            qa_pair: Dictionary containing QA pair data
            base_path: Base path for resolving image paths
            
        Returns:
            Tuple of (is_valid, valid_question_images, valid_solution_images)
        """
        # Get model for current worker
        model = self.api_key_manager.get_model_for_worker()
        parts = []
        
        try:
            # Add initial prompt
            parts.append({
                "text": f"""Please analyze this question-answer pair and its images:

Question: {qa_pair.get('question', '')}
Solution: {qa_pair.get('solution', '')}

Your tasks:
1. Determine if the question is meaningful and valid.
2. For the question images (if any), determine if each is:
   - Properly referenced in the question
   - Clear and visible
   - Actually helps understand the question

3. For the solution images (if any), determine if each is:
   - Helps explain the solution

Notes:
- Image indices start from 0 (e.g., first image is index 0, second is index 1, etc.)
- Images should be marked as valid if they show the actual content being discussed
- Images should be marked as invalid only if they are:
  * Completely irrelevant to the question/solution
  * Corrupted or unreadable
  * Duplicate or redundant
"""
            })

            # Process question images
            question_images = qa_pair.get('question_images', [])
            # Process solution images
            solution_images = qa_pair.get('solution_images', [])
            # Process question images
            parts.append({"text": "\nQuestion Images:"})
            for img_path in question_images:
                try:
                    full_path = os.path.join(base_path, img_path)
                    image_data = self._process_image(full_path)
                    if image_data:
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        })
                        if self.debug:
                            self.logger.debug(f"Added processed question image: {img_path}")
                except Exception as e:
                    self.logger.error(f"Could not load question image {img_path}: {str(e)}")
                    continue

            # Process solution images
            parts.append({"text": "\nSolution Images (starting a new section, indexes reset to 0):"})
            for img_path in solution_images:
                try:
                    full_path = os.path.join(base_path, img_path)
                    image_data = self._process_image(full_path)
                    if image_data:
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        })
                        if self.debug:
                            self.logger.debug(f"Added processed solution image: {img_path}")
                except Exception as e:
                    self.logger.error(f"Could not load solution image {img_path}: {str(e)}")
                    continue

            parts.append({
                "text": """
Please respond in this exact format:
QUESTION_VALID: [yes/no]
ANALYSIS: [Brief explanation of why the question is valid/invalid]
QUESTION_IMAGES: [comma-separated list of valid image indices starting from 0]
QUESTION_IMAGES_REASON: [Brief explanation for each image decision]
SOLUTION_IMAGES: [comma-separated list of valid image indices starting from 0]
SOLUTION_IMAGES_REASON: [Brief explanation for each image decision]

CRITICAL RESPONSE FORMAT INSTRUCTIONS:
- You MUST respond using EXACTLY this format with no additional text
- Use ONLY numeric indices for images, starting from 0
- If no images are valid, use an empty string
- Be precise and use actual numbers
- Always use numeric indices (0,1,2...)
- Use empty string for no images (e.g., "SOLUTION_IMAGES: ")
- Do not add explanatory text in the indices field

EXAMPLES OF CORRECT RESPONSES:
QUESTION_VALID: yes
ANALYSIS: The question is clear and requires technical explanation
QUESTION_IMAGES: 0,1
QUESTION_IMAGES_REASON: Image 0 shows the problem setup, Image 1 provides additional context
SOLUTION_IMAGES: 0
SOLUTION_IMAGES_REASON: Image 0 clearly illustrates the solution steps

OR

QUESTION_VALID: no
ANALYSIS: The question is too vague and lacks necessary context
QUESTION_IMAGES: 
QUESTION_IMAGES_REASON: No images provide meaningful information
SOLUTION_IMAGES: 
SOLUTION_IMAGES_REASON: No solution images are relevant

Note: By default, consider all images as valid unless there's a clear reason to exclude them."""
            })

            # Send request with retries
            for attempt in range(self.max_retries):
                try:
                    response = model.generate_content({"parts": parts})
                    response_text = response.text.strip()
                    
                    if self.debug:
                        self.logger.debug(f"Gemini response:\n{response_text}")
                    
                    # Parse response
                    response_data = {
                        'question_valid': False,
                        'analysis': '',
                        'valid_q_images': [],
                        'q_images_reason': '',
                        'valid_s_images': [],
                        's_images_reason': ''
                    }

                    for line in response_text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line.startswith('QUESTION_VALID:'):
                            response_data['question_valid'] = line.split(':')[1].strip().lower() == 'yes'
                        elif line.startswith('ANALYSIS:'):
                            response_data['analysis'] = line.split(':')[1].strip()
                        elif line.startswith('QUESTION_IMAGES:'):
                            indices = line.split(':')[1].strip()
                            if indices:
                                try:
                                    valid_indices = [int(i.strip()) for i in indices.split(',')]
                                    response_data['valid_q_images'] = [question_images[i] for i in valid_indices if i < len(question_images)]
                                except (ValueError, IndexError) as e:
                                    self.logger.error(f"Error parsing question image indices: {str(e)}")
                        elif line.startswith('QUESTION_IMAGES_REASON:'):
                            response_data['q_images_reason'] = line.split(':')[1].strip()
                        elif line.startswith('SOLUTION_IMAGES:'):
                            indices = line.split(':')[1].strip()
                            if indices:
                                try:
                                    valid_indices = [int(i.strip()) for i in indices.split(',')]
                                    response_data['valid_s_images'] = [solution_images[i] for i in valid_indices if i < len(solution_images)]
                                except (ValueError, IndexError) as e:
                                    self.logger.error(f"Error parsing solution image indices: {str(e)}")
                        elif line.startswith('SOLUTION_IMAGES_REASON:'):
                            response_data['s_images_reason'] = line.split(':')[1].strip()

                    if self.debug:
                        self.logger.debug(f"Question valid: {response_data['question_valid']}")
                        self.logger.debug(f"Analysis: {response_data['analysis']}")
                        self.logger.debug(f"Question images reason: {response_data['q_images_reason']}")
                        self.logger.debug(f"Solution images reason: {response_data['s_images_reason']}")
                    

                    return (
                        response_data['question_valid'],
                        response_data['valid_q_images'],
                        response_data['valid_s_images']
                    )
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                        return False, [], []  # Return default values instead of None
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    
        except Exception as e:
            self.logger.error(f"Error in check_qa_pair: {str(e)}")
            return False, [], []  # Return default values instead of None

    def process_qa_pair(self, qa_pair, base_path=""):
        """
        Process a single QA pair
        
        Args:
            qa_pair: Dictionary containing QA pair data
            base_path: Base path for resolving image paths
            
        Returns:
            Tuple of (is_valid, processed_qa_pair)
        """
        try:
            if self.debug:
                self.logger.debug(f"Processing QA pair: {qa_pair.get('question', '')[:100]}...")
                self.logger.debug(f"Original question images: {qa_pair.get('question_images', [])}")
                self.logger.debug(f"Original solution images: {qa_pair.get('solution_images', [])}")
            
            # Count original images
            original_images = len(qa_pair.get('question_images', [])) + len(qa_pair.get('solution_images', []))
            
            is_valid, valid_q_images, valid_s_images = self.check_qa_pair(qa_pair, base_path)
            
            if self.debug:
                self.logger.debug(f"Check result - is_valid: {is_valid}")
                self.logger.debug(f"Valid question images after check: {valid_q_images}")
                self.logger.debug(f"Valid solution images after check: {valid_s_images}")
            
            # Count valid images
            valid_images = len(valid_q_images) + len(valid_s_images)
            
            # Update statistics
            self.stats.update(is_valid, original_images, valid_images)
            
            if not is_valid:
                if self.debug:
                    self.logger.debug("Question marked as invalid, returning original pair")
                return False, qa_pair
            
            processed_pair = qa_pair.copy()
            processed_pair['question_images'] = valid_q_images
            processed_pair['solution_images'] = valid_s_images
            
            if self.debug:
                self.logger.debug(f"Final processed pair question images: {processed_pair['question_images']}")
                self.logger.debug(f"Final processed pair solution images: {processed_pair['solution_images']}")
            
            return True, processed_pair
            
        except Exception as e:
            self.logger.error(f"Error processing QA pair: {str(e)}")
            if self.debug:
                self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            return False, qa_pair  # Return original pair on error

    def filter_qa_pairs(self, input_file, output_valid_file, output_invalid_file, base_path="", batch_size=10, max_workers=5):
        """
        Filter QA pairs from input file into valid and invalid sets
        
        Args:
            input_file: Path to input JSON file containing QA pairs
            output_valid_file: Path to save valid QA pairs
            output_invalid_file: Path to save invalid QA pairs
            base_path: Base path for resolving image paths
            batch_size: Number of QA pairs to process in each batch
            max_workers: Maximum number of concurrent worker threads
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                qa_data = data.get('qa_items', [])
                metadata = data.get('metadata', {})
            else:
                qa_data = data
                metadata = {}
            if self.debug:
                qa_data = qa_data[:100]

            self.logger.info(f"Processing {len(qa_data)} QA pairs from {input_file}")
            
            valid_pairs = []
            invalid_pairs = []
            processed_count = 0
            
            with tqdm(total=len(qa_data)) as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for i in range(0, len(qa_data), batch_size):
                        batch = qa_data[i:i + batch_size]
                        for qa_pair in batch:
                            futures.append(
                                executor.submit(self.process_qa_pair, qa_pair, base_path)
                            )
                    
                    for future in futures:
                        try:
                            is_valid, processed_pair = future.result()
                            if is_valid:
                                valid_pairs.append(processed_pair)
                            else:
                                invalid_pairs.append(processed_pair)
                            
                            processed_count += 1
                            pbar.update(1)
                            
                            if processed_count % 100 == 0:
                                self._save_results(valid_pairs, invalid_pairs, output_valid_file, output_invalid_file)
                                
                        except Exception as e:
                            self.logger.error(f"Error processing batch: {str(e)}")
                    
            
            # Final save with statistics
            self._save_final_results(valid_pairs, invalid_pairs, output_valid_file, output_invalid_file, metadata)
            
        except Exception as e:
            self.logger.error(f"Error in filter_qa_pairs: {str(e)}")
            raise

    def _save_results(self, valid_pairs, invalid_pairs, output_valid_file, output_invalid_file):
        """
        Save intermediate results
        
        Args:
            valid_pairs: List of valid QA pairs
            invalid_pairs: List of invalid QA pairs
            output_valid_file: Path to save valid QA pairs
            output_invalid_file: Path to save invalid QA pairs
        """
        try:
            with open(output_valid_file, 'w', encoding='utf-8') as f:
                json.dump(valid_pairs, f, ensure_ascii=False, indent=2)
            with open(output_invalid_file, 'w', encoding='utf-8') as f:
                json.dump(invalid_pairs, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved checkpoint - Valid: {len(valid_pairs)}, Invalid: {len(invalid_pairs)}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

    def _save_final_results(self, valid_pairs, invalid_pairs, output_valid_file, output_invalid_file, original_metadata):
        """
        Save final results with statistics
        
        Args:
            valid_pairs: List of valid QA pairs
            invalid_pairs: List of invalid QA pairs
            output_valid_file: Path to save valid QA pairs
            output_invalid_file: Path to save invalid QA pairs
            original_metadata: Original metadata from input file
        """
        try:
            # Prepare metadata with statistics
            stats_dict = self.stats.to_dict()
            metadata = {
                "original_metadata": original_metadata,
                "processing_metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "statistics": stats_dict
                }
            }
            
            # Save valid pairs
            valid_data = {
                "metadata": metadata,
                "qa_items": valid_pairs
            }
            with open(output_valid_file, 'w', encoding='utf-8') as f:
                json.dump(valid_data, f, ensure_ascii=False, indent=2)
            
            # Save invalid pairs
            invalid_data = {
                "metadata": metadata,
                "qa_items": invalid_pairs
            }
            with open(output_invalid_file, 'w', encoding='utf-8') as f:
                json.dump(invalid_data, f, ensure_ascii=False, indent=2)
            
            # Log final statistics
            self.logger.info("Final Statistics:")
            for key, value in stats_dict.items():
                self.logger.info(f"{key}: {value}")
                
        except Exception as e:
            self.logger.error(f"Error saving final results: {str(e)}")

def main():
    api_keys = [

    ]
    
    api_key_manager = APIKeyManager(api_keys, workers=50)
    pipeline = QAFilterPipeline(api_key_manager, debug=False)
    
    pipeline.filter_qa_pairs(
        input_file="aggregated_qa_data_filtered.json",
        output_valid_file="qa_filtering/valid_pairs.json",
        output_invalid_file="qa_filtering/invalid_pairs.json",
        base_path="",
        batch_size=1000,
        max_workers=50
    )

if __name__ == "__main__":
    main()