"""
Gemini Q&A Extraction Pipeline - A tool for extracting question-answer pairs from HTML content using Google's Gemini API

INPUT:
- source_path: Directory containing JSON files with HTML content
- target_path: Directory where extracted Q&A data will be saved
- api_key: Google Gemini API key
- max_workers: Maximum number of concurrent worker threads
- rate_limit_per_min: Maximum API calls per minute
- log_file: Path for log files

OUTPUT:
- JSON files containing structured Q&A data extracted from HTML content
- Log files tracking successful and failed operations
- Console output showing progress and summary statistics

The pipeline parses HTML content, extracts relevant information, uses Gemini AI to identify
question-answer pairs, and saves structured data with proper rate limiting and error handling.
"""

import os
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from threading import Lock, local, get_ident
import google.generativeai as genai
from qa import QAItem, extract_content_from_response, parse_markdown_qa, save_qa_data
import logging
from logging.handlers import RotatingFileHandler
from image_downloader.acc_tree import process_html_file, format_for_llm
from image_downloader.build_folder import create_directory_structure
import concurrent.futures
import json
import time
from collections import deque
from datetime import datetime

class GeminiQAPipeline:
    def __init__(self, 
                 source_path: str,
                 target_path: str,
                 api_key: str,
                 max_workers: int = 4,
                 rate_limit_per_min: int = 2000,
                 log_file: str = "pipeline.log"):
        """
        Initialize the Gemini Q&A extraction pipeline.
        
        Args:
            source_path: Path to directory containing JSON files with HTML content
            target_path: Path to directory where extracted Q&A data will be saved
            api_key: Google Gemini API key
            max_workers: Maximum number of concurrent worker threads
            rate_limit_per_min: Maximum API calls per minute
            log_file: Path for log files
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.api_key = api_key
        self.max_workers = max_workers
        self.rate_limit_per_min = rate_limit_per_min
        
        # Set up logging system
        self._setup_logging(log_file)
        
        # Initialize thread lock and API call tracking
        self.progress_lock = Lock()
        self.api_lock = Lock()
        self.request_times = deque(maxlen=rate_limit_per_min)
        
        # Use threading.local() to store model instances for each thread
        self.thread_local = local()
        
        # Initialize statistics counters
        self.valid_files = 0
        self.total_qa_items = 0
        self.api_calls = 0
        self.failed_calls = 0
        
        self.logger.info(f"Pipeline initialized with {max_workers} workers and {rate_limit_per_min} RPM limit")

    def _setup_logging(self, log_file: str):
        """
        Set up logging system, handling success and error logs separately.
        
        Args:
            log_file: Base log file path
        """
        self.logger = logging.getLogger('GeminiQAPipeline')
        self.logger.setLevel(logging.INFO)
        
        # Create base log format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create error log handler
        error_file = log_file.replace('.log', '_error.log')
        self.error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        self.error_handler.setFormatter(formatter)
        self.error_handler.setLevel(logging.WARNING)  # Only record WARNING and ERROR levels
        self.logger.addHandler(self.error_handler)
        
        # Create success log handler
        success_file = log_file.replace('.log', '_success.log')
        self.success_handler = RotatingFileHandler(
            success_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        self.success_handler.setFormatter(formatter)
        self.success_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.success_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        self.logger.info("Logging system initialized")
    
    def log_success(self, message: str):
        """
        Log success information to the success log only.
        
        Args:
            message: Success message to log
        """
        handlers = self.logger.handlers[:]
        # Temporarily remove other handlers
        for handler in handlers:
            if handler != self.success_handler:
                self.logger.removeHandler(handler)
        
        self.logger.info(message)
        
        # Restore other handlers
        for handler in handlers:
            if handler not in self.logger.handlers:
                self.logger.addHandler(handler)
    
    def get_model(self):
        """
        Get or create model instance for each thread.
        
        Returns:
            Gemini model instance
        """
        if not hasattr(self.thread_local, "model"):
            genai.configure(api_key=self.api_key)
            self.thread_local.model = genai.GenerativeModel("gemini-1.5-flash")
            thread_id = f"Thread-{get_ident()}"
            self.logger.info(f"Created new model instance for {thread_id}")
        return self.thread_local.model
    
    def _check_rate_limit(self) -> float:
        """
        Check and wait for API rate limit.
        
        Returns:
            Wait time in seconds
        """
        with self.api_lock:
            now = datetime.now()
            while self.request_times and (now - self.request_times[0]).total_seconds() > 60:
                self.request_times.popleft()
            
            if len(self.request_times) >= self.rate_limit_per_min:
                wait_time = 60 - (now - self.request_times[0]).total_seconds()
                if wait_time > 0:
                    self.logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                return max(0, wait_time)
            
            self.request_times.append(now)
            return 0
    
    def _call_gemini_api(self, prompt: str, file_path: str = "unknown") -> Optional[str]:
        """
        Call Gemini API using thread-specific model instance with retries.
        
        Args:
            prompt: Text prompt for the Gemini API
            file_path: Path to the file being processed (for logging)
            
        Returns:
            Response content or None if all retries fail
        """
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                wait_time = self._check_rate_limit()
                if wait_time > 0:
                    time.sleep(wait_time)
                
                model = self.get_model()
                response = model.generate_content(prompt)
                
                with self.progress_lock:
                    self.api_calls += 1
                    
                return extract_content_from_response(response)
                
            except Exception as e:
                self.logger.error(f"API call failed (file: {file_path}, attempt: {attempt + 1}/{max_retries}): {str(e)}")
                with self.progress_lock:
                    self.failed_calls += 1
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        self.logger.error(f"All retries failed, abandoning file: {file_path}")
        return None
    
    def process_single_file(self, json_path: Path, pbar: tqdm) -> Optional[List[QAItem]]:
        """
        Process a single JSON file to extract Q&A content.
        
        Args:
            json_path: Path to JSON file
            pbar: Progress bar object
            
        Returns:
            List of QAItem objects or None if processing fails
        """
        try:
            self.logger.info(f"Starting to process file: {json_path}")
            
            json_content = json_path.read_text(encoding='utf-8')
            acc_tree = process_html_file(json_content)
            if acc_tree is None:
                self.logger.error(f"HTML parsing failed: {json_path}")
                with self.progress_lock:
                    pbar.update(1)
                return None

            llm_format = format_for_llm(acc_tree)
            prompt = f"""Analyze this webpage content and extract questions, images, and complete solution details in Markdown format.

            Please format your response as follows:

            **Question 1:** 
            [complete question text]

            **Images:**
            * [First image URL if available]
            * [Second image URL if available]
            [continue for each additional image...]

            **Solution:**
            [Copy the complete solution text from the webpage, including all steps, explanations, and calculations]

            **Images in Solution:**
            * [First image URL if available]
            * [Second image URL if available]
            [continue for each additional image...]

            [repeat for each additional question...]

            Requirements:
            - Keep the complete solution text exactly as shown in the webpage
            - Use Markdown formatting throughout the response
            - Mark missing content as "Not found"
            - For images, include URL only
            - For multiple questions, number them sequentially
            - Do not summarize or modify the solution text
            - Preserve all mathematical notations and formulas
            - Keep all step-by-step explanations intact
            - Preserve all line breaks and indentation in solution text
            - If there is no question in the content, mark it as "Not found"
            - If the webpage is empty or missing, return nothing

            Webpage content:
            {llm_format}
            """
            
            content = self._call_gemini_api(prompt, str(json_path))
            if content is None:
                with self.progress_lock:
                    pbar.update(1)
                return None
                
            qa_items = parse_markdown_qa(content)
            if qa_items:
                qna_folder = self.convert_to_qna_folder(json_path)
                output_path = qna_folder / "qa_data.json"
                save_qa_data(qa_items, output_path)
                
                with self.progress_lock:
                    self.valid_files += 1
                    self.total_qa_items += len(qa_items)
                    pbar.update(1)
                    
                # Use dedicated success logging method
                self.log_success(f"Successfully saved file {json_path} -> {output_path}, extracted {len(qa_items)} QA items")
                
            else:
                with self.progress_lock:
                    pbar.update(1)
                self.logger.warning(f"No QA items extracted from file: {json_path}")
                
            return qa_items
            
        except Exception as e:
            self.logger.error(f"Error processing file {json_path}: {str(e)}", exc_info=True)
            with self.progress_lock:
                pbar.update(1)
            return None
    
    def list_json_files(self) -> List[Path]:
        """
        List all JSON files in the source directory.
        
        Returns:
            List of Path objects to JSON files
        """
        files = list(self.source_path.rglob("*.json"))
        self.logger.info(f"Found {len(files)} JSON files to process")
        return files
    
    def convert_to_qna_folder(self, json_path: Path) -> Path:
        """
        Convert JSON file path to corresponding Q&A folder path.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Path to corresponding Q&A folder
        """
        relative_path = json_path.relative_to(self.source_path)
        return self.target_path / relative_path.parent / relative_path.stem
        
    def run(self):
        """Execute the complete Q&A extraction pipeline"""
        self.logger.info("Starting pipeline execution...")
        create_directory_structure(str(self.source_path), str(self.target_path))
        files = self.list_json_files()
        total_files = len(files)
        
        start_time = time.time()
                
        with tqdm(total=total_files, desc="Processing files") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_file, file, pbar): file 
                    for file in files
                }
                
                for future in concurrent.futures.as_completed(futures):
                    file = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error executing task {file}: {str(e)}", exc_info=True)

        duration = time.time() - start_time
        avg_time_per_file = duration / total_files if total_files > 0 else 0
        avg_qa_per_file = self.total_qa_items / self.valid_files if self.valid_files > 0 else 0
        
        summary = f"""
        Processing completed:
        - Total files: {total_files}
        - Successfully processed files: {self.valid_files}
        - Total QA items extracted: {self.total_qa_items}
        - Average QA items per file: {avg_qa_per_file:.2f}
        - Total API calls: {self.api_calls}
        - Failed API calls: {self.failed_calls}
        - Total processing time: {duration:.2f} seconds
        - Average processing time per file: {avg_time_per_file:.2f} seconds
        - Concurrent threads: {self.max_workers}
        """
        self.logger.info(summary)
        print(summary)

if __name__ == "__main__":
    source_path = "./downloaded_html"
    target_path = "./downloaded_qa"
    API_KEY = r''
    
    pipeline = GeminiQAPipeline(
        source_path=source_path,
        target_path=target_path,
        api_key=API_KEY,
        max_workers=16,
        rate_limit_per_min=2000,
        log_file="pipeline.log"  # Specify log file path
    )
    
    pipeline.run()