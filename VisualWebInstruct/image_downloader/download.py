"""
Concurrent Image Processing Pipeline - A tool for extracting and downloading images from HTML JSON files

INPUT:
- source_path: Directory containing HTML files in JSON format
- target_path: Directory where image files will be saved
- max_workers: Number of concurrent threads for processing

OUTPUT:
- Downloaded images organized in a directory structure mirroring the source
- Base64-encoded images saved as JSON files
- Log file with processing details and statistics
- Console output showing progress and final statistics
"""

import os
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from build_folder import create_directory_structure
from acc_tree import process_json_file, extract_image_links, extract_base64
from downloader import ImageDownloader
import json
import logging
import concurrent.futures
from dataclasses import dataclass
from threading import Lock


@dataclass
class DownloadStats:
    """Data class for tracking download statistics"""
    images_found: int = 0
    images_downloaded: int = 0
    
    def __add__(self, other):
        return DownloadStats(
            self.images_found + other.images_found,
            self.images_downloaded + other.images_downloaded
        )

class ConcurrentImageProcessingPipeline:
    """
    Concurrent image processing pipeline supporting multi-threaded operations
    """
    def __init__(self, source_path: str, target_path: str, max_workers: int = 4):
        """
        Initialize the image processing pipeline.
        
        Args:
            source_path: Path to directory containing JSON files
            target_path: Path to directory for storing images
            max_workers: Maximum number of concurrent worker threads
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.max_workers = max_workers
        self.progress_lock = Lock()
        self.setup_logging()
        
    def setup_logging(self) -> None:
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('image_processing.log'),
                logging.StreamHandler()
            ]
        )
    
    def list_json_files(self) -> List[Path]:
        """List all JSON files in the source directory"""
        return list(self.source_path.rglob("*.json"))
    
    def convert_to_image_folder(self, json_path: Path) -> Path:
        """Convert JSON file path to corresponding image folder path"""
        relative_path = json_path.relative_to(self.source_path)
        image_folder = self.target_path / relative_path.parent / relative_path.stem
        return image_folder

    def download_image_batch(self, urls: List[str], folder: Path, pbar: tqdm) -> int:
        """
        Download a batch of images concurrently
        
        Args:
            urls: List of image URLs
            folder: Destination folder
            pbar: Progress bar object
            
        Returns:
            int: Number of successfully downloaded images
        """
        downloaded = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            downloader = ImageDownloader(str(folder))
            future_to_url = {
                executor.submit(downloader.download_single, url): url 
                for url in urls
            }
            
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    success = future.result()
                    if success:
                        downloaded += 1
                except Exception as e:
                    url = future_to_url[future]
                    logging.error(f"Error downloading {url}: {str(e)}")
                with self.progress_lock:
                    pbar.update(1)
                    
        return downloaded

    def download(self, json_path: Path, pbar: tqdm) -> DownloadStats:
        """
        Process a single JSON file and download its images
        
        Args:
            json_path: Path to JSON file
            pbar: Progress bar object
            
        Returns:
            DownloadStats: Download statistics
        """
        try:
            json_content = json_path.read_text(encoding='utf-8')
            acc_tree = process_json_file(json_content)
            
            if acc_tree is None:
                logging.error(f"Failed to parse: {json_path}")
                return DownloadStats()
                
            image_urls = extract_image_links(acc_tree)
            if not image_urls:
                logging.info(f"No images found: {json_path}")
                return DownloadStats()
                
            image_folder = self.convert_to_image_folder(json_path)
            downloaded = self.download_image_batch(image_urls, image_folder, pbar)
            
            return DownloadStats(len(image_urls), downloaded)
            
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON file: {json_path}")
            return DownloadStats()
        except Exception as e:
            logging.error(f"Error processing file {json_path}: {str(e)}")
            return DownloadStats()
    
    def download_base64(self, json_path: Path) -> None:
        """
        Extract and save Base64-encoded images from a JSON file
        
        Args:
            json_path: Path to JSON file
        """
        try:
            json_content = json_path.read_text(encoding='utf-8')
            acc_tree = process_json_file(json_content)
            
            if acc_tree is None:
                logging.error(f"Failed to parse: {json_path}")
                return DownloadStats()
            
            base64_strings = extract_base64(acc_tree)
            base64_strings = list(set(base64_strings))  # Remove duplicates
            if len(base64_strings) == 0:
                logging.info(f"No Base64 images found: {json_path}")
                return
            image_folder = self.convert_to_image_folder(json_path)
            base64_path = image_folder / "base64.json"
            with open(base64_path, 'w', encoding='utf-8') as f:
                json.dump(base64_strings, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved Base64 images: {json_path}")
            
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON file: {json_path}")
            return
        except Exception as e:
            logging.error(f"Error processing file {json_path}: {str(e)}")
            return

    
    def run_download(self) -> None:
        """Execute the complete image processing workflow"""
        create_directory_structure(str(self.source_path), str(self.target_path))
        
        files = self.list_json_files()
        total_files = len(files)
        
        logging.info(f"Starting to process {total_files} files")
        
        # Create overall progress bar
        total_stats = DownloadStats()
        processed_html = 0
        
        # Use ThreadPoolExecutor to process files
        with tqdm(total=total_files, desc="Overall Progress") as file_pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create image download progress bar
                with tqdm(desc="Download Progress", position=1) as download_pbar:
                    future_to_file = {
                        executor.submit(self.download, file, download_pbar): file 
                        for file in files
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_file):
                        file = future_to_file[future]
                        try:
                            stats = future.result()
                            total_stats += stats
                            if stats.images_found > 0:
                                processed_html += 1
                        except Exception as e:
                            logging.error(f"Failed to process file {file}: {str(e)}")
                        file_pbar.update(1)
        
        summary = f"""
        Processing completed:
        - Total files: {total_files}
        - HTML files containing images: {processed_html}
        - Total images found: {total_stats.images_found}
        - Successfully downloaded images: {total_stats.images_downloaded}
        """
        logging.info(summary)
        print(summary)
    
    def run_base64(self) -> None:
        """Execute the Base64 image processing workflow"""        
        files = self.list_json_files()
        total_files = len(files)
        
        logging.info(f"Starting to process {total_files} files")
        
        # Create overall progress bar
        total_stats = DownloadStats()
        
        # Use ThreadPoolExecutor to process files
        with tqdm(total=total_files, desc="Overall Progress") as file_pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.download_base64, file): file for file in files
                }
                
                # Wait for tasks to complete and update progress bar
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        future.result()  # Get task result
                    except Exception as e:
                        logging.error(f"Failed to process file {file}: {str(e)}")
                    file_pbar.update(1)  # Update progress bar after task completion
        

if __name__ == "__main__":
    SOURCE_PATH = "./downloaded_html/html"
    TARGET_PATH = "./downloaded_html/images"
    MAX_WORKERS = 4  # Number of concurrent threads, adjust as needed
    
    pipeline = ConcurrentImageProcessingPipeline(
        SOURCE_PATH, 
        TARGET_PATH,
        max_workers=MAX_WORKERS
    )
    pipeline.run_base64()