"""
Image Downloader - A robust utility for downloading images from URLs with retry capabilities

INPUT:
- URLs: Single or batch image URLs for download
- save_dir: Directory where downloaded images will be stored
- max_retries: Maximum number of retry attempts for failed downloads
- max_workers: Number of concurrent download threads for batch operations

OUTPUT:
- Downloaded image files saved to the specified directory
- Boolean status for single downloads (True for success, False for failure)
- Count of successfully downloaded images for batch operations
- Progress bar visualization during batch downloads
"""

import requests
import os
from urllib.parse import urlparse, unquote
import logging
from time import sleep
from typing import Optional, List
import concurrent.futures
from tqdm import tqdm
from threading import Lock
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import hashlib

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ImageDownloader:
    def __init__(self, save_dir: str = "downloaded_images", max_retries: int = 3):
        """
        Initialize the image downloader.
        
        Args:
            save_dir: Directory to save downloaded images
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.save_dir = save_dir
        # Disable log output
        logging.getLogger().setLevel(logging.WARNING)
        self._setup_save_directory()
        
        # Create session and retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure adapter
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Set common request headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
        })
        self.progress_lock = Lock()
        self.downloaded_count = 0  # Add counter

    def _setup_save_directory(self):
        """Create save directory if it doesn't exist"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def _clean_filename(self, filename: str) -> str:
        """
        Remove invalid characters from filename.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename with invalid characters replaced
        """
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return cleaned or "image.jpg"
    
    def _get_file_hash(self, file_path=None, content=None, chunk_size=8192):
        """
        Calculate MD5 hash of a file or content.
        
        Args:
            file_path: Path to file for hashing
            content: Binary content for hashing
            chunk_size: Size of chunks to read when hashing files
            
        Returns:
            MD5 hash as hexadecimal string
        """
        hasher = hashlib.md5()
        if file_path:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b''):
                    hasher.update(chunk)
        elif content:
            hasher.update(content)
        return hasher.hexdigest()

    def _get_filename_from_url(self, url: str) -> str:
        """
        Extract filename from URL.
        
        Args:
            url: Image URL
            
        Returns:
            Filename extracted from URL or default name if extraction fails
        """
        try:
            parsed = urlparse(url)
            filename = unquote(os.path.basename(parsed.path))
            filename = self._clean_filename(filename)
            
            # If filename has no extension, add .jpg
            if not filename or '.' not in filename:
                path_parts = [p for p in parsed.path.split('/') if p]
                for part in reversed(path_parts):
                    if '.' in part:
                        cleaned = self._clean_filename(unquote(part))
                        if cleaned:
                            return cleaned
                # If filename is obtained but has no extension, add .jpg
                if filename:
                    return f"{filename}.jpg"
                return "image.jpg"
                
            return filename
        except:
            return "image.jpg"

    def _get_unique_filename(self, filename: str) -> str:
        """
        Generate a unique filename if the original already exists.
        
        Args:
            filename: Original filename
            
        Returns:
            Unique filename that doesn't conflict with existing files
        """
        base_name, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        
        while os.path.exists(os.path.join(self.save_dir, new_filename)):
            new_filename = f"{base_name}_{counter}{ext}"
            counter += 1
            
        return new_filename

    def download_single(self, url: str) -> bool:
        """
        Download a single image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            base_filename = self._get_filename_from_url(url)
            original_path = os.path.join(self.save_dir, base_filename)
            
            # If file already exists, get its hash
            if os.path.exists(original_path):
                existing_hash = self._get_file_hash(file_path=original_path)
                
                # Get hash of new file content
                response = self.session.get(url, timeout=30, stream=True, verify=False)
                if response.status_code != 200:
                    return False
                    
                content = response.content
                new_hash = self._get_file_hash(content=content)
                
                # If hashes match, skip download
                if existing_hash == new_hash:
                    return True
                    
                # Hashes differ, generate new filename
                filename = self._get_unique_filename(base_filename)
                file_path = os.path.join(self.save_dir, filename)
                
                # Write content
                with open(file_path, 'wb') as f:
                    f.write(content)
                return True
                
            # File doesn't exist, download directly
            return self._download_new_file(url, original_path)
            
        except Exception as e:
            return False
    
    def _download_new_file(self, url: str, file_path: str) -> bool:
        """
        Download a new file from URL.
        
        Args:
            url: Image URL
            file_path: Path to save the file
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            response = self.session.get(url, timeout=30, stream=True, verify=False)
            if response.status_code != 200:
                return False
                
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except:
            return False

    def process_download_result(self, future, pbar):
        """
        Process download result and update progress bar.
        
        Args:
            future: Future object from concurrent download
            pbar: Progress bar object
        """
        try:
            result = future.result()
            with self.progress_lock:
                if result:
                    self.downloaded_count += 1
                pbar.set_postfix({
                    "Success": f"{self.downloaded_count}",
                }, refresh=True)  # Force refresh display
                pbar.update(1)
        except Exception as e:
            with self.progress_lock:
                pbar.update(1)

    def batch_download(self, urls: List[str], max_workers: int = 4) -> int:
        """
        Download multiple images in parallel.
        
        Args:
            urls: List of image URLs
            max_workers: Maximum number of concurrent download threads
            
        Returns:
            int: Number of successfully downloaded images
        """
        if not urls:
            return 0
            
        self.downloaded_count = 0
        total_urls = len(urls)
        
        # Use tqdm.auto to avoid conflicts with the main progress bar
        from tqdm.auto import tqdm
        
        with tqdm(
            total=total_urls,
            desc="Download Progress",
            unit="img",
            unit_scale=False,
            ncols=100,
            position=1,  # Ensure download progress bar appears below main progress bar
            leave=False,  # Clear this progress bar upon completion
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for url in urls:
                    future = executor.submit(self.download_single, url)
                    futures.append(future)
                    future.add_done_callback(
                        lambda f: self.process_download_result(f, pbar)
                    )

                concurrent.futures.wait(futures)

        return self.downloaded_count

    def __del__(self):
        """Clean up resources when the object is deleted"""
        try:
            self.session.close()
        except:
            pass