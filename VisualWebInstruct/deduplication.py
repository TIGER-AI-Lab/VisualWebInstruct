"""
Multi-Directory URL Processor - A tool for extracting, normalizing, and analyzing URLs from JSON files

INPUT:
- input_dirs: List of directories containing JSON files with URLs
- output_file: Path to save the unique normalized URLs
- useless_domains_file: File containing domains to filter out
- max_workers: Number of concurrent processing workers

OUTPUT:
- Text file containing unique normalized URLs
- Domain statistics text file with frequency and percentage
- Filtered domains statistics file
- Histogram visualization of top domains

The processor extracts URLs from JSON files, normalizes them for consistency,
filters out unwanted domains, and analyzes domain frequency. It uses multiprocessing
for efficient processing of large datasets.
"""

import json
from urllib.parse import urlparse, urlunparse
from typing import Set, List, Dict, Generator, Iterable, Tuple
from pathlib import Path
import glob
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import os
from itertools import chain
import psutil
from dataclasses import dataclass
import signal
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@dataclass
class ProcessingStats:
    """Processing statistics"""
    total_dirs: int = 0
    total_files: int = 0
    processed_files: int = 0
    total_urls: int = 0
    unique_urls: int = 0
    filtered_urls: int = 0  # New: count of filtered URLs

def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent comparison
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL
    """
    try:
        parsed = urlparse(url.strip().lower())
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            ''
        ))
    except Exception:
        return url

def process_single_file(file_path: str) -> Tuple[int, List[str]]:
    """
    Process a single JSON file to extract URLs
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Tuple of (url_count, url_list)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        urls = []
        if 'visual_matches' in data:
            for match in data['visual_matches']:
                if 'link' in match:
                    urls.append(match['link'])
        
        return len(urls), urls
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0, []

def process_batch(file_paths: List[str]) -> Tuple[int, List[str]]:
    """
    Process a batch of files and return URLs
    
    Args:
        file_paths: List of paths to JSON files
        
    Returns:
        Tuple of (processed_files_count, url_list)
    """
    os.nice(10)  # Lower process priority
    
    total_urls = 0
    all_urls = []
    
    for file_path in file_paths:
        url_count, urls = process_single_file(file_path)
        total_urls += url_count
        all_urls.extend(urls)
    
    return len(file_paths), all_urls

def init_worker():
    """Initialize worker process to handle signals properly"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class MultiDirURLProcessor:
    def __init__(self, input_dirs: List[str], output_file: str, useless_domains_file: str = None, max_workers: int = None):
        """
        Initialize URL processor
        
        Args:
            input_dirs: List of directories containing JSON files
            output_file: Path to save output file
            useless_domains_file: Path to file containing domains to filter
            max_workers: Maximum number of worker processes
        """
        self.memory = psutil.virtual_memory()
        self.cpu_count = os.cpu_count()
        
        self.input_dirs = input_dirs
        self.output_file = output_file
        self.max_workers = max_workers or max(1, min(self.cpu_count * 2, 32))
        self.stats = ProcessingStats()
        self.stats.total_dirs = len(input_dirs)
        self.url_set: Set[str] = set()
        self.domain_counter = Counter()
        
        # Load list of domains to filter
        self.useless_domains: Set[str] = set()
        if useless_domains_file:
            self._load_useless_domains(useless_domains_file)
        
        self.batch_size = self._calculate_optimal_batch_size()

    def _load_useless_domains(self, file_path: str):
        """
        Load list of domains to filter
        
        Args:
            file_path: Path to file containing domains to filter
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.useless_domains = {line.strip().lower() for line in f if line.strip()}
            print(f"Loaded {len(self.useless_domains):,} domains to filter")
        except Exception as e:
            print(f"Error loading useless domains file: {str(e)}")
            self.useless_domains = set()

    def _get_domain(self, url: str) -> str:
        """
        Extract domain from URL
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url

    def _should_keep_url(self, url: str) -> bool:
        """
        Check if URL should be kept (not in filter list)
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be kept, False if it should be filtered
        """
        domain = self._get_domain(url)
        return domain not in self.useless_domains

    def _analyze_top_domains(self) -> Dict[str, int]:
        """
        Analyze domain frequency
        
        Returns:
            Dictionary of top 10 domains with their counts
        """
        return dict(self.domain_counter.most_common(10))

    def _save_domain_stats(self):
        """Save all domain statistics to files"""
        stats_file = "domain_statistics.txt"
        filtered_stats_file = "filtered_domain_statistics.txt"
        
        print(f"\nSaving domain statistics to {stats_file}")
        
        # Save statistics for retained domains
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Domain,Count,Percentage\n")
            total_urls = sum(self.domain_counter.values())
            
            for domain, count in sorted(self.domain_counter.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_urls) * 100
                f.write(f"{domain},{count},{percentage:.2f}%\n")
        
        # Save list of filtered domains
        with open(filtered_stats_file, 'w', encoding='utf-8') as f:
            f.write("# Filtered domains that were found in the dataset:\n")
            filtered_domains = sorted(self.useless_domains)
            for domain in filtered_domains:
                f.write(f"{domain}\n")

    def _plot_domain_histogram(self, domain_stats: Dict[str, int]):
        """
        Plot histogram of top 10 domains
        
        Args:
            domain_stats: Dictionary of domains and their counts
        """
        plt.figure(figsize=(15, 7))
        
        domains = list(domain_stats.keys())
        counts = list(domain_stats.values())
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(domains)))
        
        bars = plt.bar(range(len(domains)), counts, color=colors)
        
        plt.title('Top 10 Most Frequent Domains (After Filtering)', fontsize=14, pad=20)
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        plt.xticks(range(len(domains)), domains, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('domain_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _calculate_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on system memory
        
        Returns:
            Optimal batch size
        """
        available_memory = self.memory.available / (1024 * 1024 * 1024)  # GB
        return min(500, max(100, int(available_memory * 1024 / self.max_workers)))
        
    def get_all_json_files(self) -> List[str]:
        """
        Get all JSON files from input directories
        
        Returns:
            List of paths to JSON files
        """
        all_files = []
        for input_dir in self.input_dirs:
            json_pattern = os.path.join(input_dir, "*.json")
            files = glob.glob(json_pattern)
            all_files.extend(files)
            print(f"Found {len(files):,} JSON files in {input_dir}")
        return all_files

    def _print_stats(self):
        """Print processing statistics and domain frequency analysis"""
        print("\nProcessing completed:")
        print(f"Directories processed: {self.stats.total_dirs:,}")
        print(f"Files processed: {self.stats.processed_files:,}")
        print(f"Total URLs: {self.stats.total_urls:,}")
        print(f"Filtered URLs: {self.stats.filtered_urls:,}")
        print(f"Retained Unique URLs: {self.stats.unique_urls:,}")
        print(f"Filtering rate: {(self.stats.filtered_urls / self.stats.total_urls * 100):.2f}%")

        print("\nDomain Frequency Analysis:")
        print("Top 10 most frequent retained domains:")
        
        domain_stats = self._analyze_top_domains()
        total_domains = len(self.domain_counter)
        
        print(f"\nTotal distinct domains retained: {total_domains:,}")
        
        for domain, count in domain_stats.items():
            percentage = (count / self.stats.unique_urls) * 100
            print(f"{domain}: {count:,} ({percentage:.2f}%)")

        self._plot_domain_histogram(domain_stats)
        print("\nHistogram has been saved as 'domain_frequency.png'")
        
        self._save_domain_stats()

    def process_all_files(self):
        """Process all files to extract, normalize, and filter URLs"""
        json_files = self.get_all_json_files()
        self.stats.total_files = len(json_files)
        print(f"\nTotal {self.stats.total_files:,} JSON files found in {self.stats.total_dirs} directories")
        print(f"Using {self.max_workers} workers with batch size {self.batch_size}")
        
        if not json_files:
            print("No JSON files found!")
            return
        
        batches = [json_files[i:i + self.batch_size] for i in range(0, len(json_files), self.batch_size)]
        
        with tqdm(total=len(json_files), desc="Processing files") as pbar:
            with Pool(processes=self.max_workers, initializer=init_worker) as pool:
                try:
                    for files_processed, urls in pool.imap_unordered(process_batch, batches):
                        for url in urls:
                            normalized_url = normalize_url(url)
                            
                            # Check if this URL should be filtered
                            if self._should_keep_url(normalized_url):
                                if normalized_url not in self.url_set:
                                    self.url_set.add(normalized_url)
                                    domain = self._get_domain(normalized_url)
                                    if domain:
                                        self.domain_counter[domain] += 1
                            else:
                                self.stats.filtered_urls += 1
                        
                        self.stats.processed_files += files_processed
                        self.stats.total_urls += len(urls)
                        self.stats.unique_urls = len(self.url_set)
                        pbar.update(files_processed)
                except KeyboardInterrupt:
                    print("\nGracefully shutting down workers...")
                    pool.terminate()
                    pool.join()
                    raise
        
        print("\nWriting unique URLs to file...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for url in self.url_set:
                f.write(f"{url}\n")
        
        self._print_stats()

def main():
    input_dirs = [
        "data/forum_urls",
        "data/geometry_urls",
        "data/stemez_urls",
    ]
    output_file = "unique_urls.txt"
    useless_domains_file = "useless_domain.txt"  # New: file of domains to filter
    
    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        processor = MultiDirURLProcessor(input_dirs, output_file, useless_domains_file)
        processor.process_all_files()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()