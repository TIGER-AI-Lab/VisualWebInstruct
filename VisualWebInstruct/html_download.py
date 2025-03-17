#!/usr/bin/env python3
"""
HTML Downloader - A high-performance tool for concurrent web page downloads

INPUT:
- url_file: A text file containing URLs to download (one URL per line)
- domain_stats_file: A CSV file with domain statistics (used to skip high-traffic domains)
- output_dir: Directory to save downloaded HTML content

OUTPUT:
- Downloaded HTML files saved in a hierarchical directory structure
- JSON files containing the URL, HTML content, and timestamp
- Download statistics in JSON format
- Detailed logs of the download process
"""

import asyncio
import aiohttp
from urllib.parse import urlparse
from pathlib import Path
import logging
import time
from collections import Counter, defaultdict
from typing import List, Set, Dict, Optional
import os
import json
import hashlib
from aiohttp import ClientTimeout, TCPConnector
from tqdm.asyncio import tqdm
from datetime import datetime
import chardet
import signal
from itertools import cycle
import resource
from multiprocessing import Pool, cpu_count
import asyncio
import sys
import multiprocessing as mp
from itertools import zip_longest

class ProxyRotator:
    def __init__(self, proxies: List[str]):
        """
        Initialize a proxy rotator to cycle through available proxies.
        
        Args:
            proxies: List of proxy URLs
        """
        self.proxies = cycle(proxies)
        self.lock = asyncio.Lock()
    
    async def get_proxy(self) -> str:
        """
        Get the next proxy in the rotation in a thread-safe manner.
        
        Returns:
            Next proxy URL
        """
        async with self.lock:
            return next(self.proxies)

class OptimizedHTMLDownloader:
    def __init__(self, 
                 url_file: str,
                 domain_stats_file: str,
                 output_dir: str,
                 max_concurrent: int = 500,
                 timeout: int = 30,
                 skip_top_domains: int = 50,
                 max_retries: int = 3,
                 batch_size: int = 10000,
                 max_domain_concurrent: int = 10,
                 save_batch_size: int = 1000):
        """
        Initialize the HTML downloader with configuration settings.
        
        Args:
            url_file: Path to the file containing URLs to download
            domain_stats_file: Path to the file containing domain statistics
            output_dir: Directory to save the downloaded HTML files
            max_concurrent: Maximum number of concurrent connections
            timeout: HTTP request timeout in seconds
            skip_top_domains: Number of top domains to skip
            max_retries: Maximum number of retry attempts for failed downloads
            batch_size: Number of URLs to process in a batch
            max_domain_concurrent: Maximum concurrent connections per domain
            save_batch_size: Number of HTML files to save in a batch
        """
        self.url_file = url_file
        self.domain_stats_file = domain_stats_file
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.skip_top_domains = skip_top_domains
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_domain_concurrent = max_domain_concurrent
        self.save_batch_size = save_batch_size
        
        self.stats = {
            'total_urls': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'retried': 0,
            'errors': Counter()
        }
        
        self._create_directories()
        
        self._setup_logging()
        
        self.urls: Set[str] = set()
        self.top_domains: Set[str] = set()
        self.domain_semaphores: Dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(self.max_domain_concurrent)
        )
        
        self._setup_resource_limits()
        
        self.pbar = None
        
        # Signal handling
        self._setup_signal_handlers()

    def _create_directories(self):
        """Create the necessary directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir = self.output_dir / 'html'
        self.html_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        """Set up logging configuration"""
        # Create a unique log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'download_{timestamp}.log'
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Get logger and configure
        self.logger = logging.getLogger(f'downloader_{timestamp}')
        self.logger.setLevel(logging.INFO)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent log propagation to root logger
        self.logger.propagate = False
        
        self.logger.info("Logger initialized")
        self.logger.info(f"Log file: {log_file}")

    def _setup_resource_limits(self):
        """Set system resource limits"""
        # Increase file descriptor limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (50000, 50000))
        # Set process priority
        os.nice(10)

    def _setup_signal_handlers(self):
        """Set up signal handlers for clean shutdown"""
        def signal_handler(signum, frame):
            self.logger.info("\nReceived signal to terminate. Saving stats...")
            self._save_stats()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _get_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url

    def _load_top_domains(self):
        """Load domains to skip from the domain statistics file"""
        try:
            with open(self.domain_stats_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header
                domain_count = 0
                for line in f:
                    if domain_count >= self.skip_top_domains:
                        break
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        self.top_domains.add(parts[0].strip())
                        domain_count += 1
            
            self.logger.info(f"Loaded {len(self.top_domains)} domains to skip")
            self.logger.info(f"Top domains to skip: {', '.join(list(self.top_domains)[:5])}...")
        except Exception as e:
            self.logger.error(f"Error loading domain stats file: {e}")
            raise

    def _load_urls(self):
        """Load URLs to process from the URL file"""
        try:
            with open(self.url_file, 'r', encoding='utf-8') as f:
                self.urls = {line.strip() for line in f if line.strip()}
            
            self.stats['total_urls'] = len(self.urls)
            self.logger.info(f"Loaded {len(self.urls):,} URLs")
        except Exception as e:
            self.logger.error(f"Error loading URLs: {e}")
            raise

    def _group_urls_by_domain(self) -> Dict[str, List[str]]:
        """
        Group URLs by domain for better rate limiting.
        
        Returns:
            Dictionary mapping domains to lists of URLs
        """
        domain_urls = defaultdict(list)
        for url in self.urls:
            domain = self._get_domain(url)
            if domain not in self.top_domains:
                domain_urls[domain].append(url)
        return domain_urls

    async def _download_url_with_retry(self, 
                                     session: aiohttp.ClientSession, 
                                     url: str, 
                                     domain_semaphore: asyncio.Semaphore) -> tuple:
        """
        Download URL with retry mechanism.
        
        Args:
            session: aiohttp client session
            url: URL to download
            domain_semaphore: Semaphore for domain-specific rate limiting
            
        Returns:
            Tuple of (url, html_content, error_message)
        """
        for attempt in range(self.max_retries):
            try:
                async with domain_semaphore:
                    return await self._download_url(session, url)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.debug(f"Failed to download {url} after {self.max_retries} attempts: {str(e)}")
                    return url, None, str(e)
                self.stats['retried'] += 1
                await asyncio.sleep(1 * (attempt + 1))

    async def _download_url(self, session: aiohttp.ClientSession, url: str) -> tuple:
        """
        Download the content of a single URL.
        
        Args:
            session: aiohttp client session
            url: URL to download
            
        Returns:
            Tuple of (url, html_content, error_message)
        """
        try:
            async with session.get(url, timeout=ClientTimeout(total=self.timeout), ssl=False) as response:
                if response.status == 200:
                    content = await response.read()
                    encoding = chardet.detect(content)['encoding'] or 'utf-8'
                    try:
                        html = content.decode(encoding)
                        self.stats['successful'] += 1
                        return url, html, None
                    except UnicodeDecodeError:
                        self.stats['failed'] += 1
                        error_msg = f"Decode error with encoding {encoding}"
                        self.stats['errors'][error_msg] += 1
                        return url, None, error_msg
                else:
                    self.stats['failed'] += 1
                    error_msg = f"HTTP {response.status}"
                    self.stats['errors'][error_msg] += 1
                    return url, None, error_msg
        except Exception as e:
            self.stats['failed'] += 1
            error_msg = str(e)[:100]
            self.stats['errors'][error_msg] += 1
            return url, None, error_msg

    async def _save_batch(self, results: List[tuple]):
        """
        Save a batch of download results.
        
        Args:
            results: List of (url, html_content, error_message) tuples
        """
        for i in range(0, len(results), self.save_batch_size):
            batch = results[i:i + self.save_batch_size]
            save_tasks = []
            for url, html, error in batch:
                if html is not None:
                    save_tasks.append(self._save_html(url, html))
            if save_tasks:
                await asyncio.gather(*save_tasks)

    async def _save_html(self, url: str, html: str):
        """
        Save HTML content using a two-level directory structure.
        
        Args:
            url: The URL from which the HTML was downloaded
            html: The HTML content
        """
        if html is None:
            return
            
        url_hash = hashlib.md5(url.encode()).hexdigest()
        level1_dir = self.html_dir / url_hash[:2]
        level1_dir.mkdir(exist_ok=True)
        level2_dir = level1_dir / url_hash[2:4]
        level2_dir.mkdir(exist_ok=True)
        
        file_path = level2_dir / f"{url_hash}.json"
        
        data = {
            'url': url,
            'html': html,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Use async file operations to write the file
            async def write_file():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: file_path.write_text(
                        json.dumps(data, ensure_ascii=False), 
                        encoding='utf-8'
                    )
                )
            
            await write_file()
            
        except Exception as e:
            self.logger.error(f"Failed to save {url} to {file_path}: {str(e)}")
            raise

    def _save_stats(self):
        """Save download statistics to a JSON file"""
        stats_file = self.output_dir / 'download_stats.json'
        self.stats['timestamp'] = datetime.now().isoformat()
        self.stats['errors'] = dict(self.stats['errors'])
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

    async def _process_domain_urls(self, 
                                 session: aiohttp.ClientSession,
                                 domain: str, 
                                 urls: List[str]):
        """
        Process all URLs for a single domain.
        
        Args:
            session: aiohttp client session
            domain: The domain name
            urls: List of URLs for this domain
            
        Returns:
            Results of the download operations
        """
        semaphore = self.domain_semaphores[domain]
        tasks = [self._download_url_with_retry(session, url, semaphore) 
                for url in urls]
        return await asyncio.gather(*tasks)

    async def download_all(self):
        """Download all URLs"""
        self._load_top_domains()
        self._load_urls()
        
        # Group by domain
        domain_urls = self._group_urls_by_domain()
        
        # Set up connection pool
        connector = TCPConnector(
            limit=self.max_concurrent,
            force_close=True,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
            use_dns_cache=True,
            verify_ssl=False
        )
        
        # Session configuration
        timeout = ClientTimeout(total=self.timeout, connect=10)
        session_kwargs = {
            'connector': connector,
            'timeout': timeout,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
        }
        
        start_time = time.time()
        total_urls = sum(len(urls) for urls in domain_urls.values())
        
        async with aiohttp.ClientSession(**session_kwargs) as session:
            self.pbar = tqdm(total=total_urls, desc="Downloading URLs")
            
            for domain, urls in domain_urls.items():
                for i in range(0, len(urls), self.batch_size):
                    batch_urls = urls[i:i + self.batch_size]
                    results = await self._process_domain_urls(session, domain, batch_urls)
                    await self._save_batch(results)
                    self.pbar.update(len(batch_urls))
                    await asyncio.sleep(0.1)  # Avoid too aggressive requests
            
            self.pbar.close()
        
        total_time = time.time() - start_time
        
        self.logger.info(f"\nDownload completed in {total_time:.2f} seconds")
        self.logger.info(f"Total URLs: {self.stats['total_urls']:,}")
        self.logger.info(f"Successful: {self.stats['successful']:,}")
        self.logger.info(f"Failed: {self.stats['failed']:,}")
        self.logger.info(f"Skipped: {self.stats['skipped']:,}")
        self.logger.info(f"Retried: {self.stats['retried']:,}")
        
        self._save_stats()

def grouper(iterable, n, fillvalue=None):
    """
    Group an iterable into chunks of size n.
    
    Args:
        iterable: The iterable to group
        n: Size of each group
        fillvalue: Value to use for missing values in the last group
        
    Returns:
        Iterator yielding groups of the iterable
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class MultiProcessDownloader:
    def __init__(self, 
                 url_file: str,
                 domain_stats_file: str,
                 output_dir: str,
                 num_processes: int = None,
                 **kwargs):
        """
        Initialize a multi-process downloader.
        
        Args:
            url_file: Path to the file containing URLs to download
            domain_stats_file: Path to the file containing domain statistics
            output_dir: Directory to save the downloaded HTML files
            num_processes: Number of worker processes to spawn
            **kwargs: Additional arguments to pass to OptimizedHTMLDownloader
        """
        self.url_file = url_file
        self.domain_stats_file = domain_stats_file
        self.output_dir = output_dir
        self.num_processes = num_processes or mp.cpu_count()
        self.downloader_kwargs = kwargs
        
        # Set up shared counters
        self.successful = mp.Value('i', 0)
        self.failed = mp.Value('i', 0)
        self.total = mp.Value('i', 0)
        
        # Create process lock
        self.lock = mp.Lock()

    def _load_and_group_urls(self):
        """
        Load and group URLs by domain.
        
        Returns:
            Dictionary mapping domains to lists of URLs
        """
        downloader = OptimizedHTMLDownloader(
            self.url_file,
            self.domain_stats_file,
            self.output_dir,
            **self.downloader_kwargs
        )
        
        downloader._load_top_domains()
        downloader._load_urls()
        
        return downloader._group_urls_by_domain()

    def _split_work(self, domain_urls):
        """
        Split domains among worker processes to balance workload.
        
        Args:
            domain_urls: Dictionary mapping domains to lists of URLs
            
        Returns:
            List of domain lists for each worker process
        """
        domains = list(domain_urls.items())
        urls_per_domain = [(domain, len(urls)) for domain, urls in domains]
        urls_per_domain.sort(key=lambda x: x[1], reverse=True)
        
        process_work = [[] for _ in range(self.num_processes)]
        total_urls = [0] * self.num_processes
        
        for domain, urls in domains:
            min_idx = total_urls.index(min(total_urls))
            process_work[min_idx].append(domain)
            total_urls[min_idx] += len(domain_urls[domain])
        
        return process_work

    def _process_worker(self, domains, domain_urls, process_id):
        """
        Worker process function to handle a subset of domains.
        
        Args:
            domains: List of domains to process
            domain_urls: Dictionary mapping domains to lists of URLs
            process_id: Process identifier
        """
        downloader = OptimizedHTMLDownloader(
            self.url_file,
            self.domain_stats_file,
            self.output_dir,
            **self.downloader_kwargs
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        process_urls = {
            domain: domain_urls[domain]
            for domain in domains
            if domain in domain_urls
        }
        
        try:
            loop.run_until_complete(self._run_downloads(downloader, process_urls, process_id))
        finally:
            loop.close()

    async def _run_downloads(self, downloader, domain_urls, process_id):
        """
        Run downloads for a subset of domains in a worker process.
        
        Args:
            downloader: OptimizedHTMLDownloader instance
            domain_urls: Dictionary mapping domains to lists of URLs
            process_id: Process identifier
        """
        connector = TCPConnector(
            limit=downloader.max_concurrent,
            force_close=True,
            enable_cleanup_closed=True,
            ttl_dns_cache=300
        )
        
        timeout = ClientTimeout(total=downloader.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            total_urls = sum(len(urls) for urls in domain_urls.values())
            
            with tqdm(total=total_urls, desc=f"Process {process_id}", 
                     position=process_id) as pbar:
                for domain, urls in domain_urls.items():
                    for i in range(0, len(urls), downloader.batch_size):
                        batch = urls[i:i + downloader.batch_size]
                        results = await downloader._process_domain_urls(
                            session, domain, batch
                        )
                        await downloader._save_batch(results)
                        
                        success = sum(1 for _, html, _ in results if html is not None)
                        with self.lock:
                            self.successful.value += success
                            self.failed.value += len(batch) - success
                            self.total.value += len(batch)
                        
                        pbar.update(len(batch))
                        await asyncio.sleep(0.1)

    def start(self):
        """Start the multi-process download operation"""
        print(f"Starting download with {self.num_processes} processes")
        
        domain_urls = self._load_and_group_urls()
        process_work = self._split_work(domain_urls)
        
        processes = []
        for i, domains in enumerate(process_work):
            p = mp.Process(
                target=self._process_worker,
                args=(domains, domain_urls, i)
            )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        print("\nDownload completed!")
        print(f"Total URLs processed: {self.total.value}")
        print(f"Successful downloads: {self.successful.value}")
        print(f"Failed downloads: {self.failed.value}")
        print(f"Success rate: {(self.successful.value/self.total.value)*100:.2f}%")

def main():
    """Main entry point of the application"""
    url_file = "unique_urls.txt"
    domain_stats_file = "domain_statistics.txt"
    output_dir = "downloaded_html"
    
    config = {
    'max_concurrent': 100,
    'timeout': 30,
    'skip_top_domains': 50,
    'max_retries': 3,
    'batch_size': 1000,
    }

    downloader = MultiProcessDownloader(
        url_file=url_file,
        domain_stats_file=domain_stats_file,
        output_dir=output_dir,
        num_processes=8,  
        **config
    )

    downloader.start()

if __name__ == "__main__":
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    
    main()