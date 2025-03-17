import json
import os
from serpapi import GoogleSearch
from datetime import datetime
import time


APIKEY = ""
output_dir = "data/forum_urls"
os.makedirs(output_dir, exist_ok=True)

for i in range(12703):
    try:
        # Build image link
        link = r"https://github.com/jymmmmm/VISUALWEBINSTRUCT/blob/master/data/forum/" + str(i) + ".png" + r"?raw=true"
        
        params = {
            "api_key": APIKEY,
            "engine": "google_lens",
            "url": link
        }

        # Execute search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Build output filename
        output_file = os.path.join(output_dir, f"result_{i}.json")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved search result {i}")
        
        # Add delay to avoid rate limiting
        time.sleep(1)  # 1 seconds delay between requests
        
    except Exception as e:
        print(f"Error processing index {i}: {str(e)}")
        # Log error
        error_file = os.path.join(output_dir, "errors.log")
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"Index {i}: {str(e)}\n")
        continue



output_dir = "data/geometry_urls"
os.makedirs(output_dir, exist_ok=True)

for i in range(10000):
    try:
        # Build image link
        link = r"https://github.com/jymmmmm/VISUALWEBINSTRUCT/blob/master/data/geometry/" + str(i) + ".png" + r"?raw=true"
        
        params = {
            "api_key": APIKEY,
            "engine": "google_lens",
            "url": link
        }

        # Execute search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Build output filename
        output_file = os.path.join(output_dir, f"result_{i}.json")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved search result {i}")
        
        # Add delay to avoid rate limiting
        time.sleep(1)  # 1 seconds delay between requests
        
    except Exception as e:
        print(f"Error processing index {i}: {str(e)}")
        # Log error
        error_file = os.path.join(output_dir, "errors.log")
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"Index {i}: {str(e)}\n")
        continue


output_dir = "data/stemez_urls"
os.makedirs(output_dir, exist_ok=True)

for i in range(925):
    try:
        # Build image link
        link = r"https://github.com/jymmmmm/VISUALWEBINSTRUCT/blob/master/data/stemez/" + str(i) + ".png" + r"?raw=true"
        
        params = {
            "api_key": APIKEY,
            "engine": "google_lens",
            "url": link
        }

        # Execute search
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Build output filename
        output_file = os.path.join(output_dir, f"result_{i}.json")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved search result {i}")
        
        # Add delay to avoid rate limiting
        time.sleep(1)  # 1 seconds delay between requests
        
    except Exception as e:
        print(f"Error processing index {i}: {str(e)}")
        # Log error
        error_file = os.path.join(output_dir, "errors.log")
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"Index {i}: {str(e)}\n")
        continue

print("Search completed!")