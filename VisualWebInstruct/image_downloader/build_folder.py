"""
Directory Structure Creator - A utility for replicating directory structures and converting JSON files to folders

INPUT:
- source_path: Path to the source directory containing the original structure with JSON files
- target_path: Path to the target directory where the new structure will be created

OUTPUT:
- Mirrored directory structure in the target location
- For each JSON file in the source, a corresponding directory is created in the target
- Console output indicating completion status or errors
"""

import os
import json

def create_directory_structure(source_path: str, target_path: str):
    """
    Copy directory structure and convert JSON files to folders
    
    :param source_path: Source directory path
    :param target_path: Target directory path
    """
    def process_directory(current_src_path: str, current_target_path: str):
        # Ensure target directory exists
        if not os.path.exists(current_target_path):
            os.makedirs(current_target_path)
            # print(f"Created directory: {current_target_path}")
            
        # Iterate through all contents in the current directory
        for item in os.listdir(current_src_path):
            src_item_path = os.path.join(current_src_path, item)
            
            # If it's a JSON file, create a corresponding directory
            if item.endswith('.json'):
                # Remove .json extension for new directory name
                new_dir_name = os.path.splitext(item)[0]
                new_dir_path = os.path.join(current_target_path, new_dir_name)
                
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                    # print(f"Created directory for JSON: {new_dir_path}")
                    
            # If it's a directory, process recursively
            elif os.path.isdir(src_item_path):
                target_item_path = os.path.join(current_target_path, item)
                process_directory(src_item_path, target_item_path)

    # Start processing
    try:
        process_directory(source_path, target_path)
        print("Directory structure creation completed!")
    except Exception as e:
        print(f"Error creating directory structure: {str(e)}")

def main():
    # Set source and target directories
    source_path = "./downloaded_html"  # Your html directory path
    target_path = "./downloaded_images"  # Target downloaded_images directory path
    
    # Create directory structure
    create_directory_structure(source_path, target_path)

if __name__ == "__main__":
    main()