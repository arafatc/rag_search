#!/usr/bin/env python3
"""Revert Phoenix prompt integration"""
import os
import shutil
from pathlib import Path

def revert_changes():
    print(" Reverting Phoenix prompt integration...")
    
    # Files to remove
    files_to_remove = [
        "src/rag_search/phoenix_prompts.py",
        "src/prompts/prompts.json",
        "scripts/init_phoenix_prompts.py"
    ]
    
    removed_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"SUCCESS: Removed: {file_path}")
            removed_count += 1
    
    # Remove empty directories
    dirs_to_check = ["src/prompts", "scripts"]
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"SUCCESS: Removed empty directory: {dir_path}")
    
    print(f"SUCCESS: Phoenix prompt integration reverted ({removed_count} files removed)")
    print("WARNING:  Note: You'll need to manually revert changes to src/rag_search/crew.py")
    print("    or restore from git if you want complete rollback")

if __name__ == "__main__":
    revert_changes()
