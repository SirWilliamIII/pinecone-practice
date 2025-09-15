#!/usr/bin/env python3
import os

def load_obsidian_vault():
    """Load all markdown files from your Obsidian vault recursively"""
    docs = []
    base_path = os.path.expanduser("~/Markdown")

    md_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.md'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_path)
                md_files.append((full_path, relative_path))

    print(f"Found {len(md_files)} markdown files")

    for i, (file_path, relative_filename) in enumerate(md_files):

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            docs.append({
                "id": f"md_{i+1}",
                "text": content[:3000],  
                "filename": relative_filename.replace('.md', ''), 
                "category": "obsidian_note",
                "full_path": file_path
            })

        except Exception as e:
            print(f"Error reading {relative_filename}: {e}")

    print(f"Successfully loaded {len(docs)} documents")
    return docs


vault = load_obsidian_vault()
print(len(vault))

