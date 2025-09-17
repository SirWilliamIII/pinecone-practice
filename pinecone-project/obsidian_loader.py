#!/usr/bin/env python3
import os
from PIL import Image
import PyPDF2


def chunk_text(text, chunk_size=1000, overlap=200):
    """Chunk text into smaller pieces with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_obsidian_vault():
    """Load all supported files from your Obsidian vault recursively and chunk them."""
    docs = []
    base_path = os.path.expanduser("/Users/will/Markdown")
    doc_id_counter = 1

    supported_files = []
    supported_extensions = [".md", ".pdf", ".png", ".jpg", ".jpeg", ".txt"]

    for root, dirs, files in os.walk(base_path):
        print(f"Scanning directory: {root}")
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_path)
                supported_files.append((full_path, relative_path))
                print(f"  Found: {relative_path}")

    print(f"Found {len(supported_files)} supported files")

    for file_path, relative_filename in supported_files:
        try:
            # Extract category from folder structure
            parts = relative_filename.split('/')
            if len(parts) > 1:
                category = parts[0]
            else:
                category = "general"

            # Determine file type and extract content
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == ".md" or file_ext == ".txt":
                # Text files
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                chunks = chunk_text(content)

                for i, chunk in enumerate(chunks):
                    docs.append({
                        "id": f"text_{doc_id_counter}",
                        "text": chunk,
                        "filename": relative_filename,
                        "category": category,
                        "file_type": "text",
                        "full_path": file_path,
                    })
                    doc_id_counter += 1

            elif file_ext == ".pdf":
                # PDF files
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"

                if content.strip():
                    chunks = chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        docs.append({
                            "id": f"pdf_{doc_id_counter}",
                            "text": chunk,
                            "filename": relative_filename,
                            "category": category,
                            "file_type": "pdf",
                            "full_path": file_path,
                        })
                        doc_id_counter += 1

            elif file_ext in [".png", ".jpg", ".jpeg"]:
                # Image files - we'll store the path for CLIP to process
                docs.append({
                    "id": f"image_{doc_id_counter}",
                    "text": f"Image file: {relative_filename}",  # Placeholder text
                    "filename": relative_filename,
                    "category": category,
                    "file_type": "image",
                    "full_path": file_path,
                })
                doc_id_counter += 1

        except Exception as e:
            print(f"Error processing {relative_filename}: {e}")

    print(f"Successfully loaded and chunked {len(docs)} documents")
    return docs
