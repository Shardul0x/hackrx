# Use an official Python runtime as a parent image
# Using a slim-buster image for smaller size
FROM python:3.9-slim-buster # You can use 3.10-slim-buster or higher if preferred

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by unstructured, pdf2image, etc.
# These are crucial for the underlying libraries to function, even if you
# are primarily using pre-extracted texts for indexing.
# `poppler-utils` for PDF processing (used by pdf2image, unstructured)
# `tesseract-ocr` for OCR (used by unstructured)
# `libmagic-dev` for file type detection (used by unstructured)
# `libreoffice` (optional, for MS Office docs, might not be strictly needed if only using .txt)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    libreoffice \
    # Clean up apt cache to keep image size down
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure the 'extracted_texts' directory exists and contains your .txt files
# This assumes you have already run your text extraction scripts locally
# and committed the 'extracted_texts' folder to your repository.
# If this folder is not in your repo, the app will fail to find documents.

# Expose the port the app runs on (Render injects $PORT)
EXPOSE $PORT

# Command to run the application using Uvicorn
# 'main:app' assumes your FastAPI app instance is named 'app' in 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
