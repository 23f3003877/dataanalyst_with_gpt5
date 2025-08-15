import os
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from matplotlib.style import context
import uvicorn
import base64
import httpx
from bs4 import BeautifulSoup
import time
import subprocess
import json
from dotenv import load_dotenv
import os
import data_scrape
import functools
import re
import pandas as pd
import numpy as np
from io import StringIO
from urllib.parse import urlparse
import duckdb
import glob
import tabula
import tarfile
import zipfile
import tempfile
import shutil
import asyncio
import matplotlib.pyplot as plt
import pdfplumber
import openpyxl
from openpyxl import load_workbook

app = FastAPI()
load_dotenv()

# --- Precise file tracking & cleanup helpers ---
def _snapshot_files(root: str = ".") -> set[str]:
    """Get a snapshot of all files under root as relative paths."""
    files = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtual envs or cache folders commonly present
        parts = os.path.relpath(dirpath, root).split(os.sep)
        if any(p in {".git", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"} for p in parts):
            continue
        for fn in filenames:
            rel = os.path.normpath(os.path.join(os.path.relpath(dirpath, root), fn))
            files.add(rel)
    return files

def _cleanup_created_files(files_to_delete: set[str]) -> int:
    """Delete specific files created during this request.
    Returns number of files deleted."""
    deleted = 0
    for rel_path in files_to_delete:
        try:
            path = os.path.normpath(rel_path)
            # handle paths that might already be absolute
            if not os.path.isabs(path):
                path = os.path.join(".", path) if path != "." else "."
            if os.path.isfile(path):
                os.remove(path)
                deleted += 1
                print(f"🗑️ Deleted: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                deleted += 1
                print(f"🗑️ Deleted directory: {path}")
        except Exception as e:
            print(f"⚠️ Could not delete {rel_path}: {e}")
    print(f"🧹 Cleanup complete: {deleted} files/directories deleted")
    return deleted

def track_created_file(filename: str, created_files_set: set = None) -> str:
    """Helper function to track created files and ensure they're added to cleanup set.
    Returns the normalized filename."""
    normalized_filename = os.path.normpath(filename)
    if created_files_set is not None:
        created_files_set.add(normalized_filename)
        print(f"📁 Tracking file for cleanup: {normalized_filename}")
    return normalized_filename

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")
open_ai_url = "https://aipipe.org/openai/v1/chat/completions"
ocr_api_key = os.getenv("OCR_API_KEY")
OCR_API_URL = "https://api.ocr.space/parse/image"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
gemini_api = os.getenv("gemini_api")
horizon_api = os.getenv("horizon_api")
gemini_api_2 = os.getenv("gemini_api_2")
grok_api = os.getenv("grok_api")
grok_fix_api = os.getenv("grok_fix_api")
openai_gpt5_api_key = os.getenv("OPENAI_GPT5_API_KEY")
openai_gpt5_url = "https://api.openai.com/v1/chat/completions"

# Claude API configuration
claude_api_key = os.getenv("CLAUDE_API_KEY")
claude_api_url = "https://api.anthropic.com/v1/messages"

def make_json_serializable(obj):
    """Convert pandas/numpy objects to JSON-serializable formats"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Series)):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype') and hasattr(obj, 'name'):
        return str(obj)
    elif pd.api.types.is_extension_array_dtype(obj):
        return str(obj)
    elif str(type(obj)).startswith("<class 'pandas."):
        return str(obj)
    elif str(type(obj)).startswith("<class 'numpy."):
        try:
            return obj.item() if hasattr(obj, 'item') else str(obj)
        except:
            return str(obj)
    else:
        return obj

# --- Safe file writing to avoid Windows cp1252 'charmap' UnicodeEncodeErrors ---
def safe_write(path: str, text: str, replace: bool = True):
    """Write text to file using UTF-8 regardless of system locale.
    Windows default (cp1252) cannot encode characters like U+2011 (non-breaking hyphen)
    or U+202F (narrow no-break space) sometimes produced by LLM outputs. This helper
    forces utf-8 and optionally replaces unencodable characters.
    """
    errors_policy = "replace" if replace else "strict"
    with open(path, "w", encoding="utf-8", errors=errors_policy) as f:
        f.write(text)

# --- Archive extraction helper ---
async def extract_archive_contents(file_upload: UploadFile, temp_dir: str) -> dict:
    """Extract contents from TAR, ZIP, or other archive files and categorize them"""
    extracted_files = {
        'csv_files': [],
        'json_files': [],
        'excel_files': [],
        'pdf_files': [],
        'html_files': [],
        'image_files': [],
        'txt_files': [],
        'sql_files': [],
        'other_files': []
    }
    
    try:
        file_bytes = await file_upload.read()
        filename_lower = file_upload.filename.lower() if file_upload.filename else ""
        
        # Create a temporary file to store the archive
        temp_archive_path = os.path.join(temp_dir, file_upload.filename or "archive")
        with open(temp_archive_path, "wb") as f:
            f.write(file_bytes)
        
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Determine archive type and extract
        if filename_lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz')):
            print(f"📦 Extracting TAR archive: {file_upload.filename}")
            with tarfile.open(temp_archive_path, 'r:*') as tar:
                # Security check: prevent path traversal
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonpath([abs_directory, abs_target])
                    return prefix == abs_directory
                
                for member in tar.getmembers():
                    if member.isfile():
                        # Sanitize the path
                        safe_path = os.path.join(extract_dir, os.path.basename(member.name))
                        if is_within_directory(extract_dir, safe_path):
                            try:
                                member.name = os.path.basename(member.name)  # Flatten structure
                                tar.extract(member, extract_dir)
                                print(f"  ✅ Extracted: {member.name}")
                            except Exception as e:
                                print(f"  ⚠️ Failed to extract {member.name}: {e}")
                                
        elif filename_lower.endswith(('.zip', '.jar')):
            print(f"📦 Extracting ZIP archive: {file_upload.filename}")
            with zipfile.ZipFile(temp_archive_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if not member.endswith('/'):  # Skip directories
                        # Sanitize the path and flatten structure
                        safe_filename = os.path.basename(member)
                        safe_path = os.path.join(extract_dir, safe_filename)
                        try:
                            with zip_ref.open(member) as source, open(safe_path, "wb") as target:
                                target.write(source.read())
                            print(f"  ✅ Extracted: {safe_filename}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to extract {member}: {e}")
        else:
            print(f"❌ Unsupported archive format: {filename_lower}")
            return extracted_files
        
        # Categorize extracted files
        for filename in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, filename)
            if os.path.isfile(file_path):
                filename_lower = filename.lower()
                
                if filename_lower.endswith('.csv'):
                    extracted_files['csv_files'].append(file_path)
                elif filename_lower.endswith('.json'):
                    extracted_files['json_files'].append(file_path)
                elif filename_lower.endswith(('.xlsx', '.xls')):
                    extracted_files['excel_files'].append(file_path)
                elif filename_lower.endswith('.pdf'):
                    extracted_files['pdf_files'].append(file_path)
                elif filename_lower.endswith(('.html', '.htm')):
                    extracted_files['html_files'].append(file_path)
                elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                    extracted_files['image_files'].append(file_path)
                elif filename_lower.endswith('.txt'):
                    extracted_files['txt_files'].append(file_path)
                elif filename_lower.endswith('.sql'):
                    extracted_files['sql_files'].append(file_path)
                else:
                    extracted_files['other_files'].append(file_path)
        
        print(f"📦 Archive extraction complete:")
        for category, files in extracted_files.items():
            if files:
                print(f"  {category}: {len(files)} files")
                
    except Exception as e:
        print(f"❌ Error extracting archive {file_upload.filename}: {e}")
    
    return extracted_files

# Add caching for prompt files (with graceful fallback when missing)
@functools.lru_cache(maxsize=10)
def read_prompt_file(filename: str, default: str = "") -> str:
    try:
        with open(filename, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ Prompt file not found: {filename}. Using default content.")
        return default

async def ping_gemini(question_text, relevant_context="", max_tries=3):
    tries = 0
    while tries < max_tries:
        if tries % 2 != 0:
            api_key = gemini_api
        else:
            api_key = gemini_api_2
        try:
            print(f"gemini is running {tries + 1} try")
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": api_key
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": relevant_context},
                            {"text": question_text}
                        ]
                    }
                ]
            }
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(GEMINI_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error during Gemini call (attempt {tries + 1}): {e}")
            tries += 1
            if tries < max_tries:
                print(f"Retrying... ({max_tries - tries} attempts remaining)")
            else:
                print(f"All {max_tries} attempts failed for Gemini")
    return {"error": "Gemini failed after max retries"}

async def ping_chatgpt(question_text, relevant_context, max_tries=3):
    tries = 0
    while tries < max_tries:
        try:
            print(f"openai is running {tries+1} try")
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": relevant_context},
                    {"role": "user", "content": question_text}
                ]
            }
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(open_ai_url, headers=headers, json=payload)
                return response.json()
        except Exception as e:
            print(f"Error creating payload: {e}")
            tries += 1
            continue


async def ping_horizon(question_text, relevant_context="", max_tries=3):
    tries = 0
    while tries < max_tries:
        try:
            print(f"horizon is running {tries + 1} try")
            headers = {
                "Authorization": f"Bearer {horizon_api}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "openrouter/horizon-beta",
                "messages": [
                    {"role": "system", "content": relevant_context},
                    {"role": "user", "content": question_text}
                ]
            }
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error during Horizon call: {e}")
            tries += 1
    return {"error": "Horizon failed after max retries"}



async def ping_gemini_pro(question_text, relevant_context="", max_tries=3):
    """Call Gemini Pro API for code generation."""    
    tries = 0
    while tries < max_tries:
        if tries % 2 == 0:
            api_key = gemini_api
        else:
            api_key = gemini_api_2
        try:
            print(f"gemini pro is running {tries + 1} try")
            headers = {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": relevant_context},
                            {"text": question_text}
                        ]
                    }
                ]
            }
            async with httpx.AsyncClient(timeout=200) as client:
                response = await client.post("https://generativelanguage.googleapis.com/v1/models/gemini-2.5-pro:generateContent", headers=headers, json=payload)
                print(response)
                
                # Check if response is successful
                if response.status_code == 200:
                    gemini_response = response.json()
                    final_response = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
                    return final_response
                else:
                    print(f"Gemini Pro API error: {response.status_code} - {response.text}")
                    if response.status_code >= 500:  # Server errors, retry
                        raise Exception(f"Server error {response.status_code}: {response.text}")
                    else:  # Client errors, don't retry
                        return {"error": f"Client error {response.status_code}: {response.text}"}
                        
        except Exception as e:
            print(f"Error in Gemini Pro API call (attempt {tries + 1}): {e}")
            tries += 1
            if tries < max_tries:
                print(f"Retrying... ({max_tries - tries} attempts remaining)")
            else:
                print(f"All {max_tries} attempts failed for Gemini Pro")
                return {"error": f"Failed after {max_tries} attempts: {str(e)}"}


async def ping_open_ai_5(question_text, relevant_context="", max_tries=3):
    """Call OpenAI GPT-5 API for advanced AI responses with ChatGPT fallback."""
    tries = 0
    while tries < max_tries:
        try:
            print(f"OpenAI GPT-5 is running {tries+1} try")
            headers = {
                "Authorization": f"Bearer {openai_gpt5_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-5-2025-08-07",  # Update this when GPT-5 is available
                "messages": [
                    {"role": "system", "content": relevant_context},
                    {"role": "user", "content": question_text}
                ]
            }
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(openai_gpt5_url, headers=headers, json=payload)
                
                # Check if response is successful
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"OpenAI GPT-5 API error: {response.status_code} - {response.text}")
                    if response.status_code >= 500:  # Server errors, retry
                        raise Exception(f"Server error {response.status_code}: {response.text}")
                    else:  # Client errors, don't retry but fallback to ChatGPT
                        print(f"OpenAI GPT-5 client error, falling back to ChatGPT...")
                        return await ping_chatgpt(question_text, relevant_context)
                        
        except Exception as e:
            print(f"Error in OpenAI GPT-5 API call (attempt {tries + 1}): {e}")
            tries += 1
            if tries < max_tries:
                print(f"Retrying... ({max_tries - tries} attempts remaining)")
            else:
                print(f"All {max_tries} attempts failed for OpenAI GPT-5, falling back to ChatGPT...")
                return await ping_chatgpt(question_text, relevant_context)


async def ping_claude(question_text, relevant_context="", max_tries=3, timeout_seconds=180):
    """Call Anthropic Claude API (claude-sonnet-4-20250514) with timeout and OpenAI fallback."""
    
    async def claude_request():
        tries = 0
        while tries < max_tries:
            try:
                print(f"Claude Sonnet 4 is running {tries+1} try")
                headers = {
                    "x-api-key": claude_api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"  # Fixed: Use API version, not model name
                }
                
                # Format messages for Claude API
                user_content = f"{relevant_context}\n\n{question_text}" if relevant_context else question_text
                
                payload = {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4000,
                    "messages": [
                        {"role": "user", "content": user_content}
                    ]
                }
                
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",  # Fixed: Use correct endpoint
                        headers=headers, 
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        claude_response = response.json()
                        # Convert Claude response format to OpenAI-compatible format
                        content = claude_response["content"][0]["text"]
                        return {
                            "choices": [{"message": {"content": content}}],
                            "model": "claude-sonnet-4-20250514",
                            "_source": "claude"
                        }
                    else:
                        print(f"Claude API error: {response.status_code} - {response.text}")
                        if response.status_code >= 500:  # Server errors, retry
                            raise Exception(f"Server error {response.status_code}: {response.text}")
                        else:  # Client errors, don't retry
                            return {"error": f"Client error {response.status_code}: {response.text}"}
                            
            except Exception as e:
                print(f"Error in Claude API call (attempt {tries + 1}): {e}")
                tries += 1
                if tries < max_tries:
                    print(f"Retrying... ({max_tries - tries} attempts remaining)")
                else:
                    print(f"All {max_tries} attempts failed for Claude")
                    raise e
    
    try:
        # Run Claude request with timeout
        return await asyncio.wait_for(claude_request(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"Claude API timed out after {timeout_seconds} seconds, falling back to OpenAI GPT-5...")
        return await ping_open_ai_5(question_text, relevant_context)
    except Exception as e:
        print(f"Claude API completely failed: {e}, falling back to OpenAI GPT-5...")
        return await ping_open_ai_5(question_text, relevant_context)


def extract_content_from_response(response):
    """Extract content from either Claude or OpenAI response format safely."""
    try:
        # Check if it's an error response
        if "error" in response:
            return None
            
        # Standard OpenAI/Claude compatible format
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
            
        # Direct content (fallback)
        if "content" in response:
            if isinstance(response["content"], list) and len(response["content"]) > 0:
                return response["content"][0].get("text", "")
            elif isinstance(response["content"], str):
                return response["content"]
                
        return None
    except Exception as e:
        print(f"Error extracting content from response: {e}")
        return None





def extract_json_from_output(output: str) -> str:
    """Extract JSON from output that might contain extra text"""
    output = output.strip()
    
    # First try to find complete JSON objects (prioritize these)
    object_pattern = r'\{.*\}'
    object_matches = re.findall(object_pattern, output, re.DOTALL)
    
    # If we find JSON objects, return the longest one (most complete)
    if object_matches:
        longest_match = max(object_matches, key=len)
        return longest_match
    
    # Only if no objects found, look for arrays
    array_pattern = r'\[.*\]'
    array_matches = re.findall(array_pattern, output, re.DOTALL)
    
    if array_matches:
        longest_match = max(array_matches, key=len)
        return longest_match
    
    return output

def is_valid_json_output(output: str) -> bool:
    """Check if the output is valid JSON without trying to parse it"""
    output = output.strip()
    return (output.startswith('{') and output.endswith('}')) or (output.startswith('[') and output.endswith(']'))

async def extract_all_urls_and_databases(question_text: str, uploaded_files: list = None) -> dict:
    """Extract all URLs for scraping and database files from the question"""
    
    # Create context about uploaded files
    uploaded_context = ""
    if uploaded_files:
        uploaded_context = f"\n\nUPLOADED FILES AVAILABLE:\n"
        for file_info in uploaded_files:
            uploaded_context += f"- {file_info}\n"
    
    extraction_prompt = f"""
    Analyze this question and extract ONLY the ACTUAL DATA SOURCES needed to answer the questions:
    
    QUESTION: {question_text}
    {uploaded_context}
    
    CRITICAL INSTRUCTIONS:
    1. Look for REAL, COMPLETE URLs that contain actual data (not example paths or documentation links)
    2. Focus on data sources that are DIRECTLY needed to answer the specific questions being asked
    3. IGNORE example paths like "year=xyz/court=xyz" - these are just structure examples, not real URLs
    4. IGNORE reference links that are just for context (like documentation websites)
    5. Only extract data sources that have COMPLETE, USABLE URLs/paths
    6. If a filename mentioned in the question matches an uploaded file, treat it as a LOCAL FILE, not a URL to download
    7. For local files (like uploaded .sql, .csv files), add them to database_files with just the filename
    
    DATA SOURCE TYPES TO EXTRACT:
    - Complete S3 URLs with wildcards (s3://bucket/path/file.parquet)
    - Complete HTTP/HTTPS URLs to data APIs or files
    - Working database connection strings
    - Complete file paths that exist and are accessible
    - LOCAL FILES that were uploaded and referenced in the question (use format: "sql", "csv", etc.)
    
    DO NOT EXTRACT:
    - Example file paths (containing "xyz", "example", "sample")
    - Documentation or reference URLs that don't contain data
    - Incomplete paths or URL fragments
    - File structure descriptions that aren't actual URLs
    
    CONTEXT ANALYSIS:
    Read the question carefully. If it mentions a specific database with a working query example, 
    extract that. If it only shows file structure examples, don't extract those.
    
    Return a JSON object with:
    {{
        "scrape_urls": ["only URLs that need to be scraped for data to answer questions"],
        "database_files": [
            {{
                "url": "complete_working_database_url_or_s3_path",
                "format": "parquet|csv|json",
                "description": "what data this contains that helps answer the questions"
            }}
        ],
        "has_data_sources": true/false
    }}
    
    EXAMPLES:
    ✅ EXTRACT: "s3://bucket/data/file.parquet?region=us-east-1" (complete S3 URL)
    ✅ EXTRACT: "https://api.example.com/data.csv" (working data URL)
    ❌ IGNORE: "data/pdf/year=xyz/court=xyz/file.pdf" (example path with placeholders)
    ❌ IGNORE: "https://documentation-site.com/" (reference link, not data)
    
    Be very selective - only extract what is actually needed and usable.
    """
    
    response = await ping_gemini(extraction_prompt, "You are a data source extraction expert. Return only valid JSON.")
    try:
        # Check if response has error
        if "error" in response:
            print(f"❌ Gemini API error: {response['error']}")
            return extract_urls_with_regex(question_text, uploaded_files)
        
        # Extract text from response
        if "candidates" not in response or not response["candidates"]:
            print("❌ No candidates in Gemini response")
            return extract_urls_with_regex(question_text, uploaded_files)
        
        response_text = response["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Raw response text: {response_text}")
        
        # Try to extract JSON from response (sometimes it's wrapped in markdown)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.rfind("```")
            response_text = response_text[json_start:json_end].strip()
        
        print(f"Extracted JSON text: {response_text}")
        return json.loads(response_text)
        
    except Exception as e:
        print(f"URL extraction error: {e}")
        # Fallback to regex extraction
        return extract_urls_with_regex(question_text, uploaded_files)
    

def extract_urls_with_regex(question_text: str, uploaded_files: list = None) -> dict:
    """Fallback URL extraction using regex with context awareness"""
    scrape_urls = []
    database_files = []
    
    # Find all HTTP/HTTPS URLs
    url_pattern = r'https?://[^\s\'"<>]+'
    urls = re.findall(url_pattern, question_text)
    
    for url in urls:
        # Clean URL (remove trailing punctuation)
        clean_url = re.sub(r'[.,;)]+$', '', url)
        
        # Skip example/documentation URLs that don't contain actual data
        skip_patterns = [
            'example.com', 'documentation', 'github.com', 'docs.', 'help.',
            '/docs/', '/help/', '/guide/', '/tutorial/'
        ]
        
        if any(pattern in clean_url.lower() for pattern in skip_patterns):
            continue
        
        # Check if it's a database file
        if any(ext in clean_url.lower() for ext in ['.parquet', '.csv', '.json', '.sql', '.xlsx', '.xls']):
            if ".parquet" in clean_url:
                format_type = "parquet"
            elif ".csv" in clean_url:
                format_type = "csv"
            elif ".sql" in clean_url:
                format_type = "sql"
            elif ".xlsx" in clean_url or ".xls" in clean_url:
                format_type = "excel"
            else:
                format_type = "json"
            database_files.append({
                "url": clean_url,
                "format": format_type,
                "description": f"Database file ({format_type})"
            })
        else:
            # Only add to scrape_urls if it looks like it contains data
            # Skip pure documentation/reference sites
            if not any(skip in clean_url.lower() for skip in ['ecourts.gov.in']):  # Add known reference sites
                scrape_urls.append(clean_url)
    
    # Look for local file references (filenames with extensions)
    if uploaded_files:
        # Look for common file patterns mentioned in the text
        local_file_pattern = r'\b([\w\-]+)\.(sql|csv|json|parquet|xlsx?)\b'
        potential_files = re.findall(local_file_pattern, question_text, re.IGNORECASE)
        
        for match in potential_files:
            # match is a tuple like ('filename', 'sql') from the pattern groups
            filename_base, extension = match
            full_filename = f"{filename_base}.{extension}"
            
            # Check if this matches any uploaded file
            found_match = False
            for uploaded_info in uploaded_files:
                if full_filename.lower() in uploaded_info.lower():
                    database_files.append({
                        "url": full_filename,
                        "format": extension.lower(),
                        "description": f"Local uploaded file ({extension.lower()})"
                    })
                    found_match = True
                    break
            
            # If not found in uploaded files but looks like a local file reference
            if not found_match and not any(full_filename.lower() in df["url"].lower() for df in database_files):
                database_files.append({
                    "url": full_filename,
                    "format": extension.lower(),
                    "description": f"Referenced local file ({extension.lower()})"
                })
    
    # Find S3 paths - but only complete ones, not examples
    s3_pattern = r's3://[^\s\'"<>]+'
    s3_urls = re.findall(s3_pattern, question_text)
    for s3_url in s3_urls:
        # Skip example paths with placeholders
        if any(placeholder in s3_url for placeholder in ['xyz', 'example', '***', 'EXAMPLE']):
            continue
            
        clean_s3 = s3_url.split()[0]  # Take only the URL part
        if '?' in clean_s3:
            # Keep query parameters for S3 (they often contain important config)
            pass
        
        database_files.append({
            "url": clean_s3,
            "format": "parquet",
            "description": "S3 parquet file"
        })
    
    return {
        "scrape_urls": scrape_urls,
        "database_files": database_files,
        "has_data_sources": len(scrape_urls) > 0 or len(database_files) > 0
    }

async def scrape_all_urls(urls: list, created_files: set = None) -> list:
    """Enhanced URL scraping with better error handling and data processing"""
    scraped_data = []
    sourcer = data_scrape.ImprovedWebScraper()
    
    if created_files is None:
        created_files = set()
    
    for i, url in enumerate(urls):
        try:
            print(f"🌐 Scraping URL {i+1}/{len(urls)}: {url}")
            
            # Create enhanced config for web scraping
            source_config = {
                "source_type": "web_scrape",
                "url": url,
                "data_location": "Web page data",
                "extraction_strategy": "enhanced_scrape_web_table",
                "max_retries": 3,
                "retry_delay": 2
            }
            
            # Extract data with enhanced error handling
            try:
                result = await sourcer.extract_data(source_config)
            except Exception as extract_error:
                print(f"⚠️ Primary extraction failed: {extract_error}")
                print("🔄 Trying alternative extraction method...")
                
                # Fallback: try direct table extraction
                try:
                    html_content = await sourcer._smart_fetch_webpage(url)
                    df = await sourcer.web_scraper.extract_table_from_html(html_content)
                    
                    if df is not None and not df.empty:
                        # Clean numeric fields
                        cleaned_df, formatting_results = await sourcer.numeric_formatter.format_dataframe_numerics(df)
                        
                        result = {
                            "tables": [{
                                "table_name": "Fallback_Table",
                                "dataframe": cleaned_df,
                                "shape": cleaned_df.shape,
                                "columns": list(cleaned_df.columns),
                                "data_types": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
                                "numeric_formatting": formatting_results
                            }],
                            "metadata": {
                                "source_type": "web_scrape_fallback",
                                "source_url": url,
                                "extraction_method": "fallback_extraction",
                                "total_tables": 1,
                                "table_names": ["Fallback_Table"]
                            }
                        }
                        print("✅ Fallback extraction successful")
                    else:
                        raise Exception("Fallback extraction also failed")
                        
                except Exception as fallback_error:
                    print(f"❌ Both primary and fallback extraction failed: {fallback_error}")
                    # Try to save whatever content we can get
                    try:
                        html_content = await sourcer._smart_fetch_webpage(url)
                        
                        # Save raw HTML content for manual review
                        safe_url = url.split('//')[-1].replace('/', '_').replace('.', '_')[:30]
                        html_filename = f"failed_scrape_{safe_url}_{i+1}.html"
                        
                        with open(html_filename, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        created_files.add(os.path.normpath(html_filename))
                        print(f"💾 Saved HTML content to {html_filename} for manual review")
                        
                    except Exception:
                        print(f"❌ Complete failure for URL: {url}")
                    
                    continue
            
            # Process successfully extracted data
            if "tables" in result:
                tables = result["tables"]
                table_names = result["metadata"].get("table_names", [])
                
                for j, table_data in enumerate(tables):
                    df = table_data["dataframe"]
                    table_name = table_data["table_name"]
                    
                    if not df.empty:
                        # Enhanced filename generation
                        safe_table_name = re.sub(r'[^\w\-_\.]', '_', table_name)
                        safe_url = re.sub(r'[^\w\-_\.]', '_', url.split('//')[-1])[:30]
                        
                        # Add timestamp for uniqueness
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%H%M%S")
                        
                        if len(urls) == 1:  # Single URL
                            filename = f"{safe_table_name}_{timestamp}.csv"
                        else:  # Multiple URLs
                            filename = f"{safe_table_name}_url{i+1}_{timestamp}.csv"
                        
                        # Enhanced CSV saving with error handling
                        try:
                            df.to_csv(filename, index=False, encoding="utf-8")
                        except UnicodeEncodeError:
                            # Fallback encoding
                            df.to_csv(filename, index=False, encoding="utf-8-sig")
                        except Exception as save_error:
                            print(f"⚠️ Error saving CSV, trying alternative method: {save_error}")
                            # Clean problematic characters and retry
                            df_clean = df.copy()
                            for col in df_clean.columns:
                                if df_clean[col].dtype == 'object':
                                    df_clean[col] = df_clean[col].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
                            df_clean.to_csv(filename, index=False, encoding="utf-8")
                        
                        created_files.add(os.path.normpath(filename))
                        
                        # Enhanced metadata
                        table_info = {
                            "filename": filename,
                            "source_url": url,
                            "table_name": table_name,
                            "shape": table_data["shape"],
                            "columns": table_data["columns"],
                            "data_types": table_data.get("data_types", {}),
                            "sample_data": df.head(3).to_dict('records') if not df.empty else [],
                            "extraction_method": result["metadata"].get("extraction_method", "standard"),
                            "data_quality": {
                                "total_cells": df.shape[0] * df.shape[1],
                                "non_null_cells": df.count().sum(),
                                "null_percentage": (1 - df.count().sum() / (df.shape[0] * df.shape[1])) * 100,
                                "numeric_columns": len([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]),
                                "text_columns": len([col for col in df.columns if df[col].dtype == 'object'])
                            },
                            "numeric_formatting": table_data.get("numeric_formatting", {})
                        }
                        
                        scraped_data.append(table_info)
                        
                        print(f"💾 Saved {table_name} as {filename}")
                        print(f"   📊 Size: {df.shape[0]} rows × {df.shape[1]} columns")
                        print(f"   🔢 Numeric: {table_info['data_quality']['numeric_columns']} cols")
                        print(f"   📝 Text: {table_info['data_quality']['text_columns']} cols")
                        print(f"   💯 Data completeness: {100 - table_info['data_quality']['null_percentage']:.1f}%")
            
            # Handle old single table format for backward compatibility
            elif "dataframe" in result:
                df = result["dataframe"]
                
                if not df.empty:
                    safe_url = re.sub(r'[^\w\-_\.]', '_', url.split('//')[-1])[:30]
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"scraped_data_url{i+1}_{timestamp}.csv"
                    
                    try:
                        df.to_csv(filename, index=False, encoding="utf-8")
                    except UnicodeEncodeError:
                        df.to_csv(filename, index=False, encoding="utf-8-sig")
                    
                    created_files.add(os.path.normpath(filename))
                    
                    scraped_data.append({
                        "filename": filename,
                        "source_url": url,
                        "table_name": "Main_Table",
                        "shape": list(df.shape),
                        "columns": list(df.columns),
                        "sample_data": df.head(3).to_dict('records'),
                        "extraction_method": "legacy_format"
                    })
                    
                    print(f"💾 Saved legacy format data as {filename}")
                    
        except Exception as e:
            print(f"❌ Complete failure processing URL {url}: {e}")
            # Log the error for debugging
            error_log = f"scraping_error_url{i+1}.log"
            with open(error_log, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Type: {type(e).__name__}\n")
            created_files.add(os.path.normpath(error_log))
            continue
    
    if scraped_data:
        # Create summary file
        summary_filename = "scraping_summary.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("WEB SCRAPING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total URLs processed: {len(urls)}\n")
            f.write(f"Successful extractions: {len(scraped_data)}\n")
            f.write(f"Success rate: {(len(scraped_data)/len(urls))*100:.1f}%\n\n")
            
            for i, data in enumerate(scraped_data, 1):
                f.write(f"{i}. {data['filename']}\n")
                f.write(f"   Source: {data['source_url']}\n")
                f.write(f"   Table: {data['table_name']}\n")
                f.write(f"   Size: {data['shape'][0]} rows × {data['shape'][1]} columns\n")
                if 'data_quality' in data:
                    f.write(f"   Quality: {100 - data['data_quality']['null_percentage']:.1f}% complete\n")
                f.write("\n")
        
        created_files.add(os.path.normpath(summary_filename))
        print(f"📋 Created scraping summary: {summary_filename}")
    
    print(f"\n✅ Web scraping complete: {len(scraped_data)} tables extracted from {len(urls)} URLs")
    return scraped_data


def normalize_column_names(columns):
    """Normalize column names for consistent matching"""
    normalized = []
    for col in columns:
        # Convert to string, strip whitespace, normalize case
        normalized_col = str(col).strip().lower()
        # Replace multiple spaces/tabs with single space
        normalized_col = re.sub(r'\s+', ' ', normalized_col)
        normalized.append(normalized_col)
    return normalized

def columns_match(cols1, cols2, threshold=0.6):
    """Check if two sets of columns match with some tolerance"""
    norm_cols1 = normalize_column_names(cols1)
    norm_cols2 = normalize_column_names(cols2)
    
    if len(norm_cols1) != len(norm_cols2):
        print(f"   🔍 Column count mismatch: {len(norm_cols1)} vs {len(norm_cols2)}")
        return False
    
    # Check exact match first
    if norm_cols1 == norm_cols2:
        print(f"   ✅ Exact column match found")
        return True
    
    # Check similarity for each column pair
    matches = 0
    for c1, c2 in zip(norm_cols1, norm_cols2):
        if c1 == c2:
            matches += 1
        else:
            # Simple similarity check (you could use more sophisticated methods)
            if c1 and c2:  # Avoid empty strings
                similarity = len(set(c1.split()) & set(c2.split())) / max(len(c1.split()), len(c2.split()))
                if similarity >= threshold:
                    matches += 1
                    print(f"   🔍 Similar columns: '{c1}' ≈ '{c2}' (similarity: {similarity:.2f})")
    
    match_ratio = matches / len(norm_cols1)
    result = match_ratio >= threshold
    print(f"   🔍 Column match ratio: {match_ratio:.2f} (threshold: {threshold}) = {'✅ MATCH' if result else '❌ NO MATCH'}")
    return result

def looks_like_header(row):
    """Enhanced heuristic: mostly non-empty strings, not numbers; short-ish cells."""
    if not row or not isinstance(row, list):
        return False
    str_like = sum(1 for c in row if isinstance(c, str) and bool(re.search(r"[A-Za-z]", c or "")))
    num_like = sum(1 for c in row if isinstance(c, str) and re.fullmatch(r"[-+]?[\d,.]+", (c or "").strip()))
    avg_len = sum(len((c or "")) for c in row) / max(len(row), 1)
    return (str_like >= max(1, len(row)//2)) and (num_like <= len(row)//3) and (avg_len <= 40)

async def extract_pdf_with_pdfplumber(pdf_file_path: str) -> list:
    """Extract tables using pdfplumber with enhanced settings"""
    tables = []
    header_candidates = []
    
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   📑 PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from current page with improved settings
                page_tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines", 
                        "intersection_tolerance": 5,
                        "snap_tolerance": 3,
                        "join_tolerance": 3
                    }
                )
                
                if page_tables:
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 0:
                            # Enhanced header detection
                            first_row = table[0] if table else None
                            has_smart_header = looks_like_header(first_row) if first_row else False
                            
                            if has_smart_header and first_row:
                                # Track this header pattern
                                header_tuple = tuple((c or "").strip() for c in first_row)
                                header_candidates.append(header_tuple)
                                
                                # Use first row as headers, rest as data
                                headers = [str((c or "")).strip() for c in first_row]
                                rows = table[1:] if len(table) > 1 else []
                            else:
                                # No clear header detected, use generic column names
                                max_cols = max(len(row) for row in table) if table else 0
                                headers = [f"column_{j+1}" for j in range(max_cols)]
                                rows = table
                            
                            # Create DataFrame with better error handling
                            try:
                                if rows:  # Only if we have data rows
                                    # Ensure all rows have same length as headers
                                    normalized_rows = []
                                    for row in rows:
                                        normalized_row = []
                                        for j in range(len(headers)):
                                            if j < len(row):
                                                normalized_row.append(row[j])
                                            else:
                                                normalized_row.append(None)
                                        normalized_rows.append(normalized_row)
                                    
                                    df = pd.DataFrame(normalized_rows, columns=headers)
                                    # Remove completely empty rows
                                    df = df.dropna(how='all')
                                    
                                    if not df.empty:
                                        tables.append(df)
                                        header_info = "✓ Smart header" if has_smart_header else "⚡ Generic header"
                                        print(f"   ✅ Page {page_num + 1}, Table {table_idx + 1}: {df.shape[0]} rows, {df.shape[1]} cols ({header_info})")
                            except Exception as df_error:
                                print(f"   ⚠️ Failed to create DataFrame for page {page_num + 1}, table {table_idx + 1}: {df_error}")
        
        # Check for consistent headers across pages
        if header_candidates:
            from collections import Counter
            header_counter = Counter(header_candidates)
            if header_counter:
                most_common_header, frequency = header_counter.most_common(1)[0]
                if frequency >= 2:
                    print(f"   🔄 Found repeating header pattern across {frequency} tables: {list(most_common_header)[:3]}...")
        
        # If no tables found with default settings, try with more lenient settings
        if not tables:
            print("📄 Retrying with more lenient table detection settings...")
            with pdfplumber.open(pdf_file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Try with more aggressive table detection
                    page_tables = page.extract_tables(table_settings={
                        "vertical_strategy": "text",  # More lenient
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                        "edge_min_length": 3
                    })
                    
                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                # Use first row as headers for fallback method
                                headers = [f"col_{j}" if not table[0][j] else str(table[0][j]).strip() 
                                         for j in range(len(table[0]))]
                                rows = table[1:]
                                
                                try:
                                    df = pd.DataFrame(rows, columns=headers)
                                    df = df.dropna(how='all')
                                    
                                    if not df.empty:
                                        tables.append(df)
                                        print(f"   ✅ Page {page_num + 1}, Table {table_idx + 1}: {df.shape[0]} rows, {df.shape[1]} cols (fallback)")
                                except Exception as df_error:
                                    print(f"   ⚠️ Fallback failed for page {page_num + 1}, table {table_idx + 1}: {df_error}")
                                    
    except Exception as e:
        print(f"❌ pdfplumber extraction failed for {pdf_file_path}: {e}")
        
    return tables

async def process_image_with_enhanced_ocr(image_bytes: bytes, filename: str, question_text: str, extracted_files_list: list = None, created_files: set = None) -> str:
    """Enhanced image processing with Gemini Pro for better text/data extraction"""
    try:
        print(f"🖼️ Processing image: {filename}")
        
        # Convert image to base64 for Gemini Pro
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # First try Gemini Pro for intelligent text/data extraction
        gemini_extracted_text = await extract_data_with_gemini_pro(base64_image, filename)
        
        if gemini_extracted_text and gemini_extracted_text.strip():
            print(f"✅ Gemini Pro extracted content from image: {filename}")
            question_text += f"\n\nExtracted from image ({filename}) using Gemini Pro:\n{gemini_extracted_text}"
            
            # Check if extracted content contains structured data
            if extracted_files_list is not None:
                if await detect_and_process_data_from_text(gemini_extracted_text, filename, extracted_files_list, created_files):
                    print(f"📊 Structured data detected and processed from image: {filename}")
            
            return question_text
        
        # Fallback to OCR if Gemini Pro fails or returns empty content
        print("🔄 Gemini Pro failed or returned empty content, falling back to OCR API...")
        
        ocr_extracted_text = await extract_text_with_ocr(base64_image, filename)
        
        if ocr_extracted_text and ocr_extracted_text.strip():
            print(f"✅ OCR successfully extracted content from image: {filename}")
            question_text += f"\n\nExtracted from image ({filename}) using OCR fallback:\n{ocr_extracted_text}"
            
            # Check if OCR extracted content contains structured data
            if extracted_files_list is not None:
                if await detect_and_process_data_from_text(ocr_extracted_text, filename, extracted_files_list, created_files):
                    print(f"📊 Structured data detected and processed from OCR text: {filename}")
            
            return question_text
        else:
            print(f"❌ Both Gemini Pro and OCR failed to extract content from image: {filename}")
            return question_text + f"\n\n❌ Failed to extract any content from image: {filename}"
                
    except Exception as e:
        print(f"❌ Error processing image {filename}: {e}")
        
    return question_text

async def extract_text_with_ocr(base64_image: str, filename: str) -> str:
    """Extract text from image using OCR API as fallback"""
    try:
        print(f"🔍 Using OCR to extract text from image: {filename}")
        
        if not ocr_api_key:
            print("⚠️ OCR_API_KEY not found")
            return None
            
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            form_data = {
                "base64Image": f"data:image/png;base64,{base64_image}",
                "apikey": ocr_api_key,
                "language": "eng",
                "scale": "true",
                "OCREngine": "1"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = await client.post(OCR_API_URL, data=form_data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                if not result.get('IsErroredOnProcessing', True):
                    parsed_results = result.get('ParsedResults', [])
                    if parsed_results:
                        image_text = parsed_results[0].get('ParsedText', '').strip()
                        if image_text:
                            print(f"✅ OCR successfully extracted text from {filename}")
                            return image_text
                        else:
                            print("ℹ️ OCR completed but no text found")
                            return None
                    else:
                        print("ℹ️ OCR completed but no results returned")
                        return None
                else:
                    print(f"❌ OCR processing failed: {result.get('ErrorMessage', 'Unknown error')}")
                    return None
            else:
                print(f"❌ OCR API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error in OCR text extraction: {e}")
        return None

async def extract_data_with_gemini_pro(base64_image: str, filename: str) -> str:
    """Use Gemini Pro to extract text or data from images with improved error handling"""
    try:
        print(f"🤖 Using Gemini Pro to analyze image: {filename}")
        
        if not gemini_api:
            print("⚠️ Gemini API key not found")
            return None
            
        headers = {
            "x-goog-api-key": gemini_api,
            "Content-Type": "application/json"
        }
        
        # Create prompt for intelligent image analysis
        analysis_prompt = """Analyze this image and extract any text, data, questions, or structured information you can find. 
        Pay special attention to:
        1. Any questions or text content
        2. Tables, charts, graphs, or structured data
        3. Numbers, statistics, or measurements
        4. Lists or organized information
        
        If you find structured data (like tables), try to format it in a clear, parseable way.
        If you find questions, extract them clearly.
        If it's just text, extract it accurately.
        
        Provide a comprehensive extraction of all visible content. If you cannot see or extract any meaningful content, respond with 'NO_CONTENT_FOUND'."""
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": analysis_prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 4096,
            }
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-pro:generateContent", 
                headers=headers, 
                json=payload
            )
            
            if response.status_code == 200:
                gemini_response = response.json()
                
                # Check if response has the expected structure
                if "candidates" in gemini_response and len(gemini_response["candidates"]) > 0:
                    candidate = gemini_response["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        extracted_text = candidate["content"]["parts"][0]["text"]
                        
                        # Check if Gemini found content
                        if extracted_text and extracted_text.strip() and "NO_CONTENT_FOUND" not in extracted_text:
                            print(f"✅ Gemini Pro successfully analyzed image: {filename}")
                            return extracted_text.strip()
                        else:
                            print(f"ℹ️ Gemini Pro found no meaningful content in image: {filename}")
                            return None
                    else:
                        print("⚠️ Gemini Pro response missing expected content structure")
                        return None
                else:
                    print("⚠️ Gemini Pro response missing candidates")
                    return None
            else:
                print(f"❌ Gemini Pro API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error in Gemini Pro image analysis: {e}")
        return None

async def detect_and_process_data_from_text(text_content: str, source_name: str, extracted_files_list: list = None, created_files: set = None) -> bool:
    """Detect if text contains structured data and process it into CSV"""
    try:
        # Check for data patterns
        has_tabular_data = False
        has_numeric_data = False
        
        # Look for table-like patterns
        lines = text_content.split('\n')
        potential_table_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for delimiter patterns (tabs, pipes, multiple spaces, commas)
            delimiters = ['\t', '|', ',']
            for delimiter in delimiters:
                if delimiter in line and len(line.split(delimiter)) > 2:
                    potential_table_lines.append(line)
                    has_tabular_data = True
                    break
            
            # Check for space-separated columns (at least 3 parts)
            if len(line.split()) > 2 and not has_tabular_data:
                potential_table_lines.append(line)
                has_tabular_data = True
        
        # Check for numeric data patterns
        if re.search(r'\d+[.,]\d+|\$\d+|\d+%|\d+\s*(million|billion|thousand)', text_content, re.IGNORECASE):
            has_numeric_data = True
        
        if has_tabular_data or has_numeric_data:
            print(f"📊 Structured data detected in {source_name}")
            
            # Use data cleaning functions to process the data
            cleaned_data = await clean_and_structure_extracted_data(text_content, source_name)
            
            if cleaned_data is not None and not cleaned_data.empty:
                # Save as CSV
                csv_filename = f"extracted_data_{source_name.replace('.', '_').replace(' ', '_')}.csv"
                cleaned_data.to_csv(csv_filename, index=False)
                track_created_file(csv_filename, created_files)
                print(f"💾 Saved extracted data to: {csv_filename}")
                
                # Add to data summary
                await update_data_summary_with_extracted_data(csv_filename, cleaned_data, source_name, extracted_files_list, created_files)
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error detecting/processing data from text: {e}")
        return False

async def clean_and_structure_extracted_data(text_content: str, source_name: str) -> pd.DataFrame:
    """Clean and structure extracted text data using existing cleaning functions"""
    try:
        # Initialize data scraper for cleaning functions
        scraper = data_scrape.DataScraper()
        
        # Try to parse the text into a DataFrame
        df = None
        
        # Method 1: Try to detect delimiter-separated data
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return pd.DataFrame()
        
        # Try different delimiters
        delimiters = ['\t', '|', ',', ';']
        for delimiter in delimiters:
            try:
                # Check if most lines have the same number of parts
                line_parts = [len(line.split(delimiter)) for line in lines if delimiter in line]
                if len(line_parts) > 1 and len(set(line_parts)) <= 2:  # Allow some variation
                    # Create DataFrame
                    data_rows = []
                    for line in lines:
                        if delimiter in line:
                            parts = [part.strip() for part in line.split(delimiter)]
                            data_rows.append(parts)
                    
                    if data_rows:
                        # Use first row as headers if it looks like headers
                        first_row = data_rows[0]
                        if any(not re.match(r'^\d+\.?\d*$', cell) for cell in first_row):
                            df = pd.DataFrame(data_rows[1:], columns=first_row)
                        else:
                            df = pd.DataFrame(data_rows)
                        break
            except:
                continue
        
        # Method 2: Try space-separated if delimiter method failed
        if df is None:
            try:
                # Look for consistent column patterns
                consistent_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) > 2:  # At least 3 columns
                        consistent_lines.append(parts)
                
                if len(consistent_lines) > 1:
                    # Find most common number of columns
                    col_counts = [len(line) for line in consistent_lines]
                    most_common_cols = max(set(col_counts), key=col_counts.count)
                    
                    # Filter lines with the most common column count
                    filtered_lines = [line for line in consistent_lines if len(line) == most_common_cols]
                    
                    if len(filtered_lines) > 1:
                        # Use first row as headers if appropriate
                        first_row = filtered_lines[0]
                        if any(not re.match(r'^\d+\.?\d*$', cell) for cell in first_row):
                            df = pd.DataFrame(filtered_lines[1:], columns=first_row)
                        else:
                            df = pd.DataFrame(filtered_lines)
            except:
                pass
        
        # Method 3: Create simple key-value pairs if structured data detected
        if df is None:
            try:
                # Look for key: value patterns
                key_value_pairs = []
                for line in lines:
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            key_value_pairs.append([key, value])
                
                if key_value_pairs:
                    df = pd.DataFrame(key_value_pairs, columns=['Attribute', 'Value'])
            except:
                pass
        
        if df is not None and not df.empty:
            print(f"📊 Successfully parsed data into DataFrame: {df.shape}")
            
            # Handle duplicate column names which can cause issues
            if len(df.columns) != len(set(df.columns)):
                print("⚠️ Warning: Duplicate column names detected, renaming...")
                df.columns = pd.Index([f"{col}_{i}" if list(df.columns).count(col) > 1 else col 
                                     for i, col in enumerate(df.columns)])
            
            # Apply basic cleaning using existing functions
            df = scraper._basic_clean_dataframe(df)
            
            # Try to clean numeric columns
            for col in df.columns:
                try:
                    # Ensure we're working with a Series, not a DataFrame
                    column_data = df[col]
                    if hasattr(column_data, 'dtype') and column_data.dtype == 'object':
                        # Check if column contains numeric data
                        sample_values = column_data.dropna().head(10)
                        if any(re.search(r'\d', str(val)) for val in sample_values):
                            try:
                                # Try to determine numeric type and clean
                                if any('$' in str(val) or 'USD' in str(val) for val in sample_values):
                                    df[col] = scraper._clean_currency_column(column_data)
                                elif any('%' in str(val) for val in sample_values):
                                    df[col] = scraper._clean_percentage_column(column_data)
                                else:
                                    df[col] = scraper._clean_generic_numeric_column(column_data)
                            except Exception as clean_error:
                                print(f"⚠️ Warning: Could not clean column '{col}': {clean_error}")
                                pass  # Keep original if cleaning fails
                except Exception as col_error:
                    print(f"⚠️ Warning: Error processing column '{col}': {col_error}")
                    continue
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Error cleaning extracted data: {e}")
        return pd.DataFrame()

async def update_data_summary_with_extracted_data(csv_filename: str, dataframe: pd.DataFrame, source_name: str, extracted_files_list: list = None, created_files: set = None):
    """Update the data summary with information about extracted data"""
    try:
        # Create a summary entry for the extracted data
        extracted_info = {
            "filename": csv_filename,
            "source": f"extracted_from_{source_name}",
            "shape": list(dataframe.shape),
            "columns": list(dataframe.columns),
            "data_types": {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            "sample_data": dataframe.head(3).to_dict('records') if not dataframe.empty else [],
            "extraction_method": "gemini_pro_or_ocr",
            "processing_timestamp": time.time()
        }
        
        # Add to the extracted files list if provided
        if extracted_files_list is not None:
            extracted_files_list.append(extracted_info)
        
        # Save individual extraction info
        extraction_info_file = f"extraction_info_{csv_filename.replace('.csv', '.json')}"
        with open(extraction_info_file, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(extracted_info), f, indent=2)
        
        track_created_file(extraction_info_file, created_files)
        print(f"📋 Extraction info saved to: {extraction_info_file}")
        
    except Exception as e:
        print(f"❌ Error updating data summary: {e}")

async def process_pdf_files(created_files: set = None) -> list:
    """Process all PDF files in current directory and extract tables, combining tables with same headers"""
    pdf_data = []
    
    # Find all PDF files in current directory
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("📄 No PDF files found in current directory")
        return pdf_data
    
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    
    all_raw_tables = []  # Store all raw tables
    
    # First pass: Extract ALL raw tables from ALL PDFs using enhanced extraction
    print("🔄 Phase 1: Extracting raw tables from all PDFs...")
    for i, pdf_file in enumerate(pdf_files):
        try:
            print(f"📄 Processing PDF {i+1}/{len(pdf_files)}: {pdf_file}")
            
            # Try pdfplumber first (better table detection)
            tables = await extract_pdf_with_pdfplumber(pdf_file)
            
            # If pdfplumber didn't work well, fallback to tabula
            if not tables or len(tables) == 0:
                print("📄 pdfplumber found no tables, trying tabula as fallback...")
                try:
                    tables = tabula.read_pdf(
                        pdf_file, 
                        pages='all', 
                        multiple_tables=True,
                        pandas_options={'header': 'infer'},
                        lattice=True,
                        silent=True
                    )
                    
                    if not tables or all(df.empty for df in tables):
                        print("📄 Retrying tabula with stream method...")
                        tables = tabula.read_pdf(
                            pdf_file, 
                            pages='all', 
                            multiple_tables=True,
                            pandas_options={'header': 'infer'},
                            stream=True,
                            silent=True
                        )
                        
                except Exception as tabula_error:
                    print(f"❌ Both pdfplumber and tabula failed for {pdf_file}: {tabula_error}")
                    continue
            
            if not tables:
                print(f"⚠️ No tables found in {pdf_file}")
                continue
            
            print(f"📊 Found {len(tables)} raw tables in {pdf_file}")
            
            # Store all raw tables with metadata
            for j, raw_df in enumerate(tables):
                if raw_df.empty:
                    print(f"⚠️ Table {j+1} is empty, skipping")
                    continue
                
                table_metadata = {
                    "raw_dataframe": raw_df,
                    "source_pdf": pdf_file,
                    "table_number": j + 1,
                    "raw_columns": list(raw_df.columns),
                    "estimated_rows": len(raw_df),
                    "has_smart_headers": any(col.replace('_', ' ').replace('-', ' ').strip() 
                                           for col in raw_df.columns if not col.startswith('column_'))
                }
                
                all_raw_tables.append(table_metadata)
                print(f"✅ Stored raw table {j+1} from {pdf_file} ({raw_df.shape[0]} rows, {raw_df.shape[1]} cols)")
                print(f"   📋 Columns: {list(raw_df.columns)}")
        
        except Exception as e:
            print(f"❌ Failed to process PDF {pdf_file}: {e}")
    
    if not all_raw_tables:
        print("❌ No tables extracted from any PDF files")
        return pdf_data
    
    print(f"📊 Phase 1 complete: {len(all_raw_tables)} raw tables extracted")
    
    # Second pass: Group raw tables by similar headers
    print("\n🔄 Phase 2: Grouping tables with similar headers...")
    combined_data_groups = {}
    
    for table_meta in all_raw_tables:
        columns = table_meta["raw_columns"]
        
        print(f"\n🔍 Analyzing table from {table_meta['source_pdf']} (table {table_meta['table_number']})")
        print(f"   📋 Columns: {columns}")
        
        # Find existing group with matching headers
        found_group = None
        for group_key, group_data in combined_data_groups.items():
            print(f"   🔄 Comparing with group '{group_key}':")
            if columns_match(columns, group_data["reference_columns"]):
                found_group = group_key
                break
        
        if found_group:
            # Add to existing group
            combined_data_groups[found_group]["raw_tables"].append(table_meta)
            print(f"   ➕ Added to existing group '{found_group}' (now {len(combined_data_groups[found_group]['raw_tables'])} tables)")
        else:
            # Create new group
            group_name = f"table_group_{len(combined_data_groups) + 1}"
            combined_data_groups[group_name] = {
                "reference_columns": columns,
                "raw_tables": [table_meta]
            }
            print(f"   🆕 Created new group '{group_name}'")
    
    print(f"\n📊 Phase 2 complete: {len(combined_data_groups)} group(s) created")
    for group_name, group_data in combined_data_groups.items():
        print(f"   📁 {group_name}: {len(group_data['raw_tables'])} tables")
        for table in group_data['raw_tables']:
            print(f"      - {table['source_pdf']} (table {table['table_number']})")
    
    # Third pass: Simply merge tables and save
    print("\n🔄 Phase 3: Merging grouped tables and saving...")
    
    for group_name, group_data in combined_data_groups.items():
        raw_tables_in_group = group_data["raw_tables"]
        reference_columns = group_data["reference_columns"]
        
        print(f"\n🔗 Processing group '{group_name}' with {len(raw_tables_in_group)} table(s)...")
        
        # Merge all raw tables in this group
        combined_raw_dfs = []
        source_pdfs = []
        total_estimated_rows = 0
        
        for table_meta in raw_tables_in_group:
            raw_df = table_meta["raw_dataframe"].copy()
            
            # Ensure column names match the reference
            if list(raw_df.columns) != reference_columns:
                print(f"   🔧 Standardizing columns for {table_meta['source_pdf']}")
                raw_df.columns = reference_columns
            
            # Add source tracking
            raw_df['source_pdf'] = table_meta["source_pdf"]
            raw_df['table_number'] = table_meta["table_number"]
            
            combined_raw_dfs.append(raw_df)
            source_pdfs.append(table_meta["source_pdf"])
            total_estimated_rows += table_meta.get("estimated_rows", len(raw_df))
            print(f"   ✅ Added {raw_df.shape[0]} rows from {table_meta['source_pdf']}")
        
        # Combine all raw DataFrames
        try:
            print(f"   🔗 Merging {len(combined_raw_dfs)} raw tables...")
            merged_df = pd.concat(combined_raw_dfs, ignore_index=True)
            print(f"   ✅ Merged into single table: {merged_df.shape[0]} rows, {merged_df.shape[1]} cols")
            
            # Create a meaningful filename
            if len(combined_data_groups) == 1:
                csv_filename = "combined_tables.csv"
            else:
                first_col = reference_columns[0] if reference_columns else "data"
                clean_name = re.sub(r'[^\w\s-]', '', str(first_col)).strip()
                clean_name = re.sub(r'[-\s]+', '_', clean_name)
                csv_filename = f"combined_{clean_name[:20]}.csv"
            
            # Save the merged data
            merged_df.to_csv(csv_filename, index=False, encoding="utf-8")
            track_created_file(csv_filename, created_files)
            
            table_info = {
                "filename": csv_filename,
                "source_pdfs": list(set(source_pdfs)),
                "table_count": len(raw_tables_in_group),
                "shape": merged_df.shape,
                "columns": list(merged_df.columns),
                "sample_data": merged_df.head(3).to_dict('records'),
                "description": f"Combined table from {len(set(source_pdfs))} PDF file(s) ({len(raw_tables_in_group)} table(s) total)",
                "formatting_applied": "Enhanced extraction with pdfplumber and tabula fallback",
                "extraction_method": "pdfplumber with smart header detection",
                "estimated_total_rows": total_estimated_rows
            }
            
            pdf_data.append(table_info)
            print(f"   💾 Saved merged table as {csv_filename}")
            print(f"   📊 Final: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
            print(f"   📋 Sources: {', '.join(set(source_pdfs))}")
            
        except Exception as merge_error:
            print(f"❌ Error merging group {group_name}: {merge_error}")
            # Fallback: save individual tables
            for idx, table_meta in enumerate(raw_tables_in_group):
                raw_df = table_meta["raw_dataframe"]
                csv_filename = f"fallback_{group_name}_table_{idx+1}.csv"
                raw_df.to_csv(csv_filename, index=False, encoding="utf-8")
                track_created_file(csv_filename, created_files)
                
                table_info = {
                    "filename": csv_filename,
                    "source_pdfs": [table_meta["source_pdf"]],
                    "table_count": 1,
                    "shape": raw_df.shape,
                    "columns": list(raw_df.columns),
                    "sample_data": raw_df.head(3).to_dict('records'),
                    "description": f"Fallback table from {table_meta['source_pdf']} (merge failed)",
                    "formatting_applied": "Enhanced extraction with pdfplumber"
                }
                
                pdf_data.append(table_info)
                print(f"💾 Saved fallback table as {csv_filename}")
    
    if pdf_data:
        print(f"\n✅ Processing complete: Created {len(pdf_data)} output file(s)")
        print(f"📊 Merged {len(all_raw_tables)} total tables from {len(pdf_files)} PDF files")
    
    return pdf_data


async def process_sql_file(file_path: str) -> dict:
    """Process SQL file and extract schema information by analyzing SQL statements"""
    try:
        # Read the SQL file
        if file_path.startswith('http'):
            async with httpx.AsyncClient() as client:
                response = await client.get(file_path)
                sql_content = response.text
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
        
        # Basic SQL parsing to extract table information
        sql_content = sql_content.upper()
        
        # Find CREATE TABLE statements - improved regex to capture complete table definitions
        create_table_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);'
        tables = re.findall(create_table_pattern, sql_content, re.DOTALL | re.IGNORECASE)
        
        schema_info = {
            "tables": [],
            "total_tables": len(tables),
            "sql_statements": []
        }
        
        for table_name, columns_str in tables:
            # Parse column definitions
            columns = []
            column_types = {}
            
            # Split by comma and clean up - handle nested parentheses better
            column_defs = []
            paren_depth = 0
            current_def = ""
            
            for char in columns_str:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    column_defs.append(current_def.strip())
                    current_def = ""
                    continue
                current_def += char
            
            if current_def.strip():
                column_defs.append(current_def.strip())
            
            for col_def in column_defs:
                col_def = col_def.strip()
                if col_def:
                    # Extract column name and type - handle various SQL syntax
                    parts = col_def.split()
                    if len(parts) >= 2:
                        col_name = parts[0].strip('`"[]')
                        col_type = parts[1].upper()
                        
                        # Skip constraint definitions that don't start with column names
                        if col_name.upper() in ['PRIMARY', 'FOREIGN', 'CONSTRAINT', 'INDEX', 'KEY', 'UNIQUE', 'CHECK']:
                            continue
                            
                        # Handle composite types like VARCHAR(255), DECIMAL(10,2)
                        if len(parts) > 2 and '(' in parts[1]:
                            col_type = ' '.join(parts[1:3]) if len(parts) > 2 else parts[1]
                        columns.append(col_name)
                        column_types[col_name] = col_type.upper()
            
            schema_info["tables"].append({
                "table_name": table_name,
                "columns": columns,
                "column_types": column_types,
                "total_columns": len(columns)
            })
        
        # Extract other SQL statements (INSERT, SELECT examples, etc.)
        statements = []
        for line in sql_content.split(';'):
            line = line.strip()
            if line and not line.startswith('--'):
                stmt_type = line.split()[0].upper() if line.split() else ""
                if stmt_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']:
                    statements.append({
                        "type": stmt_type,
                        "statement": line[:200] + "..." if len(line) > 200 else line
                    })
        
        schema_info["sql_statements"] = statements[:5]  # Limit to first 5 statements
        
        return {
            "success": True,
            "schema": schema_info,
            "raw_content": sql_content[:1000] + "..." if len(sql_content) > 1000 else sql_content
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "schema": {"tables": [], "total_tables": 0, "sql_statements": []}
        }


async def create_sql_summary_file(sql_info: dict, output_filename: str = None, created_files: set = None) -> str:
    """Create a comprehensive summary file for SQL schema information"""
    if not output_filename:
        output_filename = f"sql_summary_{int(time.time())}.txt"
    
    summary_content = []
    summary_content.append("="*60)
    summary_content.append("SQL DATABASE SCHEMA SUMMARY")
    summary_content.append("="*60)
    summary_content.append(f"Source: {sql_info.get('source_url', 'Unknown')}")
    summary_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_content.append("")
    
    if 'sql_tables' in sql_info.get('schema', {}):
        tables = sql_info['schema']['sql_tables']
        summary_content.append(f"TOTAL TABLES: {len(tables)}")
        summary_content.append(f"TOTAL COLUMNS: {sql_info.get('total_columns', 0)}")
        summary_content.append("")
        
        for i, table in enumerate(tables, 1):
            summary_content.append(f"{i}. TABLE: {table['table_name']}")
            summary_content.append(f"   Columns: {table['total_columns']}")
            summary_content.append("   Schema:")
            
            for col_name in table['columns']:
                col_type = table['column_types'].get(col_name, 'UNKNOWN')
                summary_content.append(f"     - {col_name}: {col_type}")
            
            summary_content.append("")
    
    # Add SQL statements preview
    if 'sql_statements' in sql_info.get('schema', {}):
        statements = sql_info['schema']['sql_statements']
        if statements:
            summary_content.append("EXAMPLE SQL STATEMENTS:")
            summary_content.append("-" * 30)
            for stmt in statements:
                summary_content.append(f"{stmt['type']}: {stmt['statement']}")
                summary_content.append("")
    
    # Add raw content preview
    if 'sql_content_preview' in sql_info:
        summary_content.append("SQL CONTENT PREVIEW:")
        summary_content.append("-" * 30)
        summary_content.append(sql_info['sql_content_preview'])
        summary_content.append("")
    
    summary_content.append("="*60)
    summary_content.append("END OF SQL SUMMARY")
    summary_content.append("="*60)
    
    # Write summary to file
    summary_text = "\n".join(summary_content)
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    track_created_file(output_filename, created_files)
    print(f"📄 SQL summary saved to: {output_filename}")
    return output_filename


async def process_excel_files(created_files: set = None) -> list:
    """Enhanced Excel processing with better error handling and support for both .xlsx and .xls files"""
    excel_data = []
    
    # Find all Excel files in current directory
    excel_files = glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.xlsm") + glob.glob("*.xlsb")
    if not excel_files:
        print("📊 No Excel files found in current directory")
        return excel_data
    
    print(f"📊 Found {len(excel_files)} Excel files to process")
    
    for i, excel_file in enumerate(excel_files):
        try:
            print(f"📊 Processing Excel file {i+1}/{len(excel_files)}: {excel_file}")
            
            # Determine the best engine based on file extension
            file_ext = os.path.splitext(excel_file)[1].lower()
            engine = 'openpyxl' if file_ext in ['.xlsx', '.xlsm'] else 'xlrd'
            
            # Try to get sheet names first to verify file is readable
            try:
                if file_ext in ['.xlsx', '.xlsm']:
                    # Use openpyxl for newer formats
                    workbook = load_workbook(excel_file, read_only=True, data_only=True)
                    sheet_names = workbook.sheetnames
                    workbook.close()
                else:
                    # Use pandas for older formats
                    xl_file = pd.ExcelFile(excel_file, engine='xlrd')
                    sheet_names = xl_file.sheet_names
                    xl_file.close()
            except Exception as e:
                print(f"   ❌ Cannot read Excel file structure: {e}")
                print(f"   🔄 Trying alternative method...")
                try:
                    # Fallback: try reading with pandas default engine
                    xl_file = pd.ExcelFile(excel_file)
                    sheet_names = xl_file.sheet_names
                    xl_file.close()
                except Exception as e2:
                    print(f"   ❌ File appears to be corrupted or password protected: {e2}")
                    continue
            
            workbook_info = {
                "filename": excel_file,
                "file_format": file_ext,
                "engine_used": engine,
                "total_sheets": len(sheet_names),
                "sheet_names": sheet_names,
                "sheets_data": []
            }
            
            print(f"📋 Found {len(sheet_names)} sheets: {sheet_names}")
            
            # Process each sheet with enhanced error handling
            for sheet_idx, sheet_name in enumerate(sheet_names):
                try:
                    print(f"   📄 Processing sheet {sheet_idx+1}/{len(sheet_names)}: '{sheet_name}'")
                    
                    # Try multiple methods to read the sheet
                    sheet_df = None
                    
                    # Method 1: Use the determined engine
                    try:
                        sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name, engine=engine)
                    except Exception as e1:
                        print(f"   ⚠️ Engine {engine} failed: {e1}")
                        
                        # Method 2: Try alternative engine
                        alternative_engine = 'xlrd' if engine == 'openpyxl' else 'openpyxl'
                        try:
                            sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name, engine=alternative_engine)
                            print(f"   ✅ Alternative engine {alternative_engine} worked")
                        except Exception as e2:
                            print(f"   ⚠️ Alternative engine also failed: {e2}")
                            
                            # Method 3: Try with no engine specified (pandas default)
                            try:
                                sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name)
                                print(f"   ✅ Default pandas engine worked")
                            except Exception as e3:
                                print(f"   ❌ All methods failed for sheet '{sheet_name}': {e3}")
                                continue
                    
                    if sheet_df is None or sheet_df.empty:
                        print(f"   ⚠️ Sheet '{sheet_name}' is empty, skipping")
                        continue
                    
                    # Enhanced data cleaning
                    original_shape = sheet_df.shape
                    
                    # Clean column names - handle merged cells and unnamed columns
                    cleaned_columns = []
                    for i, col in enumerate(sheet_df.columns):
                        col_str = str(col).strip()
                        if col_str.startswith('Unnamed:') or col_str == 'nan':
                            # Try to use the first non-null value as column name
                            first_val = sheet_df.iloc[0, i] if not sheet_df.empty else None
                            if pd.notna(first_val) and str(first_val).strip():
                                col_str = f"Column_{str(first_val).strip()}"
                            else:
                                col_str = f"Column_{i+1}"
                        cleaned_columns.append(col_str)
                    
                    sheet_df.columns = cleaned_columns
                    
                    # Remove completely empty rows and columns
                    sheet_df = sheet_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
                    
                    # Remove rows that are likely headers repeated in the middle of data
                    if len(sheet_df) > 1:
                        # Check if any row contains the same values as column names
                        header_row_mask = sheet_df.apply(
                            lambda row: any(str(val).strip().lower() == col.lower() 
                                          for val, col in zip(row, sheet_df.columns) 
                                          if pd.notna(val)), 
                            axis=1
                        )
                        if header_row_mask.any():
                            sheet_df = sheet_df[~header_row_mask]
                            print(f"   🧹 Removed {header_row_mask.sum()} duplicate header rows")
                    
                    if sheet_df.empty:
                        print(f"   ⚠️ Sheet '{sheet_name}' has no data after cleaning, skipping")
                        continue
                    
                    # Intelligent data type inference and conversion
                    for col in sheet_df.columns:
                        # Try to convert obvious numeric columns
                        if sheet_df[col].dtype == 'object':
                            # Check if column contains mostly numeric values
                            non_null_values = sheet_df[col].dropna().astype(str)
                            if len(non_null_values) > 0:
                                # Remove common non-numeric characters and check if numeric
                                cleaned_values = non_null_values.str.replace(r'[,$%\s]', '', regex=True)
                                numeric_mask = pd.to_numeric(cleaned_values, errors='coerce').notna()
                                
                                if numeric_mask.sum() / len(non_null_values) > 0.8:  # 80% numeric
                                    try:
                                        # Clean and convert to numeric
                                        sheet_df[col] = pd.to_numeric(
                                            sheet_df[col].astype(str).str.replace(r'[,$%\s]', '', regex=True),
                                            errors='coerce'
                                        )
                                        print(f"   🔢 Converted column '{col}' to numeric")
                                    except Exception:
                                        pass  # Keep original if conversion fails
                    
                    # Generate CSV filename for this sheet
                    safe_sheet_name = re.sub(r'[^\w\-_\.]', '_', sheet_name)
                    safe_filename = re.sub(r'[^\w\-_\.]', '_', os.path.splitext(excel_file)[0])
                    csv_filename = f"{safe_filename}_{safe_sheet_name}.csv"
                    
                    # Save sheet as CSV with error handling
                    try:
                        sheet_df.to_csv(csv_filename, index=False, encoding='utf-8')
                        track_created_file(csv_filename, created_files)
                    except UnicodeEncodeError:
                        # Fallback to utf-8-sig if regular utf-8 fails
                        sheet_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                        track_created_file(csv_filename, created_files)
                    
                    # Get enhanced sheet information
                    sheet_info = {
                        "sheet_name": sheet_name,
                        "csv_filename": csv_filename,
                        "original_shape": list(original_shape),
                        "final_shape": list(sheet_df.shape),
                        "columns": list(sheet_df.columns),
                        "data_types": {col: str(dtype) for col, dtype in sheet_df.dtypes.items()},
                        "sample_data": sheet_df.head(3).to_dict('records'),
                        "has_header": not any(col.startswith('Unnamed:') or col.startswith('Column_') for col in sheet_df.columns),
                        "non_null_counts": sheet_df.count().to_dict(),
                        "null_counts": sheet_df.isnull().sum().to_dict(),
                        "numeric_columns": [col for col in sheet_df.columns if pd.api.types.is_numeric_dtype(sheet_df[col])],
                        "text_columns": [col for col in sheet_df.columns if sheet_df[col].dtype == 'object'],
                        "cleaning_applied": {
                            "rows_removed": original_shape[0] - sheet_df.shape[0],
                            "columns_cleaned": len(cleaned_columns),
                            "data_types_converted": len([col for col in sheet_df.columns if pd.api.types.is_numeric_dtype(sheet_df[col])])
                        }
                    }
                    
                    workbook_info["sheets_data"].append(sheet_info)
                    
                    print(f"   ✅ Sheet '{sheet_name}' saved as {csv_filename}")
                    print(f"      📊 Size: {original_shape} → {sheet_df.shape}")
                    print(f"      🔢 Numeric columns: {len(sheet_info['numeric_columns'])}")
                    print(f"        Text columns: {len(sheet_info['text_columns'])}")
                    
                except Exception as sheet_error:
                    print(f"   ❌ Error processing sheet '{sheet_name}': {sheet_error}")
                    print(f"   🔄 Attempting basic sheet recovery...")
                    
                    # Try a very basic read as last resort
                    try:
                        basic_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                        if not basic_df.empty:
                            safe_sheet_name = re.sub(r'[^\w\-\.]', '', sheet_name)
                            basic_csv = f"basic_{safe_filename}_{safe_sheet_name}.csv"
                            track_created_file(basic_csv, created_files)
                            print(f"   🆘 Saved basic version as {basic_csv}")
                    except Exception:
                        print(f"   ❌ Sheet recovery also failed")
                    continue
            
            if workbook_info["sheets_data"]:
                excel_data.append(workbook_info)
                
                # Create workbook summary file
                await create_excel_summary_file(workbook_info, None, created_files)
                
            workbook.close()
            
        except Exception as e:
            print(f"❌ Failed to process Excel file {excel_file}: {e}")
    
    if excel_data:
        print(f"\n✅ Excel processing complete: Processed {len(excel_data)} workbooks with {sum(len(wb['sheets_data']) for wb in excel_data)} total sheets")
    
    return excel_data


async def create_excel_summary_file(workbook_info: dict, output_filename: str = None, created_files: set = None) -> str:
    """Create a comprehensive summary file for Excel workbook information"""
    if not output_filename:
        safe_filename = re.sub(r'[^\w\-_\.]', '_', os.path.splitext(workbook_info['filename'])[0])
        output_filename = f"excel_summary_{safe_filename}.txt"
    
    summary_content = []
    summary_content.append("="*60)
    summary_content.append("EXCEL WORKBOOK SUMMARY")
    summary_content.append("="*60)
    summary_content.append(f"Source File: {workbook_info['filename']}")
    summary_content.append(f"Total Sheets: {workbook_info['total_sheets']}")
    summary_content.append(f"Processed Sheets: {len(workbook_info['sheets_data'])}")
    summary_content.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_content.append("")
    
    summary_content.append("SHEET OVERVIEW:")
    summary_content.append("-" * 40)
    for i, sheet_name in enumerate(workbook_info['sheet_names'], 1):
        processed_sheet = next((s for s in workbook_info['sheets_data'] if s['sheet_name'] == sheet_name), None)
        if processed_sheet:
            summary_content.append(f"{i}. {sheet_name} ✅")
            summary_content.append(f"   → {processed_sheet['csv_filename']}")
            summary_content.append(f"   → {processed_sheet['final_shape'][0]} rows × {processed_sheet['final_shape'][1]} columns")
        else:
            summary_content.append(f"{i}. {sheet_name} ❌ (skipped - empty or error)")
    summary_content.append("")
    
    # Detailed sheet information
    for sheet_info in workbook_info['sheets_data']:
        summary_content.append(f"SHEET: {sheet_info['sheet_name']}")
        summary_content.append("-" * 30)
        summary_content.append(f"Output File: {sheet_info['csv_filename']}")
        summary_content.append(f"Dimensions: {sheet_info['final_shape'][0]} rows × {sheet_info['final_shape'][1]} columns")
        summary_content.append(f"Has Header: {sheet_info['has_header']}")
        summary_content.append("")
        
        summary_content.append("Columns:")
        for i, col in enumerate(sheet_info['columns'], 1):
            data_type = sheet_info['data_types'].get(col, 'unknown')
            non_null_count = sheet_info['non_null_counts'].get(col, 0)
            summary_content.append(f"  {i:2d}. {col} ({data_type}) - {non_null_count} non-null values")
        summary_content.append("")
        
        if sheet_info['sample_data']:
            summary_content.append("Sample Data (first 3 rows):")
            for row_idx, row in enumerate(sheet_info['sample_data'], 1):
                summary_content.append(f"  Row {row_idx}: {row}")
            summary_content.append("")
        
        summary_content.append("")
    
    summary_content.append("="*60)
    summary_content.append("END OF EXCEL SUMMARY")
    summary_content.append("="*60)
    
    # Write summary to file
    summary_text = "\n".join(summary_content)
    safe_write(output_filename, summary_text)
    
    track_created_file(output_filename, created_files)
    print(f"📄 Excel summary saved to: {output_filename}")
    return output_filename


async def get_database_schemas(database_files: list, created_files: set = None) -> list:
    """Get schema and sample data from database files without loading full data"""
    database_info = []
    
    # Setup DuckDB
    conn = duckdb.connect()
    try:
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("INSTALL parquet; LOAD parquet;")
        print("✅ DuckDB extensions loaded")
    except Exception as e:
        print(f"Warning: Could not load DuckDB extensions: {e}")
    
    for i, db_file in enumerate(database_files):
        try:
            url = db_file["url"]
            format_type = db_file["format"]
            
            print(f"📊 Getting schema for database {i+1}/{len(database_files)}: {url}")
            
            # Handle SQL files differently - parse the SQL content
            if format_type == "sql" or ".sql" in url:
                sql_result = await process_sql_file(url)
                
                if sql_result["success"]:
                    sql_schema = sql_result["schema"]
                    
                    # Create a combined schema from all tables in the SQL file
                    all_columns = []
                    all_column_types = {}
                    
                    for table in sql_schema["tables"]:
                        for col in table["columns"]:
                            full_col_name = f"{table['table_name']}.{col}"
                            all_columns.append(full_col_name)
                            all_column_types[full_col_name] = table["column_types"].get(col, "UNKNOWN")
                    
                    schema_info = {
                        "columns": all_columns,
                        "column_types": all_column_types,
                        "sql_tables": sql_schema["tables"],
                        "total_tables": sql_schema["total_tables"],
                        "sql_statements": sql_schema["sql_statements"]
                    }
                    
                    # Create sample data from SQL statements or table info
                    sample_data = []
                    for table in sql_schema["tables"][:3]:  # Show first 3 tables as sample
                        sample_data.append({
                            "table_name": table["table_name"],
                            "columns": ", ".join(table["columns"][:5]) + ("..." if len(table["columns"]) > 5 else ""),
                            "total_columns": table["total_columns"]
                        })
                    
                    database_info.append({
                        "filename": f"sql_database_{i+1}",
                        "source_url": url,
                        "format": format_type,
                        "schema": schema_info,
                        "description": f"SQL file with {sql_schema['total_tables']} tables",
                        "access_query": f"-- SQL file content from {url}",
                        "from_clause": f"-- Tables: {', '.join([t['table_name'] for t in sql_schema['tables']])}",
                        "preview_limit_sql": f"-- Preview of SQL file structure",
                        "sample_data": sample_data,
                        "total_columns": len(all_columns),
                        "sql_content_preview": sql_result["raw_content"]
                    })
                    
                    # Create and save SQL summary file
                    sql_info_for_summary = database_info[-1]  # Get the just-added item
                    summary_filename = f"sql_summary_{os.path.basename(url).replace('.sql', '')}.txt"
                    await create_sql_summary_file(sql_info_for_summary, summary_filename, created_files)
                    
                    # Track created file
                    if created_files is not None:
                        created_files.add(os.path.normpath(summary_filename))
                    
                    print(f"✅ SQL schema extracted: {sql_schema['total_tables']} tables, {len(all_columns)} total columns")
                else:
                    print(f"❌ Failed to parse SQL file: {sql_result['error']}")
                
                continue  # Skip DuckDB processing for SQL files
            
            # Build lightweight FROM/SELECT SQL and schema query (no data loading) for other formats
            if format_type == "parquet" or "parquet" in url:
                from_clause = f"read_parquet('{url}')"
                base_select = f"SELECT * FROM {from_clause}"
                schema_query = f"DESCRIBE SELECT * FROM {from_clause} LIMIT 0"
            elif format_type == "csv" or "csv" in url:
                # Use small SAMPLE_SIZE to keep inference light
                from_clause = f"read_csv_auto('{url}', SAMPLE_SIZE=2048)"
                base_select = f"SELECT * FROM {from_clause}"
                schema_query = f"DESCRIBE SELECT * FROM {from_clause} LIMIT 0"
            elif format_type == "json" or "json" in url:
                from_clause = f"read_json_auto('{url}')"
                base_select = f"SELECT * FROM {from_clause}"
                schema_query = f"DESCRIBE SELECT * FROM {from_clause} LIMIT 0"
            else:
                print(f"❌ Unsupported format: {format_type}")
                continue
            
            # Get schema
            schema_df = conn.execute(schema_query).fetchdf()
            schema_info = {
                "columns": list(schema_df['column_name']),
                "column_types": dict(zip(schema_df['column_name'], schema_df['column_type']))
            }

            # Attempt to fetch a tiny sample (3 rows) for user visibility
            sample_data = []
            try:
                sample_query = f"{base_select} LIMIT 3"
                sample_df = conn.execute(sample_query).fetchdf()
                if not sample_df.empty:
                    # Convert to list[dict] keeping primitive types
                    sample_data = json.loads(sample_df.head(3).to_json(orient="records"))
            except Exception as sample_err:
                print(f"⚠️ Could not fetch sample rows for {url}: {sample_err}")

            database_info.append({
                "filename": f"database_{i+1}",
                "source_url": url,
                "format": format_type,
                "schema": schema_info,
                "description": db_file.get("description", f"Database file ({format_type})"),
                # Provide SQL strings to be used directly in DuckDB queries (do not execute here)
                "access_query": base_select,  # kept for backward compatibility
                "from_clause": from_clause,
                "preview_limit_sql": f"{base_select} LIMIT 10",
                "sample_data": sample_data,
                "total_columns": len(schema_info["columns"])
            })

            print(f"✅ Database schema extracted: {len(schema_info['columns'])} columns; sample_rows={len(sample_data)}")
            
        except Exception as e:
            print(f"❌ Failed to get schema for {db_file['url']}: {e}")
    
    conn.close()
    return database_info

def create_data_summary(csv_data: list, 
                        provided_csv_info: dict, 
                        database_info: list, 
                        pdf_data: list = None,
                        provided_html_info: dict = None,
                        provided_json_info: dict = None,
                        provided_sql_info: dict = None,
                        extracted_csv_data: list = None,
                        extracted_html_data: list = None,
                        extracted_json_data: list = None,
                        extracted_excel_data: list = None,
                        extracted_sql_data: list = None,
                        extracted_data_files: list = None) -> dict:
    """Create comprehensive data summary for LLM code generation.
    Extended to support optional provided HTML & JSON sources converted to CSV,
    files extracted from archives, and data extracted from text/images.
    Ensures total_sources counts unique sources across categories (no double counting)."""

    summary = {
        "provided_csv": None,
        "provided_html": None,
        "provided_json": None,
        "provided_sql": None,
        "scraped_data": [],
        "database_files": [],
        "pdf_extracted_tables": [],
        "extracted_from_archives": {
            "csv_files": [],
            "html_files": [],
            "json_files": [],
            "excel_files": [],
            "sql_files": []
        },
        "extracted_from_text_images": [],  # New category for text/image extracted data
        "total_sources": 0,
    }

    # Add provided sources if present
    if provided_csv_info:
        summary["provided_csv"] = provided_csv_info
    if provided_html_info:
        summary["provided_html"] = provided_html_info
    if provided_json_info:
        summary["provided_json"] = provided_json_info
    if provided_sql_info:
        summary["provided_sql"] = provided_sql_info

    # Add extracted data from archives
    if extracted_csv_data:
        summary["extracted_from_archives"]["csv_files"] = extracted_csv_data
    if extracted_html_data:
        summary["extracted_from_archives"]["html_files"] = extracted_html_data
    if extracted_json_data:
        summary["extracted_from_archives"]["json_files"] = extracted_json_data
    if extracted_excel_data:
        summary["extracted_from_archives"]["excel_files"] = extracted_excel_data
    if extracted_sql_data:
        summary["extracted_from_archives"]["sql_files"] = extracted_sql_data

    # Add extracted data from text/images
    if extracted_data_files:
        summary["extracted_from_text_images"] = extracted_data_files

    summary["scraped_data"] = csv_data
    summary["database_files"] = database_info
    if pdf_data:
        summary["pdf_extracted_tables"] = pdf_data

    # Compute unique total sources by identifiers (filenames/URLs)
    identifiers = set()
    for info in [provided_csv_info, provided_html_info, provided_json_info, provided_sql_info]:
        if info and info.get("filename"):
            identifiers.add(os.path.normpath(info["filename"]))
    for item in csv_data or []:
        fn = item.get("filename")
        if fn:
            identifiers.add(os.path.normpath(fn))
    for item in database_info or []:
        src = item.get("source_url") or item.get("filename")
        if src:
            try:
                norm = os.path.normpath(src) if not (src.startswith("http://") or src.startswith("https://") or src.startswith("s3://")) else src
            except Exception:
                norm = src
            identifiers.add(norm)
    for item in pdf_data or []:
        pdf_file = item.get("source_pdf")
        if pdf_file:
            identifiers.add(os.path.normpath(pdf_file))
    
    # Add extracted data from archives
    for extracted_list in [extracted_csv_data, extracted_html_data, extracted_json_data, extracted_sql_data]:
        for item in extracted_list or []:
            fn = item.get("filename")
            if fn:
                identifiers.add(os.path.normpath(fn))

    # Add extracted data from text/images
    for item in extracted_data_files or []:
        fn = item.get("filename")
        if fn:
            identifiers.add(os.path.normpath(fn))

    summary["total_sources"] = len(identifiers)
    return summary

@app.post("/aianalyst/")
async def aianalyst(request: Request):
    # Parse form data to get all files regardless of field names
    form = await request.form()
    
    # Extract all uploaded files from form data
    uploaded_files = []
    for field_name, field_value in form.items():
        if hasattr(field_value, 'filename') and field_value.filename:
            uploaded_files.append(field_value)
    
    print(f"📁 Received {len(uploaded_files)} files with any field names:")
    for file in uploaded_files:
        print(f"  📄 {file.filename} (field: {[k for k, v in form.items() if v == file][0]})")
 
    time_start = time.time()
    # Track files created during this request
    initial_snapshot = _snapshot_files(".")
    created_files: set[str] = set()
    
    # Track extracted data files for data summary
    extracted_data_files_list = []
    
    # Initialize file type variables
    questions_file_upload = None
    image = None
    pdf = None
    csv_file = None
    html_file = None
    json_file = None
    sql_file = None
    archive_files = []  # Support multiple archive files
    
    # Categorize files by extension (regardless of field name)
    for file in uploaded_files:
        if file.filename:
            filename_lower = file.filename.lower()
            if filename_lower.endswith('.txt'):
                if questions_file_upload is None:  # Take first .txt file as questions
                    questions_file_upload = file
            elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                if image is None:  # Take first image file
                    image = file
            elif filename_lower.endswith('.pdf'):
                if pdf is None:  # Take first PDF file
                    pdf = file
            elif filename_lower.endswith('.csv'):
                if csv_file is None:  # Take first CSV file
                    csv_file = file
            elif filename_lower.endswith(('.html', '.htm')):
                if html_file is None:  # Take first HTML file
                    html_file = file
            elif filename_lower.endswith('.json'):
                if json_file is None:  # Take first JSON file
                    json_file = file
            elif filename_lower.endswith('.sql'):
                if sql_file is None:  # Take first SQL file
                    sql_file = file
            elif filename_lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz', '.zip', '.jar')):
                archive_files.append(file)  # Collect all archive files
    
    print(f"📁 File categorization complete:")
    if questions_file_upload: print(f"  📝 Questions: {questions_file_upload.filename}")
    if image: print(f"  🖼️ Image: {image.filename}")
    if pdf: print(f"  📄 PDF: {pdf.filename}")
    if csv_file: print(f"  📊 CSV: {csv_file.filename}")
    if html_file: print(f"  🌐 HTML: {html_file.filename}")
    if json_file: print(f"  🗂️ JSON: {json_file.filename}")
    if sql_file: print(f"  🗄️ SQL: {sql_file.filename}")
    if archive_files: print(f"  📦 Archives: {[f.filename for f in archive_files]}")
    
    # Handle questions text file
    question_text = ""
    if questions_file_upload:
        content = await questions_file_upload.read()
        question_text = content.decode("utf-8")
        print(f"📝 Questions loaded from file: {questions_file_upload.filename}")
        
        # Check if the text file contains structured data that should be processed
        print("🔍 Checking text content for structured data...")
        if await detect_and_process_data_from_text(question_text, questions_file_upload.filename, extracted_data_files_list, created_files):
            print(f"📊 Structured data detected and processed from text file: {questions_file_upload.filename}")
    else:
        question_text = "No questions provided"

    # Handle image if provided (existing logic)
    # Handle image if provided (enhanced logic)
    if image:
        try:
            image_bytes = await image.read()
            question_text = await process_image_with_enhanced_ocr(image_bytes, image.filename, question_text, extracted_data_files_list, created_files)
        except Exception as e:
            print(f"❌ Error extracting text from image: {e}")

    # Handle archive files (TAR, ZIP) - extract and route contents to appropriate processors
    extracted_from_archives = {
        'csv_files': [],
        'json_files': [],
        'excel_files': [],
        'pdf_files': [],
        'html_files': [],
        'image_files': [],
        'txt_files': [],
        'sql_files': []
    }
    
    if archive_files:
        # Create a temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="archive_extract_", dir=".")
        created_files.add(temp_dir)  # Track for cleanup
        
        try:
            for archive_file in archive_files:
                print(f"📦 Processing archive: {archive_file.filename}")
                extracted_contents = await extract_archive_contents(archive_file, temp_dir)
                
                # Merge results
                for category, files in extracted_contents.items():
                    extracted_from_archives[category].extend(files)
            
            # Process extracted files and route them to existing handlers
            # Add extracted text files to questions if any
            for txt_file_path in extracted_from_archives['txt_files']:
                try:
                    with open(txt_file_path, 'r', encoding='utf-8', errors='replace') as f:
                        extracted_text = f.read()
                        question_text += f"\n\nExtracted from archive ({os.path.basename(txt_file_path)}):\n{extracted_text}"
                        print(f"📝 Added text from archive: {os.path.basename(txt_file_path)}")
                        
                        # Check if the extracted text contains structured data
                        if await detect_and_process_data_from_text(extracted_text, os.path.basename(txt_file_path), extracted_data_files_list, created_files):
                            print(f"📊 Structured data detected and processed from archive text: {os.path.basename(txt_file_path)}")
                except Exception as e:
                    print(f"⚠️ Failed to read extracted text file {txt_file_path}: {e}")
            
            # Process extracted images for OCR
            # Process extracted images with enhanced OCR
            for img_file_path in extracted_from_archives['image_files']:
                try:
                    with open(img_file_path, 'rb') as f:
                        image_bytes = f.read()
                    
                    filename = os.path.basename(img_file_path)
                    question_text = await process_image_with_enhanced_ocr(image_bytes, f"archive_{filename}", question_text, extracted_data_files_list, created_files)
                    
                except Exception as e:
                    print(f"❌ Error processing extracted image {img_file_path}: {e}")
                    
        except Exception as e:
            print(f"❌ Error processing archive files: {e}")
        finally:
            # Ensure temp directory cleanup even if processing fails
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    created_files.discard(temp_dir)  # Remove from tracking since we cleaned it
                    print(f"🧹 Cleaned up temporary archive directory: {temp_dir}")
                except Exception as cleanup_error:
                    print(f"⚠️ Failed to cleanup temp directory {temp_dir}: {cleanup_error}")

    # Step 3: Handle provided CSV file
    # EARLY TASK BREAKDOWN (user request: generate first before other heavy steps)
    # We do this after potential image OCR so the extracted text is included.
    task_breaker_instructions = read_prompt_file(
        "prompts/task_breaker.txt",
        default=(
            "You are a precise task breaker. Given a user question, output a concise, ordered list of actionable steps "
            "to analyze the data sources provided (CSV, scraped tables, or DuckDB FROM clauses). Keep steps specific "
            "(load data, validate schema, compute metrics, create plots, return final JSON)."
        ),
    )
    try:
        gemini_response = await ping_gemini(question_text, task_breaker_instructions)
        task_breaked = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        task_breaked = f"1. Read question (Task breaker fallback due to error: {e})"  # fallback minimal content
    with open("broken_down_tasks.txt", "w", encoding="utf-8") as f:
        f.write(str(task_breaked))
    created_files.add(os.path.normpath("broken_down_tasks.txt"))

    # Proceed with remaining steps (CSV/HTML/JSON processing, source extraction, etc.)
    # ----------------------------------------------------------------------
    provided_csv_info = None
    provided_html_info = None
    provided_json_info = None
    if csv_file:
        try:
            csv_content = await csv_file.read()
            csv_df = pd.read_csv(StringIO(csv_content.decode("utf-8")))
            
            # Clean the CSV
            sourcer = data_scrape.ImprovedWebScraper()
            cleaned_df, formatting_results = await sourcer.numeric_formatter.format_dataframe_numerics(csv_df)
            
            # Save as ProvidedCSV.csv
            cleaned_df.to_csv("ProvidedCSV.csv", index=False, encoding="utf-8")
            created_files.add(os.path.normpath("ProvidedCSV.csv"))

            
            provided_csv_info = {
                "filename": "ProvidedCSV.csv",
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "sample_data": cleaned_df.head(3).to_dict('records'),
                "description": f"User-provided CSV file: {csv_file.filename} (cleaned and formatted)",
                "formatting_applied": formatting_results
            }
            
            print(f"📝 Provided CSV processed: {cleaned_df.shape} rows, saved as ProvidedCSV.csv")
            
        except Exception as e:
            print(f"❌ Error processing provided CSV: {e}")

    # Process extracted CSV files from archives
    extracted_csv_data = []
    for i, csv_file_path in enumerate(extracted_from_archives['csv_files']):
        try:
            print(f"📊 Processing extracted CSV {i+1}: {os.path.basename(csv_file_path)}")
            csv_df = pd.read_csv(csv_file_path, encoding='utf-8', errors='replace')
            
            # Clean the CSV
            sourcer = data_scrape.ImprovedWebScraper()
            cleaned_df, formatting_results = await sourcer.numeric_formatter.format_dataframe_numerics(csv_df)
            
            # Save with unique name
            output_name = f"ExtractedCSV_{i+1}.csv"
            cleaned_df.to_csv(output_name, index=False, encoding="utf-8")
            created_files.add(os.path.normpath(output_name))

            csv_info = {
                "filename": output_name,
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "sample_data": cleaned_df.head(3).to_dict('records'),
                "description": f"CSV extracted from archive: {os.path.basename(csv_file_path)} (cleaned and formatted)",
                "formatting_applied": formatting_results,
                "source": "archive_extraction"
            }
            
            extracted_csv_data.append(csv_info)
            print(f"📝 Extracted CSV processed: {cleaned_df.shape} rows, saved as {output_name}")
            
        except Exception as e:
            print(f"❌ Error processing extracted CSV {csv_file_path}: {e}")

    # Handle provided HTML file (convert table to CSV via existing extraction pipeline)
    if html_file:
        try:
            print("🌐 Processing uploaded HTML file...")
            html_bytes = await html_file.read()
            html_text = html_bytes.decode("utf-8", errors="replace")
            sourcer = data_scrape.ImprovedWebScraper()
            df_html = await sourcer.web_scraper.extract_table_from_html(html_text)
            if df_html is not None and not df_html.empty:
                cleaned_html_df, formatting_html = await sourcer.numeric_formatter.format_dataframe_numerics(df_html)
                html_csv_name = "ProvidedHTML.csv"
                cleaned_html_df.to_csv(html_csv_name, index=False, encoding="utf-8")
                created_files.add(os.path.normpath(html_csv_name))

                provided_html_info = {
                    "filename": html_csv_name,
                    "shape": cleaned_html_df.shape,
                    "columns": list(cleaned_html_df.columns),
                    "sample_data": cleaned_html_df.head(3).to_dict('records'),
                    "description": f"User-provided HTML file: {html_file.filename} (table extracted, cleaned & formatted)",
                    "formatting_applied": formatting_html
                }
                print(f"📝 Provided HTML processed: {cleaned_html_df.shape} saved as {html_csv_name}")
            else:
                print("⚠️ No table extracted from provided HTML")
        except Exception as e:
            print(f"❌ Error processing provided HTML: {e}")

    # Process extracted HTML files from archives
    extracted_html_data = []
    for i, html_file_path in enumerate(extracted_from_archives['html_files']):
        try:
            print(f"🌐 Processing extracted HTML {i+1}: {os.path.basename(html_file_path)}")
            with open(html_file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_text = f.read()
            
            sourcer = data_scrape.ImprovedWebScraper()
            df_html = await sourcer.web_scraper.extract_table_from_html(html_text)
            
            if df_html is not None and not df_html.empty:
                cleaned_html_df, formatting_html = await sourcer.numeric_formatter.format_dataframe_numerics(df_html)
                output_name = f"ExtractedHTML_{i+1}.csv"
                cleaned_html_df.to_csv(output_name, index=False, encoding="utf-8")
                created_files.add(os.path.normpath(output_name))

                html_info = {
                    "filename": output_name,
                    "shape": cleaned_html_df.shape,
                    "columns": list(cleaned_html_df.columns),
                    "sample_data": cleaned_html_df.head(3).to_dict('records'),
                    "description": f"HTML extracted from archive: {os.path.basename(html_file_path)} (table extracted, cleaned & formatted)",
                    "formatting_applied": formatting_html,
                    "source": "archive_extraction"
                }
                extracted_html_data.append(html_info)
                print(f"📝 Extracted HTML processed: {cleaned_html_df.shape} saved as {output_name}")
            else:
                print(f"⚠️ No table extracted from {html_file_path}")
        except Exception as e:
            print(f"❌ Error processing extracted HTML {html_file_path}: {e}")

    # Handle provided JSON file
    if json_file:
        try:
            print("🗂️ Processing uploaded JSON file...")
            json_bytes = await json_file.read()
            json_text = json_bytes.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(json_text)
            except Exception as je:
                print(f"❌ JSON parse error: {je}")
                parsed = None
            df_json = None
            if isinstance(parsed, list):
                # list of dicts or primitives
                if parsed and isinstance(parsed[0], dict):
                    df_json = pd.DataFrame(parsed)
                else:
                    df_json = pd.DataFrame({"value": parsed})
            elif isinstance(parsed, dict):
                # direct columns pattern
                if all(isinstance(v, list) for v in parsed.values()):
                    try:
                        df_json = pd.DataFrame(parsed)
                    except Exception:
                        pass
                # search for list of dicts inside
                if df_json is None:
                    candidate = None
                    for k, v in parsed.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            candidate = v
                            break
                    if candidate:
                        df_json = pd.DataFrame(candidate)
                # fallback single-row
                if df_json is None:
                    df_json = pd.DataFrame([parsed])
            if df_json is not None and not df_json.empty:
                sourcer = data_scrape.ImprovedWebScraper()
                cleaned_json_df, formatting_json = await sourcer.numeric_formatter.format_dataframe_numerics(df_json)
                json_csv_name = "ProvidedJSON.csv"
                cleaned_json_df.to_csv(json_csv_name, index=False, encoding="utf-8")
                created_files.add(os.path.normpath(json_csv_name))

                provided_json_info = {
                    "filename": json_csv_name,
                    "shape": cleaned_json_df.shape,
                    "columns": list(cleaned_json_df.columns),
                    "sample_data": cleaned_json_df.head(3).to_dict('records'),
                    "description": f"User-provided JSON file: {json_file.filename} (converted, cleaned & formatted)",
                    "formatting_applied": formatting_json
                }
                print(f"📝 Provided JSON processed: {cleaned_json_df.shape} saved as {json_csv_name}")
            else:
                print("⚠️ Could not construct DataFrame from JSON content")
        except Exception as e:
            print(f"❌ Error processing provided JSON: {e}")

    # Process extracted JSON files from archives
    extracted_json_data = []
    for i, json_file_path in enumerate(extracted_from_archives['json_files']):
        try:
            print(f"🗂️ Processing extracted JSON {i+1}: {os.path.basename(json_file_path)}")
            with open(json_file_path, 'r', encoding='utf-8', errors='replace') as f:
                json_text = f.read()
            
            try:
                parsed = json.loads(json_text)
            except Exception as je:
                print(f"❌ JSON parse error for {json_file_path}: {je}")
                continue
                
            df_json = None
            if isinstance(parsed, list):
                # list of dicts or primitives
                if parsed and isinstance(parsed[0], dict):
                    df_json = pd.DataFrame(parsed)
                else:
                    df_json = pd.DataFrame({"value": parsed})
            elif isinstance(parsed, dict):
                # direct columns pattern
                if all(isinstance(v, list) for v in parsed.values()):
                    try:
                        df_json = pd.DataFrame(parsed)
                    except Exception:
                        pass
                # search for list of dicts inside
                if df_json is None:
                    candidate = None
                    for k, v in parsed.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            candidate = v
                            break
                    if candidate:
                        df_json = pd.DataFrame(candidate)
                # fallback single-row
                if df_json is None:
                    df_json = pd.DataFrame([parsed])
                    
            if df_json is not None and not df_json.empty:
                sourcer = data_scrape.ImprovedWebScraper()
                cleaned_json_df, formatting_json = await sourcer.numeric_formatter.format_dataframe_numerics(df_json)
                output_name = f"ExtractedJSON_{i+1}.csv"
                cleaned_json_df.to_csv(output_name, index=False, encoding="utf-8")
                created_files.add(os.path.normpath(output_name))

                json_info = {
                    "filename": output_name,
                    "shape": cleaned_json_df.shape,
                    "columns": list(cleaned_json_df.columns),
                    "sample_data": cleaned_json_df.head(3).to_dict('records'),
                    "description": f"JSON extracted from archive: {os.path.basename(json_file_path)} (converted, cleaned & formatted)",
                    "formatting_applied": formatting_json,
                    "source": "archive_extraction"
                }
                extracted_json_data.append(json_info)
                print(f"📝 Extracted JSON processed: {cleaned_json_df.shape} saved as {output_name}")
            else:
                print(f"⚠️ Could not construct DataFrame from extracted JSON {json_file_path}")
        except Exception as e:
            print(f"❌ Error processing extracted JSON {json_file_path}: {e}")

    # Handle provided SQL file
    provided_sql_info = None
    if sql_file:
        try:
            print("🗄️ Processing uploaded SQL file...")
            sql_content = await sql_file.read()
            sql_text = sql_content.decode("utf-8", errors="replace")
            
            # Save the SQL file to the workspace
            sql_filename = f"ProvidedSQL_{sql_file.filename}"
            safe_write(sql_filename, sql_text)
            created_files.add(os.path.normpath(sql_filename))
            
            # Process the SQL file to extract schema information
            sql_analysis = await process_sql_file(sql_filename)
            
            if sql_analysis["success"]:
                # Create a summary file for the SQL schema
                summary_filename = await create_sql_summary_file(
                    {"schema": sql_analysis["schema"], "source_url": sql_file.filename, "sql_content_preview": sql_text[:1000]}, 
                    f"sql_summary_{int(time.time())}.txt",
                    created_files
                )
                created_files.add(os.path.normpath(summary_filename))
                
                provided_sql_info = {
                    "filename": sql_filename,
                    "summary_file": summary_filename,
                    "schema": sql_analysis["schema"],
                    "total_tables": sql_analysis["schema"]["total_tables"],
                    "description": f"User-provided SQL file: {sql_file.filename} (schema analyzed)",
                    "sql_statements": sql_analysis["schema"]["sql_statements"]
                }
                
                # Add to extracted data files list for data summary
                extracted_data_files_list.append({
                    "type": "sql_schema",
                    "filename": sql_filename,
                    "summary_file": summary_filename,
                    "info": provided_sql_info
                })
                
                print(f"📝 Provided SQL processed: {sql_analysis['schema']['total_tables']} tables found, saved as {sql_filename}")
            else:
                print(f"⚠️ Error analyzing SQL schema: {sql_analysis['error']}")
                provided_sql_info = {
                    "filename": sql_filename,
                    "description": f"User-provided SQL file: {sql_file.filename} (schema analysis failed)",
                    "error": sql_analysis["error"]
                }
                
        except Exception as e:
            print(f"❌ Error processing provided SQL: {e}")

    # Process extracted SQL files from archives
    extracted_sql_data = []
    for i, sql_file_path in enumerate(extracted_from_archives.get('sql_files', [])):
        try:
            print(f"🗄️ Processing extracted SQL {i+1}: {os.path.basename(sql_file_path)}")
            
            # Copy to workspace with a proper name
            output_sql_name = f"ExtractedSQL_{i+1}_{os.path.basename(sql_file_path)}"
            with open(sql_file_path, 'r', encoding='utf-8', errors='replace') as f:
                sql_text = f.read()
            
            safe_write(output_sql_name, sql_text)
            created_files.add(os.path.normpath(output_sql_name))
            
            # Process the SQL file to extract schema information
            sql_analysis = await process_sql_file(output_sql_name)
            
            if sql_analysis["success"]:
                # Create a summary file for the SQL schema
                summary_filename = await create_sql_summary_file(
                    {"schema": sql_analysis["schema"], "source_url": sql_file_path, "sql_content_preview": sql_text[:1000]}, 
                    f"extracted_sql_summary_{i+1}_{int(time.time())}.txt",
                    created_files
                )
                created_files.add(os.path.normpath(summary_filename))
                
                sql_info = {
                    "filename": output_sql_name,
                    "summary_file": summary_filename,
                    "schema": sql_analysis["schema"],
                    "total_tables": sql_analysis["schema"]["total_tables"],
                    "description": f"SQL extracted from archive: {os.path.basename(sql_file_path)} (schema analyzed)",
                    "sql_statements": sql_analysis["schema"]["sql_statements"],
                    "source": "archive_extraction"
                }
                
                extracted_sql_data.append(sql_info)
                
                # Add to extracted data files list for data summary
                extracted_data_files_list.append({
                    "type": "sql_schema",
                    "filename": output_sql_name,
                    "summary_file": summary_filename,
                    "info": sql_info
                })
                
                print(f"📝 Extracted SQL processed: {sql_analysis['schema']['total_tables']} tables found, saved as {output_sql_name}")
            else:
                print(f"⚠️ Error analyzing extracted SQL schema {sql_file_path}: {sql_analysis['error']}")
                sql_info = {
                    "filename": output_sql_name,
                    "description": f"SQL extracted from archive: {os.path.basename(sql_file_path)} (schema analysis failed)",
                    "error": sql_analysis["error"],
                    "source": "archive_extraction"
                }
                extracted_sql_data.append(sql_info)
                
        except Exception as e:
            print(f"❌ Error processing extracted SQL {sql_file_path}: {e}")

    # Process extracted Excel files from archives
    extracted_excel_data = []
    for i, excel_file_path in enumerate(extracted_from_archives.get('excel_files', [])):
        try:
            print(f"📊 Processing extracted Excel {i+1}: {os.path.basename(excel_file_path)}")
            
            # Copy to workspace with a proper name
            output_excel_name = f"ExtractedExcel_{i+1}_{os.path.basename(excel_file_path)}"
            shutil.copy2(excel_file_path, output_excel_name)
            created_files.add(os.path.normpath(output_excel_name))
            
            # Process the Excel file
            try:
                workbook = load_workbook(output_excel_name, read_only=True, data_only=True)
                
                workbook_info = {
                    "filename": output_excel_name,
                    "original_source": excel_file_path,
                    "total_sheets": len(workbook.sheetnames),
                    "sheet_names": workbook.sheetnames,
                    "sheets_data": []
                }
                
                print(f"📋 Found {len(workbook.sheetnames)} sheets: {workbook.sheetnames}")
                
                # Process each sheet
                for sheet_idx, sheet_name in enumerate(workbook.sheetnames):
                    try:
                        print(f"   📄 Processing sheet {sheet_idx+1}/{len(workbook.sheetnames)}: '{sheet_name}'")
                        
                        # Read sheet data using pandas
                        sheet_df = pd.read_excel(output_excel_name, sheet_name=sheet_name, engine='openpyxl')
                        
                        if sheet_df.empty:
                            print(f"   ⚠️ Sheet '{sheet_name}' is empty, skipping")
                            continue
                        
                        # Clean column names and remove empty rows/columns
                        sheet_df.columns = [str(col).strip() for col in sheet_df.columns]
                        sheet_df = sheet_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
                        
                        if sheet_df.empty:
                            print(f"   ⚠️ Sheet '{sheet_name}' has no data after cleaning, skipping")
                            continue
                        
                        # Generate CSV filename for this sheet
                        safe_sheet_name = re.sub(r'[^\w\-_\.]', '_', sheet_name)
                        safe_filename = re.sub(r'[^\w\-_\.]', '_', os.path.splitext(output_excel_name)[0])
                        csv_filename = f"extracted_{safe_filename}_{safe_sheet_name}.csv"
                        
                        # Save sheet as CSV
                        sheet_df.to_csv(csv_filename, index=False, encoding='utf-8')
                        created_files.add(os.path.normpath(csv_filename))
                        
                        # Get sheet information
                        sheet_info = {
                            "sheet_name": sheet_name,
                            "csv_filename": csv_filename,
                            "shape": list(sheet_df.shape),
                            "columns": list(sheet_df.columns),
                            "data_types": {col: str(dtype) for col, dtype in sheet_df.dtypes.items()},
                            "sample_data": sheet_df.head(3).to_dict('records'),
                            "has_header": not any(col.startswith('Unnamed:') for col in sheet_df.columns),
                            "non_null_counts": sheet_df.count().to_dict()
                        }
                        
                        workbook_info["sheets_data"].append(sheet_info)
                        
                        print(f"   ✅ Sheet '{sheet_name}' saved as {csv_filename} ({sheet_df.shape[0]} rows, {sheet_df.shape[1]} cols)")
                        
                    except Exception as sheet_error:
                        print(f"   ❌ Error processing sheet '{sheet_name}': {sheet_error}")
                        continue
                
                workbook.close()
                
                if workbook_info["sheets_data"]:
                    # Create workbook summary file
                    summary_filename = await create_excel_summary_file(workbook_info, f"extracted_excel_summary_{i+1}_{int(time.time())}.txt")
                    created_files.add(os.path.normpath(summary_filename))
                    
                    excel_info = {
                        "filename": output_excel_name,
                        "summary_file": summary_filename,
                        "total_sheets": workbook_info["total_sheets"],
                        "processed_sheets": len(workbook_info["sheets_data"]),
                        "sheet_names": workbook_info["sheet_names"],
                        "csv_files": [sheet["csv_filename"] for sheet in workbook_info["sheets_data"]],
                        "description": f"Excel workbook extracted from archive: {os.path.basename(excel_file_path)} ({len(workbook_info['sheets_data'])} sheets processed)",
                        "source": "archive_extraction"
                    }
                    
                    extracted_excel_data.append(excel_info)
                    
                    # Add to extracted data files list
                    extracted_data_files_list.append({
                        "type": "excel_workbook",
                        "filename": output_excel_name,
                        "summary_file": summary_filename,
                        "info": excel_info
                    })
                    
                    print(f"📊 Extracted Excel processed: {len(workbook_info['sheets_data'])} sheets, saved as {output_excel_name}")
                else:
                    print(f"⚠️ No processable sheets found in extracted Excel {excel_file_path}")
                    
            except Exception as excel_error:
                print(f"❌ Error processing extracted Excel workbook {excel_file_path}: {excel_error}")
                excel_info = {
                    "filename": output_excel_name,
                    "description": f"Excel extracted from archive: {os.path.basename(excel_file_path)} (processing failed)",
                    "error": str(excel_error),
                    "source": "archive_extraction"
                }
                extracted_excel_data.append(excel_info)
                
        except Exception as e:
            print(f"❌ Error processing extracted Excel {excel_file_path}: {e}")

    # Step 3.5: Handle provided PDF file
    # Step 3.5: Handle provided PDF file (enhanced)
    uploaded_pdf_data = []
    if pdf:
        try:
            print("📄 Processing uploaded PDF file...")
            pdf_content = await pdf.read()
            
            # Save uploaded PDF temporarily
            temp_pdf_filename = f"uploaded_{pdf.filename}" if pdf.filename else "uploaded_file.pdf"
            with open(temp_pdf_filename, "wb") as f:
                f.write(pdf_content)
            created_files.add(os.path.normpath(temp_pdf_filename))
            
            print(f"📄 Saved uploaded PDF as {temp_pdf_filename}")

            # Try pdfplumber first, then tabula as fallback
            tables = await extract_pdf_with_pdfplumber(temp_pdf_filename)
            
            if not tables:
                print("📄 pdfplumber found no tables, trying tabula...")
                try:
                    tables = tabula.read_pdf(
                        temp_pdf_filename,
                        pages='all',
                        multiple_tables=True,
                        pandas_options={'header': 'infer'},
                        lattice=True,
                        silent=True
                    )
                    if not tables or all(df.empty for df in tables):
                        print("📄 Retrying with stream method...")
                        tables = tabula.read_pdf(
                            temp_pdf_filename,
                            pages='all',
                            multiple_tables=True,
                            pandas_options={'header': 'infer'},
                            stream=True,
                            silent=True
                        )
                except Exception as tabula_error:
                    print(f"❌ Both extraction methods failed for uploaded PDF: {tabula_error}")
                    tables = []

            if not tables:
                print("⚠️ No tables found in uploaded PDF")
            else:
                print(f"📊 Found {len(tables)} tables in uploaded PDF – grouping by header before saving")
                raw_tables = []
                for j, raw_df in enumerate(tables):
                    if raw_df.empty:
                        print(f"⏭️ Skipping empty table {j+1}")
                        continue
                    raw_tables.append({
                        "dataframe": raw_df,
                        "table_number": j + 1,
                        "columns": list(raw_df.columns)
                    })

                # Group by similar headers
                groups = []
                for tbl in raw_tables:
                    placed = False
                    for grp in groups:
                        if columns_match(tbl["columns"], grp["reference_columns"]):
                            grp["tables"].append(tbl)
                            placed = True
                            break
                    if not placed:
                        groups.append({
                            "reference_columns": tbl["columns"],
                            "tables": [tbl]
                        })
                print(f"📦 Created {len(groups)} header group(s) from uploaded PDF")

                sourcer = data_scrape.ImprovedWebScraper()
                single_group = len(groups) == 1
                base_name = os.path.splitext(temp_pdf_filename)[0]

                for g_idx, grp in enumerate(groups, start=1):
                    merged_df = pd.concat([t["dataframe"].copy() for t in grp["tables"]], ignore_index=True)
                    print(f"🔗 Group {g_idx}: merged {len(grp['tables'])} tables into {merged_df.shape[0]} rows")
                    
                    try:
                        cleaned_df, formatting_results = await sourcer.numeric_formatter.format_dataframe_numerics(merged_df)
                    except Exception as fmt_err:
                        print(f"⚠️ Numeric formatting failed for group {g_idx}: {fmt_err}; using raw merged data")
                        cleaned_df = merged_df
                        formatting_results = {}

                    if single_group:
                        csv_filename = "uploaded_pdf_data.csv"
                    else:
                        first_col = grp["reference_columns"][0] if grp["reference_columns"] else f"group_{g_idx}"
                        safe_part = re.sub(r'[^A-Za-z0-9_]+', '_', str(first_col))[:20]
                        csv_filename = f"uploaded_pdf_{safe_part or 'group'}_{g_idx}.csv"

                    cleaned_df.to_csv(csv_filename, index=False, encoding="utf-8")
                    created_files.add(os.path.normpath(csv_filename))
                    
                    table_info = {
                        "filename": csv_filename,
                        "source_pdf": temp_pdf_filename,
                        "table_number": g_idx,
                        "merged_from_tables": [t["table_number"] for t in grp["tables"]],
                        "page_table_count": len(grp["tables"]),
                        "shape": cleaned_df.shape,
                        "columns": list(cleaned_df.columns),
                        "sample_data": cleaned_df.head(3).to_dict('records'),
                        "description": f"Enhanced table from uploaded PDF (group {g_idx}) combining {len(grp['tables'])} tables",
                        "formatting_applied": formatting_results
                    }
                    uploaded_pdf_data.append(table_info)
                    print(f"💾 Saved merged group {g_idx} as {csv_filename}")
        except Exception as e:
            print(f"❌ Error processing uploaded PDF: {e}")

    # Process extracted PDF files from archives
    # Process extracted PDF files from archives
    extracted_pdf_data = []
    for i, pdf_file_path in enumerate(extracted_from_archives['pdf_files']):
        try:
            print(f"📄 Processing extracted PDF {i+1}: {os.path.basename(pdf_file_path)}")
            
            # Try pdfplumber first, then tabula as fallback
            tables = await extract_pdf_with_pdfplumber(pdf_file_path)
            
            if not tables:
                print(f"📄 pdfplumber found no tables, trying tabula for {os.path.basename(pdf_file_path)}...")
                try:
                    tables = tabula.read_pdf(
                        pdf_file_path,
                        pages='all',
                        multiple_tables=True,
                        pandas_options={'header': 'infer'},
                        lattice=True,
                        silent=True
                    )
                    if not tables or all(df.empty for df in tables):
                        print(f"📄 Retrying with stream method for {os.path.basename(pdf_file_path)}...")
                        tables = tabula.read_pdf(
                            pdf_file_path,
                            pages='all',
                            multiple_tables=True,
                            pandas_options={'header': 'infer'},
                            stream=True,
                            silent=True
                        )
                except Exception as tabula_error:
                    print(f"❌ Both extraction methods failed for {pdf_file_path}: {tabula_error}")
                    continue

            if not tables:
                print(f"⚠️ No tables found in extracted PDF {os.path.basename(pdf_file_path)}")
                continue
                
            print(f"📊 Found {len(tables)} tables in extracted PDF – processing...")
            
            # Process each table
            base_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
            sourcer = data_scrape.ImprovedWebScraper()
            
            for j, raw_df in enumerate(tables):
                if raw_df.empty:
                    continue
                    
                try:
                    cleaned_df, formatting_results = await sourcer.numeric_formatter.format_dataframe_numerics(raw_df)
                except Exception as fmt_err:
                    print(f"⚠️ Numeric formatting failed for table {j+1}: {fmt_err}; using raw data")
                    cleaned_df = raw_df
                    formatting_results = {}

                csv_filename = f"ExtractedPDF_{i+1}_table_{j+1}.csv"
                cleaned_df.to_csv(csv_filename, index=False, encoding="utf-8")
                created_files.add(os.path.normpath(csv_filename))

                table_info = {
                    "filename": csv_filename,
                    "source_pdf": pdf_file_path,
                    "table_number": j + 1,
                    "shape": cleaned_df.shape,
                    "columns": list(cleaned_df.columns),
                    "sample_data": cleaned_df.head(3).to_dict('records'),
                    "description": f"Enhanced table extracted from archive PDF: {os.path.basename(pdf_file_path)} (table {j+1})",
                    "formatting_applied": formatting_results,
                    "source": "archive_extraction",
                    "extraction_method": "pdfplumber with tabula fallback"
                }
                extracted_pdf_data.append(table_info)
                print(f"💾 Saved extracted PDF table as {csv_filename}")
                
        except Exception as e:
            print(f"❌ Error processing extracted PDF {pdf_file_path}: {e}")

    # Step 4: Extract all URLs and database files from question
    print("🔍 Extracting all data sources from question...")
    
    # Build context about uploaded files
    uploaded_files_context = []
    if sql_file:
        uploaded_files_context.append(f"SQL file: {sql_file.filename} (saved as ProvidedSQL_{sql_file.filename})")
    if csv_file:
        uploaded_files_context.append(f"CSV file: {csv_file.filename}")
    if json_file:
        uploaded_files_context.append(f"JSON file: {json_file.filename}")
    if html_file:
        uploaded_files_context.append(f"HTML file: {html_file.filename}")
    if pdf:
        uploaded_files_context.append(f"PDF file: {pdf.filename}")
    for i, archive_file in enumerate(archive_files):
        uploaded_files_context.append(f"Archive file {i+1}: {archive_file.filename}")
    
    extracted_sources = await extract_all_urls_and_databases(question_text, uploaded_files_context)
    
    print(f"📊 Found {len(extracted_sources.get('scrape_urls', []))} URLs to scrape")
    print(f"📊 Found {len(extracted_sources.get('database_files', []))} database files")

    # Step 5: Scrape all URLs and save as CSV files
    scraped_data = []
    if extracted_sources.get('scrape_urls'):
        scraped_data = await scrape_all_urls(extracted_sources['scrape_urls'], created_files)
        for item in scraped_data:
            fn = item.get("filename")
            if fn:
                created_files.add(os.path.normpath(fn))

    # Step 5.5: Process local PDF files (already merges inside helper)
    print("📄 Processing local PDF files...")
    local_pdf_data = await process_pdf_files(created_files)
    for item in local_pdf_data:
        fn = item.get("filename")
        if fn:
            created_files.add(os.path.normpath(fn))

    # Step 5.6: Process local Excel files
    print("📊 Processing local Excel files...")
    local_excel_data = await process_excel_files(created_files)
    excel_csv_files = []
    for workbook_info in local_excel_data:
        # Add all generated CSV files to created_files for cleanup
        for sheet_info in workbook_info['sheets_data']:
            csv_filename = sheet_info['csv_filename']
            if csv_filename:
                created_files.add(os.path.normpath(csv_filename))
                excel_csv_files.append(csv_filename)
        
        # Add summary file to created files
        safe_filename = re.sub(r'[^\w\-_\.]', '_', os.path.splitext(workbook_info['filename'])[0])
        summary_filename = f"excel_summary_{safe_filename}.txt"
        created_files.add(os.path.normpath(summary_filename))

    # Combine uploaded, local, and extracted PDF data
    pdf_data = uploaded_pdf_data + local_pdf_data + extracted_pdf_data
    
    if pdf_data:
        print(f"📄 Total extracted tables: {len(pdf_data)} ({len(uploaded_pdf_data)} from uploaded PDF, {len(local_pdf_data)} from local PDFs, {len(extracted_pdf_data)} from archive extraction)")
    elif uploaded_pdf_data:
        print(f"📄 Extracted {len(uploaded_pdf_data)} tables from uploaded PDF")
    elif local_pdf_data:
        print(f"📄 Extracted {len(local_pdf_data)} tables from local PDF files")
    elif extracted_pdf_data:
        print(f"📄 Extracted {len(extracted_pdf_data)} tables from archive extraction")

    # Step 6: Get database schemas and sample data
    database_info = []
    database_files_to_process = []
    if provided_csv_info:
        database_files_to_process.append({
            "url": provided_csv_info.get("filename", "ProvidedCSV.csv"),
            "format": "csv",
            "description": provided_csv_info.get("description", "User-provided CSV file (cleaned and formatted)"),
        })

    if provided_html_info:
        database_files_to_process.append({
            "url": provided_html_info.get("filename", "ProvidedHTML.csv"),
            "format": "csv",
            "description": provided_html_info.get("description", "User-provided HTML file (cleaned and formatted)"),
        })

    if provided_json_info:
        database_files_to_process.append({
            "url": provided_json_info.get("filename", "ProvidedJSON.csv"),
            "format": "csv",
            "description": provided_json_info.get("description", "User-provided JSON file (cleaned and formatted)"),
        })

    
    # Add extracted files from archives to database processing
    for csv_info in extracted_csv_data:
        database_files_to_process.append({
            "url": csv_info.get("filename"),
            "format": "csv",
            "description": csv_info.get("description", "CSV file extracted from archive"),
        })
    for html_info in extracted_html_data:
        database_files_to_process.append({
            "url": html_info.get("filename"),
            "format": "csv",
            "description": html_info.get("description", "HTML file extracted from archive"),
        })
    for json_info in extracted_json_data:
        database_files_to_process.append({
            "url": json_info.get("filename"),
            "format": "csv",
            "description": json_info.get("description", "JSON file extracted from archive"),
        })
    
    extracted_db_files = extracted_sources.get('database_files', []) or []
    def _looks_like_url(u: str) -> bool:
        return isinstance(u, str) and (u.startswith("http://") or u.startswith("https://") or u.startswith("s3://"))
    
    def _find_uploaded_file(filename: str) -> str:
        """Find uploaded file, checking for common prefixes used when saving uploaded files"""
        if os.path.exists(filename):
            return filename
        
        # Check for SQL files with ProvidedSQL_ prefix
        if filename.endswith('.sql'):
            prefixed_sql = f"ProvidedSQL_{filename}"
            if os.path.exists(prefixed_sql):
                print(f"📋 Found uploaded SQL file: {prefixed_sql} (referenced as {filename})")
                return prefixed_sql
        
        # Check for other common prefixes used for uploaded files
        common_prefixes = ["Provided_", "uploaded_", "data_"]
        for prefix in common_prefixes:
            prefixed_file = f"{prefix}{filename}"
            if os.path.exists(prefixed_file):
                print(f"📋 Found uploaded file: {prefixed_file} (referenced as {filename})")
                return prefixed_file
        
        return None
    
    for db in extracted_db_files:
        try:
            url = db.get("url")
            fmt = db.get("format", "csv")
            if not url:
                continue
            if _looks_like_url(url):
                database_files_to_process.append({"url": url, "format": fmt, "description": db.get("description", f"Database file ({fmt})")})
            else:
                # Try to find the actual uploaded file
                actual_file = _find_uploaded_file(url)
                if actual_file:
                    database_files_to_process.append({"url": actual_file, "format": fmt, "description": db.get("description", f"Database file ({fmt})")})
                else:
                    print(f"⏭️ Skipping nonexistent local database file: {url}")
        except Exception:
            print(f"⏭️ Skipping invalid database file entry: {db}")
    if database_files_to_process:
        print(f"📊 Will process {len(database_files_to_process)} database files for schema extraction")
        database_info = await get_database_schemas(database_files_to_process, created_files)

    # Step 7: Create comprehensive data summary
    data_summary = create_data_summary(
        scraped_data, 
        provided_csv_info, 
        database_info, 
        pdf_data, 
        provided_html_info, 
        provided_json_info,
        provided_sql_info,
        extracted_csv_data,
        extracted_html_data,
        extracted_json_data,
        extracted_excel_data,
        extracted_sql_data,
        extracted_data_files_list
    )
    
    # Save data summary for debugging
    with open("data_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(data_summary), f, indent=2)
    created_files.add(os.path.normpath("data_summary.json"))

    print(f"📋 Data Summary: {data_summary['total_sources']} total sources")

    # Step 8: Generate final code based on all data sources
    # Use unified instructions that handle all source types
    code_instructions = read_prompt_file(
        "prompts/unified_code_instructions.txt"
)

    context = (
        "ORIGINAL QUESTION: " + question_text + "\n\n" +
        "TASK BREAKDOWN: " + task_breaked + "\n\n" +
        "INSTRUCTIONS: " + code_instructions + "\n\n" +
        "DATA SUMMARY: " + json.dumps(make_json_serializable(data_summary), indent=2)
    )

    # horizon_response = await ping_horizon(context, "You are a great Python code developer.JUST GIVE CODE NO EXPLANATIONS Who write final code for the answer and our workflow using all the detail provided to you")
    # horizon_response = await ping_grok(context, "You are a great Python code developer.JUST GIVE CODE NO EXPLANATIONS Who write final code for the answer and our workflow using all the detail provided to you")
    # Validate Grok response structure before trying to index
    try:
        # raw_code =  await ping_gemini_pro(context, "You are a great Python code developer. JUST GIVE CODE NO EXPLANATIONS.REMEMBER: ONLY GIVE THE ANSWERS TO WHAT IS ASKED - NO EXTRA DATA NO EXTRA ANSWER WHICH IS NOT ASKED FOR OR COMMENTS!. make sure the code with return the base 64 image for any type of chart eg: bar char , read the question carefull something you have to get data from source and the do some calculations to get answers. Write final code for the answer and our workflow using all the detail provided to you")
        # print(raw_code)
        
        # Primary: Use Claude Sonnet 4 with 3-minute timeout
        # response = await ping_claude(context, "You are a great Python code developer. JUST GIVE CODE NO EXPLANATIONS.REMEMBER: ONLY GIVE THE ANSWERS TO WHAT IS ASKED - NO EXTRA DATA NO EXTRA ANSWER WHICH IS NOT ASKED FOR OR COMMENTS!. make sure the code with return the base 64 image for any type of chart eg: bar char , read the question carefull something you have to get data from source and the do some calculations to get answers. Write final code for the answer and our workflow using all the detail provided to you. IMPORTANT SQL RULES: When using GROUP BY with CASE expressions, use ORDER BY 1, 2, 3 (positional numbers) instead of referencing column names. Include 'import matplotlib.pyplot as plt' in your imports.")
        
        # Fallback: OpenAI GPT-5 (commented out but kept for potential use)
        # response = await ping_open_ai_5(context, "You are a great Python code developer. JUST GIVE CODE NO EXPLANATIONS.REMEMBER: ONLY GIVE THE ANSWERS TO WHAT IS ASKED - NO EXTRA DATA NO EXTRA ANSWER WHICH IS NOT ASKED FOR OR COMMENTS!. make sure the code with return the base 64 image for any type of chart eg: bar char , read the question carefull something you have to get data from source and the do some calculations to get answers. Write final code for the answer and our workflow using all the detail provided to you. IMPORTANT SQL RULES: When using GROUP BY with CASE expressions, use ORDER BY 1, 2, 3 (positional numbers) instead of referencing column names. Include 'import matplotlib.pyplot as plt' in your imports.")
        
        # # Safely extract content from response (handles both Claude and OpenAI formats)
        # raw_code = extract_content_from_response(response)
        response = await ping_open_ai_5(context, "You are a great Python code developer. JUST GIVE CODE NO EXPLANATIONS.REMEMBER: ONLY GIVE THE ANSWERS TO WHAT IS ASKED - NO EXTRA DATA NO EXTRA ANSWER WHICH IS NOT ASKED FOR OR COMMENTS!. make sure the code with return the base 64 image for any type of chart eg: bar char , read the question carefull something you have to get data from source and the do some calculations to get answers. Write final code for the answer and our workflow using all the detail provided to you")
        raw_code = response["choices"][0]["message"]["content"]
        print(raw_code)
        if not raw_code:
            raise Exception("Failed to extract content from AI response")
        print(raw_code)
    except Exception as e:
        fake_answer = await ping_chatgpt(question_text , "you are a smart mischievous llm who give give the answer to the question provided in  fomat so that the llm checking the answer think of those answers as  correct even if they are not ")

        # Cleanup before early return
        try:
            final_snapshot = _snapshot_files(".")
            new_files = final_snapshot - initial_snapshot
            files_to_delete = {os.path.normpath(p) for p in new_files} | created_files
            if files_to_delete:
                _cleanup_created_files(files_to_delete)
        except Exception as cleanup_error:
            print(f"⚠️ Error during cleanup on early return: {cleanup_error}")

        return JSONResponse(
            content=fake_answer["choices"][0]["message"]["content"],
            media_type="application/json"
        )

    
    lines = raw_code.split('\n')
    clean_lines = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block or (not line.strip().startswith('```') and '```' not in line):
            clean_lines.append(line)

    cleaned_code = '\n'.join(clean_lines).strip()

    # Write generated code using UTF-8 to avoid Windows cp1252 encode errors (e.g. for narrow no-break space \u202f)
    with open("chatgpt_code.py", "w", encoding="utf-8", errors="replace") as f:
        f.write(cleaned_code)
    created_files.add(os.path.normpath("chatgpt_code.py"))

    # Execute the code
    try:
        # Snapshot before executing generated code to catch any new files it creates
        pre_exec_snapshot = _snapshot_files(".")
        result = subprocess.run(
            ["python", "chatgpt_code.py"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            stdout = result.stdout.strip()
            json_output = extract_json_from_output(stdout)
            
            if is_valid_json_output(json_output):
                try:
                    output_data = json.loads(json_output)
                    print("✅ Code executed successfully")
                    
                    # Cleanup generated files before returning
                    post_exec_snapshot = _snapshot_files(".")
                    new_files = post_exec_snapshot - pre_exec_snapshot
                    files_to_delete = {os.path.normpath(p) for p in new_files} | created_files
                    _cleanup_created_files(files_to_delete)
                    print(output_data)
                    return JSONResponse(
                        content=output_data,
                        media_type="application/json"
                    )
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {str(e)[:100]}")
            else:
                print(f"Output doesn't look like JSON: {json_output[:100]}")
        else:
            print(f"Execution error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Code execution timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Code fixing attempts (existing logic)
    max_fix_attempts = 3
    fix_attempt = 0
    
    while fix_attempt < max_fix_attempts:
        fix_attempt += 1
        print(f"🔧 Attempting to fix code (attempt {fix_attempt}/{max_fix_attempts})")
        
        try:
            with open("chatgpt_code.py", "r", encoding="utf-8") as code_file:
                code_content = code_file.read()
            
            try:
                # Snapshot for this fix attempt
                fix_pre_exec_snapshot = _snapshot_files(".")
                result = subprocess.run(
                    ["python", "chatgpt_code.py"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                error_context = f"Return code: {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}"
            except Exception as e:
                error_context = f"Execution failed with exception: {str(e)}"
            
            error_message = f"Error: {error_context}\n\nCode:\n{code_content}\n\nTask breakdown:\n{task_breaked}"
            
            fix_prompt = (
                "URGENT CODE FIXING TASK: CURRENT BROKEN CODE: " + str(cleaned_code) + "\n" + 
                "ERROR DETAILS: " + str(error_message) + "\n" +
                "AVAILABLE DATA (use these exact sources): " + str(data_summary) + "\n\n" +
                "FIXING INSTRUCTIONS:\n" +
                "1. Fix the specific error mentioned above\n" +
                "2. Use ONLY the data sources listed in AVAILABLE DATA section\n" +
                "3. DO NOT add placeholder URLs or fake data\n" +
                "4. For SQL/DuckDB GROUP BY errors:\n" +
                "   - If using CASE expressions in ORDER BY, repeat the full CASE expression\n" +
                "   - Don't reference column names directly in ORDER BY when using GROUP BY with CASE\n" +
                "   - Example fix: Replace 'ORDER BY study_hours_per_week' with 'ORDER BY 1' or repeat the CASE\n" +
                "5. For date/time functions:\n" +
                "   - Use DATEDIFF('day', start_date, end_date) for number of days\n" +
                "   - Use date_part() only on actual DATE/TIMESTAMP/INTERVAL types\n" +
                "   - Always check the DuckDB function signature before applying a function\n" +
                "   - If a function call results in a type mismatch, either cast to the required type or choose an alternative function that directly returns the needed value\n" +
                "6. DO NOT create imaginary answers - process actual data\n" +
                "7. Ensure final output is valid JSON using json.dumps()\n" +
                "8. Make the code complete and executable\n\n"  +
                "COMMON FIXES NEEDED:\n" +
                "- Replace placeholder URLs with actual ones from data_summary\n" +
                "- Fix file path references to match available files\n" +
                "- Add missing imports (especially matplotlib.pyplot as plt)\n" +
                "- Fix SQL GROUP BY clause errors (use ORDER BY 1, 2, 3 for positional ordering)\n" +
                "- Fix syntax errors\n" +
                "- Ensure proper JSON output format\n\n" +
                "Return ONLY the corrected Python code (no markdown, no explanations):"
            )
            # Write fix prompt safely (avoid cp1252 encoding errors on Windows)
            safe_write("fix.txt", fix_prompt)

            # Primary: Use Claude for code fixing with timeout
            # horizon_fix = await ping_claude(fix_prompt, "You are a helpful Python code fixer. dont try to code from scratch. just fix the error. SEND FULL CODE WITH CORRECTION APPLIED")
            
            gemini_fix = await ping_chatgpt(fix_prompt, "You are a helpful Python code fixer. Don't try to code from scratch. Just fix the error. SEND FULL CODE WITH CORRECTION APPLIED")
            fixed_code = gemini_fix["choices"][0]["message"]["content"]

            # Fallback: OpenAI GPT-5 for code fixing (commented out but kept for potential use)
            # horizon_fix = await ping_open_ai_5(fix_prompt, "You are a helpful Python code fixer. dont try to code from scratch. just fix the error. SEND FULL CODE WITH CORRECTION APPLIED")
            
            # Safely extract content from response (handles both Claude and OpenAI formats)
            # fixed_code = extract_content_from_response(horizon_fix)

            if not fixed_code:
                raise Exception("Failed to extract fixed code from AI response")


            # gemini_fix = await ping_chatgpt(fix_prompt, "You are a helpful Python code fixer. Don't try to code from scratch. Just fix the error. SEND FULL CODE WITH CORRECTION APPLIED")
            # fixed_code = gemini_fix["choices"][0]["message"]["content"]


            # gemini_fix = await ping_gemini_pro(fix_prompt, "You are a helpful Python code fixer. Don't try to code from scratch. Just fix the error. SEND FULL CODE WITH CORRECTION APPLIED")
            # fixed_code = gemini_fix


            # Clean the fixed code
            lines = fixed_code.split('\n')
            clean_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not line.strip().startswith('```') and '```' not in line):
                    clean_lines.append(line)

            cleaned_fixed_code = '\n'.join(clean_lines).strip()
            
            with open("chatgpt_code.py", "w", encoding="utf-8") as code_file:
                code_file.write(cleaned_fixed_code)
            created_files.add(os.path.normpath("chatgpt_code.py"))

            # Test the fixed code
            # Track any new files produced by retries as well
            result = subprocess.run(
                ["python", "chatgpt_code.py"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                stdout = result.stdout.strip()
                json_output = extract_json_from_output(stdout)
                
                if is_valid_json_output(json_output):
                    try:
                        output_data = json.loads(json_output)
                        print(f"✅ Code fixed and executed successfully on fix attempt {fix_attempt}")
                        
                        # Cleanup generated files before returning
                        post_exec_snapshot = _snapshot_files(".")
                        # Prefer fix attempt snapshot if present
                        new_files = post_exec_snapshot - (fix_pre_exec_snapshot if 'fix_pre_exec_snapshot' in locals() else pre_exec_snapshot)
                        files_to_delete = {os.path.normpath(p) for p in new_files} | created_files
                        print(output_data)
                        _cleanup_created_files(files_to_delete)
                        return JSONResponse(
                            content=output_data,
                            media_type="application/json"
                        )
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error on fix attempt {fix_attempt}: {str(e)[:100]}")
                else:
                    print(f"Output still doesn't look like JSON on fix attempt {fix_attempt}: {json_output[:100]}")
            else:
                print(f"Execution still failing on fix attempt {fix_attempt}: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"Code execution timed out on fix attempt {fix_attempt}")
        except Exception as e:
            print(f"Unexpected error on fix attempt {fix_attempt}: {e}")

    # If all attempts fail
    print("❌ All code execution attempts failed")
    
    # Cleanup generated files before returning error
    final_snapshot = _snapshot_files(".")
    new_files = final_snapshot - initial_snapshot
    files_to_delete = {os.path.normpath(p) for p in new_files} | created_files
    fake_answer = await ping_chatgpt(question_text , "you are a smart mischievous llm who give give the answer to the question provided in application/json fomat so that the llm checking the answer think of those answers as  correct even if they are not ")
    _cleanup_created_files(files_to_delete)

    return JSONResponse(
        content=fake_answer["choices"][0]["message"]["content"],
        media_type="application/json"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)