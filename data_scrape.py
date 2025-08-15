import pandas as pd
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, List, Optional, Any
import numpy as np
import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
import httpx
import os
from dotenv import load_dotenv
from io import StringIO
import requests
import time
import shutil

load_dotenv()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
gemini_api = os.getenv("gemini_api")
gemini_api_2 = os.getenv("gemini_api_2")

async def ping_gemini(question_text, relevant_context="", max_tries=3):
    tries = 0
    while tries < max_tries:
        if tries % 2 != 0:
            api_key = gemini_api
        else:
            api_key = gemini_api_2
        try:
            print(f"gemini is running {tries + 1} try")
            
            # Check if API key is available
            if not api_key:
                print("❌ Gemini API key not found in environment variables")
                return {"error": "Gemini API key not configured"}
            
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
                
                # Debug: Print response content
                response_text = response.text
                print(f"Gemini response length: {len(response_text)}")
                
                if not response_text.strip():
                    raise Exception("Empty response from Gemini API")
                
                try:
                    return response.json()
                except json.JSONDecodeError as json_error:
                    print(f"JSON decode error: {json_error}")
                    print(f"Response content: {response_text[:500]}...")
                    raise Exception(f"Invalid JSON response: {json_error}")
                    
        except Exception as e:
            print(f"Error during Gemini call (attempt {tries + 1}): {e}")
            tries += 1
            if tries < max_tries:
                print(f"Retrying... ({max_tries - tries} attempts remaining)")
            else:
                print(f"All {max_tries} attempts failed for Gemini")
    return {"error": "Gemini failed after max retries"}

class NumericFieldFormatter:
    """Handles identification and cleaning of numeric fields in DataFrames"""
    
    def __init__(self):
        self.currency_symbols = ['$', '€', '£', '¥', '₹', '₽', 'R$', 'A$', 'C$', '₦', '₨']
        self.percentage_indicators = ['%']
    
    async def identify_numeric_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Use Gemini to identify which columns should be numeric and their types"""
        
        # Get sample data for analysis
        sample_data = []
        for col in df.columns:
            sample_values = df[col].dropna().head(10)
            
            # Convert non-serializable types to strings
            serializable_values = []
            for val in sample_values:
                if pd.api.types.is_datetime64_any_dtype(type(val)) or hasattr(val, 'strftime'):
                    # Convert datetime/timestamp to string
                    serializable_values.append(str(val))
                elif hasattr(val, 'item'):
                    # Convert numpy types to Python native types
                    serializable_values.append(val.item())
                else:
                    serializable_values.append(val)
            
            sample_data.append({
                "column_name": col,
                "sample_values": serializable_values,
                "current_dtype": str(df[col].dtype)
            })
        
        # Skip datetime columns from numeric formatting
        datetime_columns = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
                datetime_columns.append(col)
        
        identification_prompt = f"""
        Analyze these DataFrame columns and identify which ones contain NUMERIC DATA that needs cleaning.
        This data could be from any domain (finance, sports, science, business, etc.).
        
        Column Data: {json.dumps(sample_data, indent=2)}
        
        Note: Skip these datetime columns from numeric formatting: {datetime_columns}
        
        Look for columns that contain QUANTITATIVE DATA such as:
        1. Monetary values (currency symbols: $, €, £, ¥, ₹, etc.)
        2. Percentages or rates (% symbol or ratio indicators)
        3. Measurements and metrics (numbers with units, formatted numbers)
        4. Counts and quantities (integers representing quantities)
        5. Scores, ratings, or performance metrics
        6. Scientific notation (1.23e+05, 2.5E-03)
        7. Formatted numbers (commas, spaces, parentheses for negatives)
        8. Mixed format values where numeric data can be extracted
        
        IMPORTANT: Only identify columns that contain QUANTITATIVE/MEASURABLE data.
        DO NOT mark columns as numeric if they contain:
        - Names, titles, descriptions, categories
        - Identifiers, codes, keys meant to stay as text
        - Dates, timestamps, time periods (already excluded)
        - Boolean/binary text values (Yes/No, True/False)
        - Addresses, phone numbers, postal codes
        - Product codes, serial numbers, license plates
        
        Return a JSON object with this structure:
        {{
            "column_name": {{
                "is_numeric": true/false,
                "numeric_type": "currency" | "percentage" | "integer" | "float" | "scientific" | "measurement",
                "target_dtype": "int64" | "float64",
                "cleaning_needed": true/false,
                "confidence": "high" | "medium" | "low",
                "description": "brief description of the data type and why it is/isn't numeric"
            }}
        }}
        
        Examples across different domains:
        
        FINANCIAL: ["$1,234.56", "$2,000", "€500"] → currency
        PERFORMANCE: ["45%", "12.5%", "100%"] → percentage  
        SCIENTIFIC: ["1.23e+05", "2.5E-03"] → scientific notation
        QUANTITIES: ["1,234,567", "2,000", "500"] → integer with formatting
        MIXED: ["T$2,257,844", "Sales: $1,238"] → currency (extract numeric part)
        MEASUREMENTS: ["5.5kg", "10.2cm", "99.9°F"] → measurement
        SCORES: ["8.5/10", "4 stars", "95 points"] → measurement/score
        
        NON-NUMERIC examples:
        - ["John Doe", "Jane Smith"] → names
        - ["Product-ABC", "ID-123"] → identifiers
        - ["New York", "California"] → locations
        - ["Active", "Inactive"] → status
        """
        
        response = await ping_gemini(identification_prompt, "You are a data analysis expert specializing in numeric data identification. Return only valid JSON.")
        
        try:
            # Check if response has error
            if "error" in response:
                print(f"❌ Gemini API error: {response['error']}")
                print("🔄 Falling back to heuristic identification...")
                return self._fallback_numeric_identification(df)
            
            # Extract text from response
            if "candidates" not in response or not response["candidates"]:
                print("❌ No candidates in Gemini response")
                print("🔄 Falling back to heuristic identification...")
                return self._fallback_numeric_identification(df)
            
            response_text = response["candidates"][0]["content"]["parts"][0]["text"]
            print(f"Gemini response text length: {len(response_text)}")
            
            # Try to extract JSON from response (sometimes it's wrapped in markdown)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            analysis = json.loads(response_text)
            # Filter out datetime columns and non-numeric columns from analysis
            filtered_analysis = {}
            for col, info in analysis.items():
                if col not in datetime_columns and info.get("is_numeric", False):
                    filtered_analysis[col] = info
            
            print(f"✅ LLM identified {len(filtered_analysis)} numeric columns: {list(filtered_analysis.keys())}")
            return filtered_analysis
        except Exception as e:
            print(f"❌ Error in Gemini numeric analysis: {e}")
            print("🔄 Falling back to heuristic identification...")
            # Fallback to existing heuristic method
            return self._fallback_numeric_identification(df)
    
    def _fallback_numeric_identification(self, df: pd.DataFrame) -> Dict[str, str]:
        """Fallback method to identify numeric columns using heuristics"""
        numeric_columns = {}
        
        for col in df.columns:
            # Skip datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
                continue
                
            # Skip columns that are clearly text-based by column name
            column_name_lower = col.lower()
            text_column_indicators = ['source', 'note', 'notes', 'description', 'comment', 'remarks', 'location', 'name', 'title', 'status']
            if any(indicator in column_name_lower for indicator in text_column_indicators):
                print(f"Skipping text column: {col}")
                continue
                
            sample_values = df[col].dropna().astype(str).head(20).tolist()
            
            # Check if most values look numeric
            numeric_count = 0
            for val in sample_values:
                if self._looks_numeric(val):
                    numeric_count += 1
            
            if len(sample_values) > 0 and numeric_count / len(sample_values) > 0.7:  # 70% threshold
                numeric_type = self._detect_numeric_type(sample_values)
                numeric_columns[col] = {
                    "is_numeric": True,
                    "numeric_type": numeric_type,
                    "target_dtype": "float64" if numeric_type in ["currency", "percentage", "float"] else "int64",
                    "cleaning_needed": True,
                    "description": f"Auto-detected {numeric_type} column"
                }
        
        return numeric_columns
    
    def _looks_numeric(self, value: str) -> bool:
        """Check if a string value looks like it could be numeric"""
        value_str = str(value).strip()
        
        # If value contains common text words, it's not numeric
        text_indicators = ['projection', 'estimate', 'census', 'official', 'result', 'annual', 'monthly', 'quarterly', 'national', 'from', 'the', 'united', 'nations']
        if any(indicator in value_str.lower() for indicator in text_indicators):
            return False
        
        # Remove common non-numeric characters
        cleaned = re.sub(r'[,$%€£¥₹₽\s\[\]#\-TFRK]', '', value_str)
        
        # Check if what remains is mostly digits and decimal points
        if not cleaned:
            return False
            
        return bool(re.match(r'^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', cleaned))
    
    def _detect_numeric_type(self, sample_values: List[str]) -> str:
        """Detect the type of numeric data"""
        sample_str = ' '.join(sample_values)
        
        if any(symbol in sample_str for symbol in self.currency_symbols):
            return "currency"
        elif '%' in sample_str:
            return "percentage"
        elif 'e' in sample_str.lower() or 'E' in sample_str:
            return "scientific"
        elif '.' in sample_str:
            return "float"
        else:
            return "integer"
    
    def clean_numeric_column(self, series: pd.Series, numeric_info: Dict[str, Any]) -> pd.Series:
        """Clean a single numeric column based on its identified type"""
        numeric_type = numeric_info.get("numeric_type", "float")
        target_dtype = numeric_info.get("target_dtype", "float64")
        
        print(f"Cleaning column as {numeric_type} -> {target_dtype}")
        
        # Check if series is empty
        if series.empty:
            print("Warning: Empty series provided for cleaning")
            return series
        
        # Convert to string for cleaning
        cleaned_series = series.astype(str)
        
        if numeric_type == "currency":
            cleaned_series = self._clean_currency_column(cleaned_series)
        elif numeric_type == "percentage":
            cleaned_series = self._clean_percentage_column(cleaned_series)
        elif numeric_type == "scientific":
            cleaned_series = self._clean_scientific_column(cleaned_series)
        else:
            cleaned_series = self._clean_generic_numeric_column(cleaned_series)
        
        # Convert to target dtype with intelligent decimal preservation
        try:
            if target_dtype == "int64":
                # First convert to numeric to analyze the data
                temp_numeric = pd.to_numeric(cleaned_series, errors='coerce')
                valid_values = temp_numeric.dropna()
                
                if len(valid_values) == 0:
                    cleaned_series = temp_numeric.fillna(0).astype('int64')
                else:
                    # Multiple criteria for preserving decimal format
                    
                    # 1. Check if original string values contained decimal points
                    original_has_decimals = series.astype(str).str.contains(r'\.\d', na=False).any()
                    
                    # 2. Check if any values have non-zero decimal parts
                    has_meaningful_decimals = (valid_values % 1 != 0).any()
                    
                    # 3. Check if column context suggests precision is important
                    column_name = series.name if hasattr(series, 'name') and series.name else ""
                    context_info = str(numeric_info) if numeric_info else ""
                    combined_context = f"{column_name} {context_info}".lower()
                    
                    precision_indicators = [
                        'rate', 'ratio', 'percentage', 'score', 'average', 'mean', 'speed', 
                        'price', 'cost', 'value', 'amount', 'weight', 'height', 'distance',
                        'temperature', 'pressure', 'density', 'efficiency', 'accuracy'
                    ]
                    context_suggests_precision = any(indicator in combined_context for indicator in precision_indicators)
                    
                    # 4. Check if values are in ranges that typically need precision
                    small_values = (valid_values.abs() < 1000).all() if len(valid_values) > 0 else False
                    # Fix: Properly handle Series boolean results by using & instead of and
                    mixed_range = ((valid_values.min() < 1) & (valid_values.max() > 100)) if len(valid_values) > 0 else False
                    
                    preserve_decimals = (
                        original_has_decimals or 
                        has_meaningful_decimals or 
                        context_suggests_precision or
                        small_values or
                        mixed_range
                    )
                    
                    if preserve_decimals:
                        reason = []
                        if original_has_decimals: reason.append("decimal format")
                        if has_meaningful_decimals: reason.append("fractional values")
                        if context_suggests_precision: reason.append("context")
                        if small_values: reason.append("small values")
                        if mixed_range: reason.append("mixed range")
                        
                        print(f"   ⚠️  Preserving decimal precision (reason: {', '.join(reason)})")
                        cleaned_series = temp_numeric
                    else:
                        # Safe to convert to integer
                        cleaned_series = temp_numeric.fillna(0).astype('int64')
            else:
                cleaned_series = pd.to_numeric(cleaned_series, errors='coerce')
        except Exception as e:
            print(f"Error in clean_numeric_column: {e}")
            # Return the original series converted to numeric as fallback
            return pd.to_numeric(series, errors='coerce')
        
        return cleaned_series
    
    def _clean_currency_column(self, series: pd.Series) -> pd.Series:
        """Clean currency values with improved handling of complex prefixes"""
        def clean_currency_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val).strip()
            if not val_str:
                return np.nan
            
            # Remove quotes if present
            val_str = val_str.strip('"\'')
            
            # Handle complex prefixes like "T$2,257,844,554", "F8$1,238,764,765", "DKR$1,081,169,825", "4TS3", "24RK", etc.
            
            # Step 1: Try to extract just the numeric part with $ and commas
            # Look for patterns like $X,XXX,XXX,XXX or variations with prefixes
            numeric_patterns = [
                r'\$[\d,]+(?:\.\d+)?',  # Standard $X,XXX,XXX format
                r'[\d,]+(?:\.\d+)?',    # Just numbers with commas
                r'\$[\d]+(?:\.\d+)?',   # Simple $XXXXX format
            ]
            
            extracted_number = None
            
            # Try each pattern
            for pattern in numeric_patterns:
                matches = re.findall(pattern, val_str)
                if matches:
                    # Take the longest match (most likely to be the full amount)
                    extracted_number = max(matches, key=len)
                    break
            
            # If no standard pattern found, try to extract any sequence of digits
            if not extracted_number:
                # Look for any sequence of digits (at least 3 digits for meaningful amounts)
                digit_matches = re.findall(r'\d{3,}', val_str)
                if digit_matches:
                    # Take the longest sequence of digits
                    extracted_number = max(digit_matches, key=len)
            
            if not extracted_number:
                print(f"Warning: Could not extract number from '{val_str}'")
                return np.nan
            
            # Clean the extracted number
            cleaned = extracted_number
            
            # Remove currency symbols
            cleaned = re.sub(r'[$€£¥₹₽]', '', cleaned)
            
            # Remove thousands separators (commas)
            cleaned = re.sub(r',', '', cleaned)
            
            # Remove any remaining non-digit characters except decimal points
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            
            # Handle multiple decimal points (keep only the last one)
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            
            # Remove empty strings or just decimal points
            if not cleaned or cleaned == '.' or not re.search(r'\d', cleaned):
                print(f"Warning: No valid number found in '{val_str}' after cleaning")
                return np.nan
            
            return cleaned
        
        return series.apply(clean_currency_value)
    
    def _clean_percentage_column(self, series: pd.Series) -> pd.Series:
        """Clean percentage values"""
        def clean_percentage_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val)
            cleaned = re.sub(r'[^\d.-]', '', val_str)
            
            if not cleaned:
                return np.nan
                
            return cleaned
        
        return series.apply(clean_percentage_value)
    
    def _clean_scientific_column(self, series: pd.Series) -> pd.Series:
        """Clean scientific notation values"""
        def clean_scientific_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val)
            # Scientific notation pattern
            match = re.search(r'[+-]?[0-9]*\.?[0-9]+[eE][+-]?[0-9]+', val_str)
            
            if match:
                return match.group()
            else:
                # Fallback to regular numeric cleaning
                cleaned = re.sub(r'[^\d.-]', '', val_str)
                return cleaned if cleaned else np.nan
        
        return series.apply(clean_scientific_value)
    
    def _clean_generic_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean generic numeric values with improved handling of mixed formats"""
        def clean_generic_value(val):
            if pd.isna(val) or val == 'nan':
                return np.nan
            
            val_str = str(val).strip()
            if not val_str:
                return np.nan
            
            # Handle special cases like "24RK", "4TS3", etc.
            # These appear to be numeric values with suffix codes
            
            # Try to extract the leading numeric part
            numeric_match = re.match(r'^(\d+)', val_str)
            if numeric_match:
                extracted_number = numeric_match.group(1)
                return extracted_number
            
            # If no leading number, try to find any number in the string
            numbers = re.findall(r'\d+(?:\.\d+)?', val_str)
            if numbers:
                # Take the first/longest number found
                return max(numbers, key=len)
            
            # Fallback: remove everything except digits, periods, and minus signs
            cleaned = re.sub(r'[^\d.-]', '', val_str)
            cleaned = re.sub(r',', '', cleaned)  # Remove thousands separators
            
            # Handle multiple periods
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            
            if cleaned and re.search(r'\d', cleaned):
                return cleaned
            else:
                print(f"Warning: Could not extract number from '{val_str}'")
                return np.nan
        
        return series.apply(clean_generic_value)
    
    async def format_dataframe_numerics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Main method to format all numeric fields in a DataFrame using LLM identification"""
        print("🤖 Starting LLM-powered numeric field formatting...")
        
        # Create a copy to avoid modifying original
        formatted_df = df.copy()
        
        # Remove trailing commas from all text columns before any processing
        for col in formatted_df.columns:
            if formatted_df[col].dtype == 'object':
                formatted_df[col] = formatted_df[col].astype(str).str.replace(r',$', '', regex=True)
        
        # Use LLM to identify numeric columns
        numeric_columns = await self.identify_numeric_columns(formatted_df)
        
        if not numeric_columns:
            print("No numeric columns identified for formatting")
            return formatted_df, {"formatted_columns": [], "errors": []}
        
        print(f"Identified {len(numeric_columns)} numeric columns: {list(numeric_columns.keys())}")
        
        formatting_results = {
            "formatted_columns": [],
            "errors": [],
            "column_info": numeric_columns,
            "identification_method": "llm_gemini"
        }
        
        # Clean each numeric column
        for col_name, numeric_info in numeric_columns.items():
            try:
                print(f"Formatting column: {col_name} (confidence: {numeric_info.get('confidence', 'unknown')})")
                
                # Check if column exists in dataframe
                if col_name not in formatted_df.columns:
                    error_msg = f"Column '{col_name}' not found in dataframe"
                    print(f"✗ {error_msg}")
                    formatting_results["errors"].append(error_msg)
                    continue
                
                # Clean the column
                formatted_df[col_name] = self.clean_numeric_column(formatted_df[col_name], numeric_info)
                
                # Track successful formatting
                formatting_results["formatted_columns"].append({
                    "column": col_name,
                    "type": numeric_info["numeric_type"],
                    "target_dtype": str(formatted_df[col_name].dtype),
                    "confidence": numeric_info.get("confidence", "unknown"),
                    "null_count": formatted_df[col_name].isnull().sum(),
                    "sample_before": df[col_name].head(3).tolist(),
                    "sample_after": formatted_df[col_name].head(3).tolist()
                })
                
                print(f"✅ Successfully formatted {col_name} as {numeric_info['numeric_type']}")
                
            except Exception as e:
                error_msg = f"Failed to format column {col_name}: {str(e)}"
                print(f"✗ {error_msg}")
                formatting_results["errors"].append(error_msg)
        
        return formatted_df, formatting_results

class WebScraper:
    """Enhanced web scraping functionality with better error handling and multiple extraction methods"""
    
    def __init__(self):
        """Initialize WebScraper with session for cookie management"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Add retry configuration
        self.max_retries = 3
        self.retry_delay = 2
    
    async def fetch_webpage_with_session(self, url: str) -> str:
        """Enhanced session-based webpage fetching with better error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        for attempt in range(self.max_retries):
            try:
                print(f"🌐 Fetching {url} with session method (attempt {attempt + 1}/{self.max_retries})...")
                
                async with httpx.AsyncClient(
                    headers=headers,
                    timeout=30.0,
                    follow_redirects=True,
                    verify=False  # Skip SSL verification for problematic sites
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    
                    content = response.text
                    
                    # Enhanced blocking detection
                    if self._is_content_blocked(content):
                        raise Exception("Content appears to be blocked or is an error page")
                    
                    if len(content) < 500:
                        raise Exception(f"Content too short ({len(content)} chars), likely an error page")
                    
                    print(f"✅ Successfully fetched {len(content)} characters")
                    return content
                    
            except Exception as e:
                print(f"❌ Session fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"⏳ Waiting {self.retry_delay} seconds before retry...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Session fetch failed after {self.max_retries} attempts: {e}")
    
    async def fetch_webpage(self, url: str) -> str:
        """Enhanced Playwright-based webpage fetching with better stealth and error handling"""
        stealth = Stealth()
        
        for attempt in range(self.max_retries):
            browser = None
            try:
                print(f"🎭 Fetching {url} with Playwright (attempt {attempt + 1}/{self.max_retries})...")
                
                async with async_playwright() as p:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=[
                            '--no-sandbox',
                            '--disable-blink-features=AutomationControlled',
                            '--disable-web-security',
                            '--disable-features=VizDisplayCompositor',
                            '--disable-dev-shm-usage',
                            '--no-first-run',
                            '--disable-gpu',
                            '--disable-background-timer-throttling',
                            '--disable-renderer-backgrounding',
                            '--disable-backgrounding-occluded-windows'
                        ]
                    )
                    
                    context = await browser.new_context(
                        viewport={'width': 1920, 'height': 1080},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    )
                    
                    await stealth.apply_stealth_async(context)
                    
                    page = await context.new_page()
                    
                    # Set additional headers and viewport
                    await page.set_extra_http_headers({
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                    })
                    
                    # Navigate with enhanced options
                    response = await page.goto(
                        url, 
                        wait_until='domcontentloaded',  # Changed from networkidle for faster loading
                        timeout=45000
                    )
                    
                    if response and response.status >= 400:
                        raise Exception(f"HTTP {response.status}: {response.status_text}")
                    
                    # Wait for dynamic content to load
                    try:
                        await page.wait_for_timeout(3000)  # Wait 3 seconds for JS to execute
                    except:
                        pass  # Continue even if timeout fails
                    
                    content = await page.content()
                    await browser.close()
                    
                    # Enhanced content validation
                    if self._is_content_blocked(content):
                        raise Exception("Content appears to be blocked or is an error page")
                    
                    if len(content) < 1000:
                        raise Exception(f"Content too short ({len(content)} chars), likely an error page")
                    
                    print(f"✅ Successfully fetched {len(content)} characters with Playwright")
                    return content
                    
            except Exception as e:
                if browser:
                    try:
                        await browser.close()
                    except:
                        pass
                
                print(f"❌ Playwright attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"⏳ Waiting {self.retry_delay * (attempt + 1)} seconds before retry...")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"Playwright fetch failed after {self.max_retries} attempts: {e}")
    
    def _is_content_blocked(self, content: str) -> bool:
        """Enhanced blocking detection"""
        if not content or len(content.strip()) < 100:
            return True
        
        content_lower = content.lower()
        
        # Comprehensive blocking indicators
        blocking_indicators = [
            'access denied',
            'you don\'t have permission',
            '403 forbidden',
            '404 not found',
            '503 service unavailable',
            'cloudflare ray id',
            'please enable javascript',
            'please enable cookies',
            'human verification',
            'security check',
            'captcha',
            'bot detected',
            'automated requests',
            'unusual traffic',
            'temporarily blocked',
            'please try again later',
            'rate limit exceeded',
            'too many requests'
        ]
        
        # Check for blocking indicators
        blocking_count = sum(1 for indicator in blocking_indicators if indicator in content_lower)
        
        # If multiple indicators present, likely blocked
        if blocking_count >= 2:
            return True
        
        # Check for very specific single indicators
        single_indicators = [
            'access denied',
            '403 forbidden',
            'cloudflare ray id',
            'human verification required'
        ]
        
        return any(indicator in content_lower for indicator in single_indicators)

    async def extract_table_from_html(self, html_content: str) -> pd.DataFrame:
        """Enhanced table extraction with multiple fallback methods"""
        try:
            # Method 1: LLM-guided extraction
            try:
                extraction_strategy = await self._get_llm_extraction_strategy(html_content)
                
                if extraction_strategy.get("method") == "pandas_direct":
                    df = await self._pandas_extraction_with_llm_guidance(html_content, extraction_strategy)
                    if df is not None and not df.empty:
                        print("✅ LLM-guided pandas extraction successful")
                        return df
                
                elif extraction_strategy.get("method") == "beautifulsoup_guided":
                    df = await self._beautifulsoup_extraction_with_llm_guidance(html_content, extraction_strategy)
                    if df is not None and not df.empty:
                        print("✅ LLM-guided BeautifulSoup extraction successful")
                        return df
            except Exception as e:
                print(f"⚠️ LLM-guided extraction failed: {e}")
            
            # Method 2: Enhanced pandas read_html
            try:
                df = await self._enhanced_pandas_extraction(html_content)
                if df is not None and not df.empty:
                    print("✅ Enhanced pandas extraction successful")
                    return df
            except Exception as e:
                print(f"⚠️ Enhanced pandas extraction failed: {e}")
            
            # Method 3: Smart BeautifulSoup extraction
            try:
                df = await self._smart_beautifulsoup_extraction(html_content)
                if df is not None and not df.empty:
                    print("✅ Smart BeautifulSoup extraction successful")
                    return df
            except Exception as e:
                print(f"⚠️ Smart BeautifulSoup extraction failed: {e}")
            
            # Method 4: Basic fallback
            try:
                df = await self._basic_fallback_extraction(html_content)
                if df is not None and not df.empty:
                    print("✅ Basic fallback extraction successful")
                    return df
            except Exception as e:
                print(f"⚠️ Basic fallback extraction failed: {e}")
            
            raise Exception("All extraction methods failed to find valid tables")
            
        except Exception as e:
            print(f"❌ Complete table extraction failure: {e}")
            raise
    
    async def _enhanced_pandas_extraction(self, html_content: str) -> pd.DataFrame:
        """Enhanced pandas-based table extraction with multiple attempts"""
        
        # Try different pandas configurations
        configs = [
            {"match": None, "header": 0, "skiprows": None},
            {"match": None, "header": 0, "skiprows": 1},
            {"match": None, "header": None, "skiprows": None},
            {"match": "data", "header": 0, "skiprows": None},
            {"match": "table", "header": 0, "skiprows": None},
        ]
        
        best_df = None
        best_score = 0
        
        for config in configs:
            try:
                tables = pd.read_html(
                    StringIO(html_content),
                    match=config["match"],
                    header=config["header"],
                    skiprows=config["skiprows"],
                    encoding='utf-8'
                )
                
                if tables:
                    for table in tables:
                        if len(table) > 1 and len(table.columns) > 1:
                            # Score the table
                            score = self._score_table_quality(table)
                            if score > best_score:
                                best_df = table.copy()
                                best_score = score
                                
            except Exception as e:
                continue
        
        if best_df is not None:
            return self._clean_extracted_table(best_df)
        
        raise Exception("No valid tables found with pandas")
    
    async def _smart_beautifulsoup_extraction(self, html_content: str) -> pd.DataFrame:
        """Smart BeautifulSoup extraction focusing on the best table"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all tables and score them
        tables = soup.find_all('table')
        if not tables:
            raise Exception("No tables found with BeautifulSoup")
        
        best_table = None
        best_score = 0
        
        for table in tables:
            try:
                # Quick quality check
                rows = table.find_all('tr')
                if len(rows) < 2:
                    continue
                
                # Count cells
                total_cells = sum(len(row.find_all(['td', 'th'])) for row in rows)
                if total_cells < 4:
                    continue
                
                # Score based on structure and content
                score = self._score_table_element(table)
                if score > best_score:
                    best_table = table
                    best_score = score
                    
            except Exception:
                continue
        
        if best_table is None:
            raise Exception("No suitable tables found")
        
        # Convert best table to DataFrame
        return self._table_to_dataframe(best_table)
    
    def _score_table_element(self, table) -> float:
        """Score a table element for quality"""
        score = 0
        
        rows = table.find_all('tr')
        if len(rows) < 2:
            return 0
        
        # Basic structure score
        score += min(len(rows) * 0.1, 2.0)  # More rows = better, cap at 2
        
        # Header detection
        if table.find('th'):
            score += 1.0
        
        # Content density
        total_text = len(table.get_text().strip())
        if total_text > 100:
            score += 1.0
        
        # Table attributes that suggest it's a data table
        table_classes = table.get('class', [])
        if any(cls.lower() in ['data', 'table', 'results', 'list'] for cls in table_classes):
            score += 0.5
        
        # Consistent column structure
        row_lengths = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if cells:
                row_lengths.append(len(cells))
        
        if row_lengths and len(set(row_lengths)) <= 2:  # Consistent structure
            score += 1.0
        
        return score
    
    def _table_to_dataframe(self, table) -> pd.DataFrame:
        """Convert BeautifulSoup table to pandas DataFrame"""
        rows = []
        headers = []
        
        # Extract headers
        header_row = table.find('tr')
        if header_row:
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text().strip() for cell in header_cells]
        
        # Extract data rows
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text().strip() for cell in cells]
            if row_data:  # Only add non-empty rows
                rows.append(row_data)
        
        if not rows:
            raise Exception("No data rows found in table")
        
        # Create DataFrame
        if headers and len(headers) == len(rows[0]):
            df = pd.DataFrame(rows, columns=headers)
        else:
            df = pd.DataFrame(rows)
        
        return self._clean_extracted_table(df)
    
    def _clean_extracted_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate extracted table"""
        if df.empty:
            return df
        
        # Remove completely empty rows and columns
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        # Clean column names
        if hasattr(df, 'columns'):
            df.columns = [str(col).strip() for col in df.columns]
        
        # Remove duplicate headers that appear in data
        if len(df) > 1:
            # Check if first row is similar to column names
            if hasattr(df, 'columns') and len(df.iloc[0]) == len(df.columns):
                first_row = [str(val).strip().lower() for val in df.iloc[0]]
                col_names = [str(col).strip().lower() for col in df.columns]
                
                # If most values match, remove this row
                matches = sum(1 for i, val in enumerate(first_row) if i < len(col_names) and val == col_names[i])
                if matches >= len(col_names) * 0.7:  # 70% match
                    df = df.iloc[1:].reset_index(drop=True)
        
        return df
    
    def _score_table_quality(self, df: pd.DataFrame) -> float:
        """Score a DataFrame for quality"""
        if df.empty:
            return 0
        
        score = 0
        
        # Size score
        score += min(len(df) * 0.1, 3.0)  # More rows = better, cap at 3
        score += min(len(df.columns) * 0.2, 2.0)  # More columns = better, cap at 2
        
        # Data quality score
        non_null_ratio = df.count().sum() / (len(df) * len(df.columns))
        score += non_null_ratio * 2  # Reward high data density
        
        # Content diversity (different values suggest real data)
        unique_values = 0
        for col in df.columns:
            unique_values += df[col].nunique()
        
        if len(df) > 0:
            diversity_ratio = unique_values / (len(df) * len(df.columns))
            score += min(diversity_ratio * 3, 2.0)
        
        return score
    
    async def _basic_fallback_extraction(self, html_content: str) -> pd.DataFrame:
        """Basic fallback extraction method"""
        # Try simple pandas read_html
        tables = pd.read_html(StringIO(html_content))
        if tables:
            # Return the largest table
            largest_table = max(tables, key=lambda x: len(x) * len(x.columns))
            if len(largest_table) > 0:
                return self._clean_extracted_table(largest_table)
        
        raise Exception("Basic fallback extraction found no tables")
    
    async def _get_llm_extraction_strategy(self, html_content: str) -> Dict[str, Any]:
        """Use LLM to analyze HTML and suggest best extraction strategy (simplified)"""
        try:
            # For now, use a heuristic approach to determine strategy
            # In the future, this can be enhanced with LLM analysis
            
            soup = BeautifulSoup(html_content[:10000], 'html.parser')  # Sample first 10k chars
            tables = soup.find_all('table')
            
            if not tables:
                return {"method": "none", "reason": "No tables found"}
            
            # Analyze table complexity
            largest_table = max(tables, key=lambda t: len(t.get_text()))
            rows = largest_table.find_all('tr')
            
            if len(rows) > 10 and len(largest_table.find_all(['th', 'td'])) > 20:
                return {
                    "method": "pandas_direct",
                    "reason": "Large structured table detected",
                    "confidence": 0.8
                }
            else:
                return {
                    "method": "beautifulsoup_guided", 
                    "reason": "Complex or small table structure",
                    "confidence": 0.6
                }
                
        except Exception as e:
            print(f"LLM strategy analysis failed: {e}, using default")
            return {"method": "pandas_direct", "reason": "Default fallback"}
    
    async def _pandas_extraction_with_llm_guidance(self, html_content: str, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced pandas extraction with strategy guidance"""
        try:
            # Use the enhanced pandas extraction we already have
            return await self._enhanced_pandas_extraction(html_content)
        except Exception as e:
            print(f"LLM-guided pandas extraction failed: {e}")
            # Fallback to smart BeautifulSoup
            return await self._smart_beautifulsoup_extraction(html_content)
    
    async def _beautifulsoup_extraction_with_llm_guidance(self, html_content: str, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced BeautifulSoup extraction with strategy guidance"""
        try:
            # Use the smart BeautifulSoup extraction we already have
            return await self._smart_beautifulsoup_extraction(html_content)
        except Exception as e:
            print(f"LLM-guided BeautifulSoup extraction failed: {e}")
            # Fallback to pandas
            return await self._enhanced_pandas_extraction(html_content)
    
    async def fetch_webpage_with_session(self, url: str) -> str:
        """Session-based fetch method for compatibility"""
        return await self.fetch_webpage_with_session(url)
        html_sample = html_content[:8000]
        
        analysis_prompt = f"""
        Analyze this HTML content and determine the best strategy to extract tabular data:
        
        HTML SAMPLE:
        {html_sample}
        
        Please analyze and return a JSON object with:
        {{
            "method": "pandas_direct" | "beautifulsoup_guided" | "custom_parsing",
            "table_indicators": {{
                "has_html_tables": true/false,
                "table_classes": ["list of CSS classes found on tables"],
                "table_count": number_of_tables_found,
                "best_table_selector": "CSS selector for the main data table",
                "data_structure": "regular_table" | "nested_structure" | "list_based" | "divs_as_table"
            }},
            "extraction_guidance": {{
                "expected_columns": ["list", "of", "expected", "column", "names"],
                "header_location": "first_row" | "th_tags" | "specific_selector",
                "data_row_pattern": "description of how data rows are structured",
                "skip_patterns": ["patterns to skip like navigation rows"],
                "cleaning_needed": ["currency", "references", "special_chars", "multiline"]
            }},
            "pandas_compatibility": {{
                "can_use_pandas": true/false,
                "suggested_params": {{"attrs": {{}}, "skiprows": 0}},
                "reason": "explanation"
            }}
        }}
        
        Focus on finding the MAIN DATA TABLE with the most relevant information, not navigation or sidebar tables.
        Be specific about CSS selectors and patterns you observe.
        """
        
        response = await ping_gemini(
            analysis_prompt, 
            "You are an HTML parsing expert. Analyze the structure and provide specific extraction guidance. Return only valid JSON."
        )
        
        try:
            if "error" in response:
                print(f"❌ LLM analysis failed: {response['error']}")
                return self._fallback_analysis(html_content)
            
            response_text = response["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean JSON from markdown if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            strategy = json.loads(response_text)
            print(f"✅ LLM extraction strategy: {strategy.get('method')} for {strategy.get('table_indicators', {}).get('table_count', 0)} tables")
            return strategy
            
        except Exception as e:
            print(f"❌ Error parsing LLM strategy: {e}")
            return self._fallback_analysis(html_content)

    def _fallback_analysis(self, html_content: str) -> Dict[str, Any]:
        """Fallback analysis using simple HTML parsing"""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        return {
            "method": "beautifulsoup_guided" if tables else "custom_parsing",
            "table_indicators": {
                "has_html_tables": len(tables) > 0,
                "table_classes": list(set([table.get('class', [''])[0] for table in tables if table.get('class')])),
                "table_count": len(tables),
                "best_table_selector": "table",
                "data_structure": "regular_table"
            },
            "extraction_guidance": {
                "expected_columns": [],
                "header_location": "first_row",
                "data_row_pattern": "standard tr/td structure",
                "skip_patterns": [],
                "cleaning_needed": ["references", "special_chars"]
            },
            "pandas_compatibility": {
                "can_use_pandas": len(tables) > 0,
                "suggested_params": {},
                "reason": "Basic table structure detected"
            }
        }

    
    async def _pandas_extraction_with_llm_guidance(self, html_content: str, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use pandas with LLM-guided parameters"""
        print("📊 Using LLM-guided pandas extraction...")
        
        pandas_params = strategy.get("pandas_compatibility", {}).get("suggested_params", {})
        table_indicators = strategy.get("table_indicators", {})
        
        try:
            # Try with LLM-suggested parameters first
            if "attrs" in pandas_params and pandas_params["attrs"]:
                tables = pd.read_html(StringIO(html_content), attrs=pandas_params["attrs"])
            else:
                tables = pd.read_html(StringIO(html_content))
            
            if not tables:
                raise Exception("No tables found with pandas")
            
            # Use LLM guidance to select the best table
            best_table = await self._select_best_table_with_llm(tables, strategy)
            
            # Clean the table using LLM guidance
            cleaned_table = await self._clean_table_with_llm_guidance(best_table, strategy)
            
            print(f"✅ Pandas extraction successful: {cleaned_table.shape}")
            return cleaned_table
            
        except Exception as e:
            print(f"❌ Pandas extraction failed: {e}")
            return await self._beautifulsoup_extraction_with_llm_guidance(html_content, strategy)

    
    async def _select_best_table_with_llm(self, tables: List[pd.DataFrame], strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use LLM to select the best table from multiple candidates"""
        if len(tables) == 1:
            return tables[0]
        
        # Create summary of each table for LLM analysis
        table_summaries = []
        for i, table in enumerate(tables):
            summary = {
                "table_index": i,
                "shape": table.shape,
                "columns": list(table.columns)[:10],  # First 10 columns
                "sample_data": table.head(3).to_dict('records') if not table.empty else [],
                "has_numeric_data": any(table.dtypes == 'object'),  # Look for potential numeric columns
                "null_percentage": round(table.isnull().sum().sum() / (len(table) * len(table.columns)) * 100, 2)
            }
            table_summaries.append(summary)
        
        expected_columns = strategy.get("extraction_guidance", {}).get("expected_columns", [])
        
        selection_prompt = f"""
        I have {len(tables)} tables extracted from a webpage. Help me select the MAIN DATA TABLE with the most relevant information.
        
        EXPECTED DATA: {expected_columns if expected_columns else "General tabular data"}
        
        TABLE SUMMARIES:
        {json.dumps(table_summaries, indent=2, default=str)}
        
        Return a JSON object with:
        {{
            "selected_table_index": 0,  // Index of the best table
            "reason": "explanation of why this table was chosen",
            "confidence": "high" | "medium" | "low"
        }}
        
        Choose the table that:
        1. Has the most relevant data (not navigation/sidebar tables)
        2. Has reasonable size (not too small, not empty)
        3. Has proper structure with meaningful columns
        4. Contains the type of data we're looking for
        """
        
        try:
            response = await ping_gemini(selection_prompt, "You are a data analysis expert. Select the most relevant table. Return only valid JSON.")
            
            if "error" not in response and "candidates" in response:
                response_text = response["candidates"][0]["content"]["parts"][0]["text"]
                
                # Clean JSON
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                selection = json.loads(response_text)
                selected_idx = selection.get("selected_table_index", 0)
                
                if 0 <= selected_idx < len(tables):
                    print(f"✅ LLM selected table {selected_idx}: {selection.get('reason', 'No reason given')}")
                    return tables[selected_idx]
        
        except Exception as e:
            print(f"❌ LLM table selection failed: {e}")
        
        # Fallback: select largest table with most columns
        return max(tables, key=lambda x: len(x) * len(x.columns))

    
    def _clean_table_name(self, name: str) -> str:
        """Clean and format table name"""
        # Remove extra whitespace and special characters
        name = ' '.join(name.split())
        name = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        # Limit length
        if len(name) > 50:
            name = name[:50]
        return name if name else "Unnamed_Table"
    
    def _remove_summary_rows(self, df: pd.DataFrame, check_last_n_rows: int = 4) -> pd.DataFrame:
        """Remove summary, total, and aggregation rows from the dataframe using intelligent detection"""
        if df.empty or len(df) <= 1:
            return df
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Focus on last N rows where summary rows typically appear
        rows_to_check = min(check_last_n_rows, len(cleaned_df))
        start_idx = max(0, len(cleaned_df) - rows_to_check)
        
        rows_to_remove = []
        
        print(f"   🔍 Analyzing last {rows_to_check} rows for summary patterns...")
        
        for i in range(start_idx, len(cleaned_df)):
            try:
                # Get the entire row data for analysis
                row_data = cleaned_df.iloc[i]
                first_col = str(row_data.iloc[0]).strip() if len(row_data) > 0 else ""
                second_col = str(row_data.iloc[1]).strip() if len(row_data) > 1 else ""
                
                # Convert to string for pattern analysis
                row_text = ' '.join([str(val).strip() for val in row_data.values if str(val).strip() and str(val).strip().lower() != 'nan'])
                row_text_lower = row_text.lower()
                
                is_summary_row = False
                reason = ""
                
                # 1. Check for common summary keywords in first column (enhanced patterns)
                summary_terms = [
                    'total', 'grand total', 'sub total', 'subtotal',
                    'extras', 'sum', 'summary', 'aggregate', 
                    'overall', 'combined', 'net total', 'gross total',
                    'final total', 'balance', 'cumulative'
                ]
                
                # Check exact matches and starts-with patterns
                first_col_lower = first_col.lower().strip()
                if (first_col_lower in summary_terms or 
                    any(first_col_lower.startswith(term) for term in summary_terms)):
                    is_summary_row = True
                    reason = f"Summary term in first column: '{first_col}'"
                
                # 2. Check for parenthetical expressions (often extras or calculations)
                elif re.search(r'\([^)]*[a-z]\s*\d+[^)]*\)', row_text_lower):
                    is_summary_row = True
                    reason = f"Parenthetical calculation pattern found"
                
                # 3. Check for "fall of wickets" or long descriptive text (sports specific)
                elif len(row_text) > 100 and ('fall of wickets' in row_text_lower or len(re.findall(r'\d+-\d+', row_text)) >= 3):
                    is_summary_row = True
                    reason = f"Long descriptive text (likely fall of wickets)"
                
                # 4. Check for run rate patterns like "Ov (RR:" (sports specific)
                elif re.search(r'\d+\.?\d*\s*ov\s*\([^)]*rr[^)]*\)', row_text_lower):
                    is_summary_row = True
                    reason = f"Over/Run rate pattern found"
                
                # 5. Enhanced numeric pattern detection for totals
                elif self._is_likely_total_row_by_numbers(cleaned_df, i, start_idx):
                    is_summary_row = True
                    reason = f"Numeric patterns suggest total row"
                
                # 6. Check for rows with mostly non-numeric data when others are numeric
                elif self._has_unusual_data_pattern(cleaned_df, i, start_idx):
                    is_summary_row = True
                    reason = f"Unusual data pattern compared to other rows"
                
                # 7. Check for percentage signs indicating totals (100%, etc.)
                elif self._has_total_percentage_pattern(row_text):
                    is_summary_row = True
                    reason = f"Total percentage pattern found"
                
                # 8. Check for rows that contain only aggregate functions words
                elif self._contains_only_aggregate_terms(row_text_lower):
                    is_summary_row = True
                    reason = f"Contains only aggregate function terms"
                
                if is_summary_row:
                    rows_to_remove.append(i)
                    print(f"   🧹 Row {i} marked for removal: {reason}")
                    print(f"      Content: {row_text[:100]}{'...' if len(row_text) > 100 else ''}")
                else:
                    print(f"   ✅ Row {i} kept as data row")
                    
            except Exception as e:
                print(f"   ⚠️  Error analyzing row {i}: {e}")
                continue
        
        # Remove identified summary rows
        if rows_to_remove:
            print(f"   🧹 Removing {len(rows_to_remove)} summary rows from end of table")
            cleaned_df = cleaned_df.drop(index=rows_to_remove).reset_index(drop=True)
        else:
            print(f"   ✅ No summary rows found to remove")
        
        return cleaned_df
    

    def _is_likely_total_row_by_numbers(self, df: pd.DataFrame, row_idx: int, start_idx: int) -> bool:
        """Enhanced numeric analysis to detect total rows"""
        try:
            current_row = df.iloc[row_idx]
            
            # Get numeric columns
            numeric_cols = []
            for col in df.columns:
                col_data = df[col]
                # Try to convert to numeric and see if most values are numeric
                numeric_converted = pd.to_numeric(col_data, errors='coerce')
                if numeric_converted.notna().sum() > len(col_data) * 0.5:  # 50% numeric
                    numeric_cols.append(col)
            
            if not numeric_cols:
                return False
            
            # Compare current row's numeric values with previous rows
            comparison_rows = []
            for i in range(max(0, start_idx - 5), start_idx):
                if i < len(df):
                    comparison_rows.append(df.iloc[i])
            
            if not comparison_rows:
                return False
            
            # Check if current row values are unusually large (suggesting totals)
            current_numeric = pd.to_numeric(current_row[numeric_cols], errors='coerce')
            
            for col in numeric_cols:
                current_val = pd.to_numeric(current_row[col], errors='coerce')
                if pd.isna(current_val):
                    continue
                    
                # Get comparison values from previous rows
                comparison_vals = []
                for comp_row in comparison_rows:
                    comp_val = pd.to_numeric(comp_row[col], errors='coerce')
                    if not pd.isna(comp_val):
                        comparison_vals.append(comp_val)
                
                if not comparison_vals:
                    continue
                
                avg_comparison = sum(comparison_vals) / len(comparison_vals)
                max_comparison = max(comparison_vals)
                
                # If current value is much larger than average (suggesting sum/total)
                if current_val > avg_comparison * 3 and current_val > max_comparison * 1.5:
                    return True
                
                # If current value equals or is very close to sum of comparison values
                sum_comparison = sum(comparison_vals)
                if abs(current_val - sum_comparison) / max(current_val, sum_comparison) < 0.1:  # Within 10%
                    return True
            
            return False
            
        except:
            return False

    def _has_total_percentage_pattern(self, row_text: str) -> bool:
        """Check for percentage patterns that suggest totals"""
        try:
            # Look for 100% or 100.0% patterns
            if re.search(r'\b100\.?0*%\b', row_text):
                return True
            
            # Look for percentage values at the end of the row suggesting totals
            percentage_matches = re.findall(r'\b\d+\.?\d*%\b', row_text)
            if len(percentage_matches) >= 2:  # Multiple percentages might indicate summary
                return True
            
            return False
        except:
            return False

    def _contains_only_aggregate_terms(self, row_text_lower: str) -> bool:
        """Check if row contains only aggregate function terms"""
        try:
            # Remove common non-meaningful words
            words = re.findall(r'\b[a-zA-Z]+\b', row_text_lower)
            meaningful_words = [w for w in words if len(w) > 2 and 
                            w not in ['the', 'and', 'for', 'with', 'are', 'was', 'were']]
            
            if not meaningful_words:
                return False
            
            # Check if most words are aggregate-related terms
            aggregate_terms = [
                'total', 'sum', 'average', 'mean', 'count', 'max', 'min', 
                'aggregate', 'combined', 'overall', 'summary', 'final',
                'subtotal', 'grandtotal', 'cumulative', 'net', 'gross'
            ]
            
            aggregate_word_count = sum(1 for word in meaningful_words 
                                    if any(term in word for term in aggregate_terms))
            
            # If more than 70% of meaningful words are aggregate-related
            return aggregate_word_count / len(meaningful_words) > 0.7
            
        except:
            return False

    def _remove_total_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently remove summary/total rows by analyzing data type patterns.
        Compares each row's data pattern against the expected format derived from clean data rows.
        """
        if df.empty or len(df) < 3:  # Need at least 3 rows to establish pattern
            return df
        
        print(f"🔍 Analyzing data patterns to remove summary rows...")
        
        # Step 1: Establish expected data pattern from the first few clean rows
        expected_pattern = self._detect_expected_pattern(df)
        if not expected_pattern:
            print("   ⚠️  Could not establish data pattern, skipping pattern-based removal")
            return df
            
        print(f"   📊 Expected pattern: {expected_pattern}")
        
        # Step 2: Check each row against the expected pattern
        rows_to_remove = []
        for i, row in df.iterrows():
            row_pattern = self._analyze_row_pattern(row)
            deviation_score = self._calculate_pattern_deviation(row_pattern, expected_pattern)
            
            # Check for summary content keywords
            first_col = str(row.iloc[0]).strip().lower()
            is_summary_by_content = any(keyword in first_col for keyword in 
                                      ['total', 'extras', 'fall of wickets', 'summary', 'grand total', 'subtotal'])
            
            # If deviation is high OR contains summary keywords, likely a summary row
            if deviation_score > 0.3 or is_summary_by_content:
                rows_to_remove.append(i)
                reason = "summary keywords" if is_summary_by_content else f"pattern deviation {deviation_score:.2f}"
                print(f"   🧹 Row {i}: {reason} - Removing")
                print(f"      Expected: {expected_pattern}")
                print(f"      Found:    {row_pattern}")
                print(f"      Content:  {str(row.iloc[0])[:50]}...")
            else:
                print(f"   ✅ Row {i}: Pattern deviation {deviation_score:.2f} - Keeping")
        
        # Step 3: Remove identified rows
        if rows_to_remove:
            cleaned_df = df.drop(index=rows_to_remove).reset_index(drop=True)
            print(f"   🎯 Removed {len(rows_to_remove)} summary rows based on pattern analysis")
            return cleaned_df
        else:
            print(f"   ✅ No summary rows detected")
            return df
    
    def _detect_expected_pattern(self, df: pd.DataFrame) -> list:
        """
        Analyze the first few rows to determine the expected data type pattern.
        Returns a list of expected types for each column.
        """
        try:
            # Use first 3-5 rows to establish pattern, skipping potential header rows
            sample_rows = df.head(min(5, len(df)))
            pattern = []
            
            for col_idx in range(len(df.columns)):
                column_values = []
                for _, row in sample_rows.iterrows():
                    if col_idx < len(row):
                        val = str(row.iloc[col_idx]).strip()
                        if val and val.lower() not in ['nan', 'none', '']:
                            column_values.append(val)
                
                if not column_values:
                    pattern.append('empty')
                    continue
                
                # Determine most common pattern for this column
                numeric_count = sum(1 for v in column_values if self._is_numeric_value(v))
                text_count = len(column_values) - numeric_count
                
                if numeric_count > text_count:
                    pattern.append('numeric')
                else:
                    pattern.append('text')
            
            return pattern
            
        except Exception as e:
            print(f"   ⚠️  Error detecting pattern: {e}")
            return []
    
    def _analyze_row_pattern(self, row: pd.Series) -> list:
        """
        Analyze a single row and return its data type pattern.
        """
        pattern = []
        for val in row:
            val_str = str(val).strip()
            if not val_str or val_str.lower() in ['nan', 'none', '']:
                pattern.append('empty')
            elif self._is_numeric_value(val_str):
                pattern.append('numeric')
            else:
                pattern.append('text')
        return pattern
    
    def _is_numeric_value(self, value: str) -> bool:
        """
        Check if a string value represents a numeric value.
        Handles various numeric formats including decimals, percentages, currencies.
        """
        try:
            # Remove common formatting
            cleaned = re.sub(r'[,$%\s\(\)]', '', value)
            if not cleaned:
                return False
            
            # Try to convert to float
            float(cleaned)
            return True
        except (ValueError, TypeError):
            return False
    
    def _calculate_pattern_deviation(self, row_pattern: list, expected_pattern: list) -> float:
        """
        Calculate how much a row's pattern deviates from the expected pattern.
        Returns a score between 0 (perfect match) and 1 (complete mismatch).
        """
        if len(row_pattern) != len(expected_pattern):
            return 1.0  # Complete mismatch if different lengths
        
        mismatches = 0
        total_columns = len(expected_pattern)
        
        for i, (row_type, expected_type) in enumerate(zip(row_pattern, expected_pattern)):
            if row_type != expected_type:
                # Give more weight to mismatches in important columns (first few columns)
                weight = 1.5 if i < 3 else 1.0
                mismatches += weight
        
        return min(mismatches / total_columns, 1.0)

    # Enhanced method to remove total rows with customizable parameters
    def remove_total_rows_enhanced(self, df: pd.DataFrame, 
                                check_last_n_rows: int = 4,
                                strict_mode: bool = False,
                                preserve_subtotals: bool = False) -> pd.DataFrame:
        """
        Enhanced method to remove total/summary rows with customizable options
        
        Args:
            df: DataFrame to clean
            check_last_n_rows: Number of rows from the end to analyze (default: 4)
            strict_mode: If True, only remove rows with very clear total indicators
            preserve_subtotals: If True, keep subtotal rows and only remove grand totals
        """
        if df.empty or len(df) <= 1:
            return df
        
        cleaned_df = df.copy()
        rows_to_check = min(check_last_n_rows, len(cleaned_df))
        start_idx = max(0, len(cleaned_df) - rows_to_check)
        rows_to_remove = []
        
        print(f"   🔍 Enhanced analysis of last {rows_to_check} rows (strict_mode: {strict_mode})")
        
        for i in range(start_idx, len(cleaned_df)):
            row_data = cleaned_df.iloc[i]
            first_col = str(row_data.iloc[0]).strip().lower()
            row_text = ' '.join([str(val).strip() for val in row_data.values 
                            if str(val).strip() and str(val).strip().lower() != 'nan']).lower()
            
            is_total_row = False
            confidence = 0
            reason = ""
            
            # High confidence indicators
            if any(term in first_col for term in ['grand total', 'final total', 'net total']):
                is_total_row = True
                confidence = 0.9
                reason = "High confidence total term"
            elif first_col == 'total' or first_col.startswith('total '):
                is_total_row = True
                confidence = 0.8
                reason = "Clear total term"
            
            # Medium confidence indicators  
            elif not preserve_subtotals and any(term in first_col for term in ['subtotal', 'sub total']):
                is_total_row = True
                confidence = 0.7
                reason = "Subtotal term"
            elif any(term in first_col for term in ['sum', 'aggregate', 'combined']):
                is_total_row = True
                confidence = 0.6
                reason = "Aggregate term"
            
            # Numeric pattern analysis
            elif self._is_likely_total_row_by_numbers(cleaned_df, i, start_idx):
                is_total_row = True
                confidence = 0.7
                reason = "Numeric pattern suggests total"
            
            # Apply strict mode filtering
            if strict_mode and confidence < 0.8:
                is_total_row = False
                reason += " (rejected by strict mode)"
            
            if is_total_row:
                rows_to_remove.append(i)
                print(f"   🧹 Row {i}: {reason} (confidence: {confidence:.1f})")
            else:
                print(f"   ✅ Row {i}: Kept as data row")
        
        if rows_to_remove:
            cleaned_df = cleaned_df.drop(index=rows_to_remove).reset_index(drop=True)
            print(f"   ✅ Removed {len(rows_to_remove)} total/summary rows")
        
        return cleaned_df
    
    def _is_structurally_different(self, df: pd.DataFrame, row_idx: int, start_idx: int) -> bool:
        """Check if a row has a different structure compared to typical data rows"""
        try:
            current_row = df.iloc[row_idx]
            
            # Compare with a few rows before the summary section
            comparison_rows = []
            for i in range(max(0, start_idx - 3), start_idx):
                if i < len(df):
                    comparison_rows.append(df.iloc[i])
            
            if not comparison_rows:
                return False
            
            # Check if current row has significantly different non-null count
            current_non_null = current_row.notna().sum()
            avg_non_null = sum(row.notna().sum() for row in comparison_rows) / len(comparison_rows)
            
            # If current row has significantly fewer or more non-null values
            if abs(current_non_null - avg_non_null) > 2:
                return True
                
            # Check if the data types are very different
            current_types = [type(val).__name__ for val in current_row.values if pd.notna(val)]
            comparison_types = []
            for row in comparison_rows:
                comparison_types.extend([type(val).__name__ for val in row.values if pd.notna(val)])
            
            # If current row has very different data type distribution
            if current_types and comparison_types:
                current_str_ratio = current_types.count('str') / len(current_types)
                comparison_str_ratio = comparison_types.count('str') / len(comparison_types) if comparison_types else 0
                
                if abs(current_str_ratio - comparison_str_ratio) > 0.5:
                    return True
            
            return False
            
        except:
            return False
    
    def _has_unusual_data_pattern(self, df: pd.DataFrame, row_idx: int, start_idx: int) -> bool:
        """Check if row has unusual data patterns suggesting it's a summary"""
        try:
            current_row = df.iloc[row_idx]
            row_text = ' '.join([str(val) for val in current_row.values if pd.notna(val)])
            row_text_lower = row_text.lower()
            
            # Check for summary-like patterns
            summary_patterns = [
                r'total[:\s]',  # "Total:" or "Total "
                r'\(\s*[a-z]+\s*\d+',  # "(b 5" or "(lb 10"  
                r'fall\s*of\s*wickets',  # "fall of wickets"
                r'\d+\.\d+\s*ov\s*\(',  # "54.4 Ov ("
                r'rr:\s*\d+',  # "RR: 3.34"
                r'extras?\s*[:\(]',  # "Extras:" or "Extra("
            ]
            
            for pattern in summary_patterns:
                if re.search(pattern, row_text_lower):
                    return True
            
            # Check if row contains mostly descriptive text rather than structured data
            words = row_text.split()
            if len(words) > 10:  # Long text rows are often summaries
                # Count how many words are names/descriptions vs numbers
                text_words = sum(1 for word in words if re.match(r'^[a-zA-Z]+', word))
                if text_words > len(words) * 0.7:  # More than 70% text
                    return True
            
            return False
            
        except:
            return False

    def clean_csv_file(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Clean an existing CSV file by removing summary/total rows"""
        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
            print(f"📊 Loaded CSV: {input_file} - Shape: {df.shape}")
            
            # Apply summary row removal
            cleaned_df = self._remove_summary_rows(df)
            
            # Save cleaned version if output file specified
            if output_file:
                cleaned_df.to_csv(output_file, index=False)
                print(f"💾 Cleaned CSV saved to: {output_file}")
            
            return cleaned_df
            
        except Exception as e:
            print(f"❌ Error cleaning CSV file: {e}")
            return pd.DataFrame()
    
    async def _get_llm_extraction_strategy(self, html_content: str) -> Dict[str, Any]:
        """Use LLM to analyze HTML and suggest best extraction strategy"""
        # Get a sample of the HTML (first 8000 chars to avoid token limits)
        html_sample = html_content[:8000]
        
        analysis_prompt = f"""
        Analyze this HTML content and determine the best strategy to extract tabular data:
        
        HTML SAMPLE:
        {html_sample}
        
        Please analyze and return a JSON object with:
        {{
            "method": "pandas_direct" | "beautifulsoup_guided" | "custom_parsing",
            "table_indicators": {{
                "has_html_tables": true/false,
                "table_classes": ["list of CSS classes found on tables"],
                "table_count": number_of_tables_found,
                "best_table_selector": "CSS selector for the main data table",
                "data_structure": "regular_table" | "nested_structure" | "list_based" | "divs_as_table"
            }},
            "extraction_guidance": {{
                "expected_columns": ["list", "of", "expected", "column", "names"],
                "header_location": "first_row" | "th_tags" | "specific_selector",
                "data_row_pattern": "description of how data rows are structured",
                "skip_patterns": ["patterns to skip like navigation rows"],
                "cleaning_needed": ["currency", "references", "special_chars", "multiline"]
            }},
            "pandas_compatibility": {{
                "can_use_pandas": true/false,
                "suggested_params": {{"attrs": {{}}, "skiprows": 0}},
                "reason": "explanation"
            }}
        }}
        
        Focus on finding the MAIN DATA TABLE with the most relevant information, not navigation or sidebar tables.
        Be specific about CSS selectors and patterns you observe.
        """
        
        response = await ping_gemini(
            analysis_prompt, 
            "You are an HTML parsing expert. Analyze the structure and provide specific extraction guidance. Return only valid JSON."
        )
        
        try:
            if "error" in response:
                print(f"❌ LLM analysis failed: {response['error']}")
                return self._fallback_analysis(html_content)
            
            response_text = response["candidates"][0]["content"]["parts"][0]["text"]
            
            # Clean JSON from markdown if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            strategy = json.loads(response_text)
            print(f"✅ LLM extraction strategy: {strategy.get('method')} for {strategy.get('table_indicators', {}).get('table_count', 0)} tables")
            return strategy
            
        except Exception as e:
            print(f"❌ Error parsing LLM strategy: {e}")
            return self._fallback_analysis(html_content)
    
    def _fallback_analysis(self, html_content: str) -> Dict[str, Any]:
        """Fallback analysis using simple HTML parsing"""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        return {
            "method": "beautifulsoup_guided" if tables else "custom_parsing",
            "table_indicators": {
                "has_html_tables": len(tables) > 0,
                "table_classes": list(set([table.get('class', [''])[0] for table in tables if table.get('class')])),
                "table_count": len(tables),
                "best_table_selector": "table",
                "data_structure": "regular_table"
            },
            "extraction_guidance": {
                "expected_columns": [],
                "header_location": "first_row",
                "data_row_pattern": "standard tr/td structure",
                "skip_patterns": [],
                "cleaning_needed": ["references", "special_chars"]
            },
            "pandas_compatibility": {
                "can_use_pandas": len(tables) > 0,
                "suggested_params": {},
                "reason": "Basic table structure detected"
            }
        }
    
    async def _pandas_extraction_with_llm_guidance(self, html_content: str, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use pandas with LLM-guided parameters"""
        print("📊 Using LLM-guided pandas extraction...")
        
        pandas_params = strategy.get("pandas_compatibility", {}).get("suggested_params", {})
        table_indicators = strategy.get("table_indicators", {})
        
        try:
            # Try with LLM-suggested parameters first
            if "attrs" in pandas_params and pandas_params["attrs"]:
                tables = pd.read_html(StringIO(html_content), attrs=pandas_params["attrs"])
            else:
                tables = pd.read_html(StringIO(html_content))
            
            if not tables:
                raise Exception("No tables found with pandas")
            
            # Use LLM guidance to select the best table
            best_table = await self._select_best_table_with_llm(tables, strategy)
            
            # Clean the table using LLM guidance
            cleaned_table = await self._clean_table_with_llm_guidance(best_table, strategy)
            
            print(f"✅ Pandas extraction successful: {cleaned_table.shape}")
            return cleaned_table
            
        except Exception as e:
            print(f"❌ Pandas extraction failed: {e}")
            return await self._beautifulsoup_extraction_with_llm_guidance(html_content, strategy)
    
    async def _select_best_table_with_llm(self, tables: List[pd.DataFrame], strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use LLM to select the best table from multiple candidates"""
        if len(tables) == 1:
            return tables[0]
        
        # Create summary of each table for LLM analysis
        table_summaries = []
        for i, table in enumerate(tables):
            summary = {
                "table_index": i,
                "shape": table.shape,
                "columns": list(table.columns)[:10],  # First 10 columns
                "sample_data": table.head(3).to_dict('records') if not table.empty else [],
                "has_numeric_data": any(table.dtypes == 'object'),  # Look for potential numeric columns
                "null_percentage": round(table.isnull().sum().sum() / (len(table) * len(table.columns)) * 100, 2)
            }
            table_summaries.append(summary)
        
        expected_columns = strategy.get("extraction_guidance", {}).get("expected_columns", [])
        
        selection_prompt = f"""
        I have {len(tables)} tables extracted from a webpage. Help me select the MAIN DATA TABLE with the most relevant information.
        
        EXPECTED DATA: {expected_columns if expected_columns else "General tabular data"}
        
        TABLE SUMMARIES:
        {json.dumps(table_summaries, indent=2, default=str)}
        
        Return a JSON object with:
        {{
            "selected_table_index": 0,  // Index of the best table
            "reason": "explanation of why this table was chosen",
            "confidence": "high" | "medium" | "low"
        }}
        
        Choose the table that:
        1. Has the most relevant data (not navigation/sidebar tables)
        2. Has reasonable size (not too small, not empty)
        3. Has proper structure with meaningful columns
        4. Contains the type of data we're looking for
        """
        
        try:
            response = await ping_gemini(selection_prompt, "You are a data analysis expert. Select the most relevant table. Return only valid JSON.")
            
            if "error" not in response and "candidates" in response:
                response_text = response["candidates"][0]["content"]["parts"][0]["text"]
                
                # Clean JSON
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                selection = json.loads(response_text)
                selected_idx = selection.get("selected_table_index", 0)
                
                if 0 <= selected_idx < len(tables):
                    print(f"✅ LLM selected table {selected_idx}: {selection.get('reason', 'No reason given')}")
                    return tables[selected_idx]
        
        except Exception as e:
            print(f"❌ LLM table selection failed: {e}")
        
        # Fallback: select largest table with most columns
        return max(tables, key=lambda x: len(x) * len(x.columns))
    
    async def _clean_table_with_llm_guidance(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Clean table using LLM guidance"""
        cleaning_needed = strategy.get("extraction_guidance", {}).get("cleaning_needed", [])
        skip_patterns = strategy.get("extraction_guidance", {}).get("skip_patterns", [])
        
        print(f"🧹 Cleaning table with guidance: {cleaning_needed}")
        
        # Basic cleaning
        df_clean = df.copy()
        
        # FIRST: Remove trailing commas from all text columns before any other processing
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Text columns
                df_clean[col] = df_clean[col].astype(str).str.replace(',', '')
        
        # Remove empty rows and columns
        df_clean = df_clean.dropna(how='all').reset_index(drop=True)
        df_clean = df_clean.loc[:, ~(df_clean.astype(str) == '').all()]
        
        # Apply LLM-guided cleaning
        for clean_type in cleaning_needed:
            if clean_type == "references":
                df_clean = df_clean.map(lambda x: re.sub(r'\[\d+\]', '', str(x)) if pd.notna(x) else x)
            elif clean_type == "special_chars":
                df_clean = df_clean.map(lambda x: str(x).replace('\xa0', ' ').replace('\u2013', '-') if pd.notna(x) else x)
            elif clean_type == "multiline":
                df_clean = df_clean.map(lambda x: ' '.join(str(x).split()) if pd.notna(x) else x)
        
        # Remove header-like rows based on skip patterns
        if skip_patterns or len(df_clean) > 5:
            df_clean = self._remove_duplicate_headers(df_clean)
        
        print(f"✅ Table cleaned: {df.shape} → {df_clean.shape}")
        return df_clean
    
    def _remove_duplicate_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that look like duplicate headers"""
        if len(df) <= 1:
            return df
        
        # Check for rows that match column names
        header_like_indices = []
        column_names_lower = [str(col).lower().strip() for col in df.columns]
        
        for idx, row in df.iterrows():
            row_values_lower = [str(val).lower().strip() for val in row.values]
            
            # Check if row values match column names (fuzzy matching)
            matches = sum(1 for col, val in zip(column_names_lower, row_values_lower) 
                         if col and val and (col in val or val in col))
            
            if matches >= len(df.columns) * 0.6:  # 60% match threshold
                header_like_indices.append(idx)
        
        if header_like_indices:
            print(f"🧹 Removing {len(header_like_indices)} duplicate header rows")
            df = df.drop(header_like_indices).reset_index(drop=True)
        
        return df
    
    async def _beautifulsoup_extraction_with_llm_guidance(self, html_content: str, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Use BeautifulSoup with LLM guidance"""
        print("🔄 Using LLM-guided BeautifulSoup extraction...")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        table_indicators = strategy.get("table_indicators", {})
        
        # Find tables using LLM-suggested selector with fallback
        suggested_selector = table_indicators.get("best_table_selector", "table")
        tables = []
        
        # Try LLM-suggested selector first
        if suggested_selector and suggested_selector != "table":
            print(f"🎯 Trying LLM-suggested selector: {suggested_selector}")
            if "." in suggested_selector and not suggested_selector.startswith("."):
                # Handle class-based selectors
                tables = soup.select(suggested_selector)
            else:
                tables = soup.find_all(suggested_selector)
        
        # Fallback to generic table search if no tables found
        if not tables:
            print("🔄 LLM selector failed, falling back to generic table search...")
            tables = soup.find_all('table')
            
            # If still no tables, try more specific selectors
            if not tables:
                print("🔄 Trying alternative table selectors...")
                # Try common table patterns
                alternative_selectors = [
                    'table.wikitable',
                    'table[class*="table"]',
                    'div[class*="table"] table',
                    '.wikitable',
                    '[class*="result"] table',
                    '.standings table'
                ]
                
                for alt_selector in alternative_selectors:
                    tables = soup.select(alt_selector)
                    if tables:
                        print(f"✅ Found tables with selector: {alt_selector}")
                        break
        
        if not tables:
            raise Exception(f"No tables found on page. Tried selector: {suggested_selector} and fallback methods")
        
        print(f"📊 Found {len(tables)} table(s) on page")
        
        # Score and select best table
        best_table = self._score_and_select_table(tables, strategy)
        
        # Extract data with LLM guidance
        df = await self._extract_table_data_guided(best_table, strategy)
        
        return df
    
    def _score_and_select_table(self, tables, strategy: Dict[str, Any]) -> Any:
        """Score tables and select the best one"""
        best_table = None
        best_score = 0
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:  # Must have header + data
                continue
            
            # Score based on content and structure
            data_cells = sum(len(row.find_all(['td', 'th'])) for row in rows)
            text_content = len(table.get_text(strip=True))
            
            # Prefer tables with more structured content
            score = data_cells * 0.7 + (text_content / 100) * 0.3
            
            if score > best_score:
                best_score = score
                best_table = table
        
        return best_table or tables[0]
    
    async def _extract_table_data_guided(self, table, strategy: Dict[str, Any]) -> pd.DataFrame:
        """Extract table data with LLM guidance"""
        guidance = strategy.get("extraction_guidance", {})
        
        all_rows = table.find_all('tr')
        if not all_rows:
            raise Exception("No rows found in table")
        
        # Extract headers based on guidance
        header_location = guidance.get("header_location", "first_row")
        if header_location == "th_tags":
            # Look for th tags anywhere in the table
            header_cells = table.find_all('th')
            headers = [cell.get_text(strip=True) for cell in header_cells]
        else:
            # Use first row
            header_row = all_rows[0]
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
        
        # Clean headers
        headers = [self._clean_cell_text(h) for h in headers]
        headers = [h if h else f"Column_{i}" for i, h in enumerate(headers)]
        
        # Extract data rows
        data_rows = []
        for row in all_rows[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            
            row_data = [self._clean_cell_text(cell.get_text(strip=True)) for cell in cells]
            
            # Skip empty or irrelevant rows based on guidance
            if not any(cell.strip() for cell in row_data):
                continue
            
            # Ensure row matches header length
            while len(row_data) < len(headers):
                row_data.append('')
            row_data = row_data[:len(headers)]
            
            data_rows.append(row_data)
        
        if not data_rows:
            raise Exception("No data rows extracted")
        
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Apply final cleaning
        return await self._clean_table_with_llm_guidance(df, strategy)
    
    def _clean_cell_text(self, text: str) -> str:
        """Clean individual cell text"""
        if not text:
            return ""
        
        # Remove references [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Clean whitespace and special characters
        text = text.replace('\xa0', ' ').replace('\u2013', '-').replace('\u2014', '-')
        text = ' '.join(text.split())
        
        return text.strip()
    
    async def _fallback_extraction(self, html_content: str) -> pd.DataFrame:
        """Final fallback extraction method"""
        print("🔄 Using fallback extraction...")
        
        # Try pandas first
        try:
            tables = pd.read_html(StringIO(html_content))
            if tables:
                main_table = max(tables, key=lambda x: len(x) * len(x.columns))
                return self._basic_clean_dataframe(main_table)
        except:
            pass
        
        # Try BeautifulSoup as last resort
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            raise Exception("No extractable data found")
        
        # Use the largest table
        best_table = max(tables, key=lambda t: len(t.find_all('tr')))
        
        # Basic extraction
        rows = best_table.find_all('tr')
        data = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = [self._clean_cell_text(cell.get_text()) for cell in cells]
            if any(cell.strip() for cell in row_data):
                data.append(row_data)
        
        if not data:
            raise Exception("No data extracted")
        
        # Create DataFrame with consistent columns
        max_cols = max(len(row) for row in data)
        headers = [f"Column_{i}" for i in range(max_cols)]
        
        # Pad all rows
        for row in data:
            while len(row) < max_cols:
                row.append('')
        
        df = pd.DataFrame(data, columns=headers)
        return self._basic_clean_dataframe(df)
    
    def _basic_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame cleaning"""
        # Remove empty rows and columns
        df = df.dropna(how='all').reset_index(drop=True)
        df = df.loc[:, ~(df.astype(str) == '').all()]
        
        # Remove duplicate headers
        df = self._remove_duplicate_headers(df)
        
        # Apply intelligent pattern-based total row removal
        df = self._remove_total_rows(df)
        
        return df
    
    def _beautifulsoup_table_extract(self, html_content: str) -> pd.DataFrame:
        """Extract table using BeautifulSoup with improved parsing"""
        print("🔄 Using BeautifulSoup fallback for table extraction...")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to find wikitable first (Wikipedia standard)
        wikitables = soup.find_all('table', class_='wikitable')
        if wikitables:
            print(f"📊 Found {len(wikitables)} wikitables")
            tables = wikitables
        else:
            # Find all tables and select the largest one by number of rows
            tables = soup.find_all('table')
            if not tables:
                raise Exception("No tables found in HTML")
            print(f"📊 Found {len(tables)} total tables")
        
        # Get the table with most meaningful rows (data)
        best_table = None
        max_score = 0
        
        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) < 3:  # Skip very small tables
                continue
                
            # Score table based on size and structure
            data_rows = 0
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:  # Must have at least 2 columns
                    cell_text = ' '.join([cell.get_text(strip=True) for cell in cells])
                    if len(cell_text.strip()) > 10:  # Must have meaningful content
                        data_rows += 1
            
            score = data_rows * len(rows[0].find_all(['td', 'th']) if rows else [])
            
            if score > max_score:
                max_score = score
                best_table = table
                print(f"📊 Table {i+1}: {len(rows)} rows, score {score} (current best)")
        
        if not best_table:
            raise Exception("No suitable table found")
        
        return self._extract_table_data(best_table)
    
    def _extract_table_data(self, table) -> pd.DataFrame:
        """Extract clean data from a BeautifulSoup table object"""
        all_rows = table.find_all('tr')
        
        # Extract headers from first row
        header_row = all_rows[0]
        headers = []
        for cell in header_row.find_all(['th', 'td']):
            header_text = cell.get_text(strip=True)
            # Clean header text
            header_text = re.sub(r'\[\d+\]', '', header_text)  # Remove reference numbers
            header_text = ' '.join(header_text.split())  # Clean whitespace
            headers.append(header_text if header_text else f"Column_{len(headers)}")
        
        print(f"📊 Extracted headers: {headers}")
        
        # Extract data rows (skip header)
        data_rows = []
        for row_idx, row in enumerate(all_rows[1:], 1):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
                
            row_data = []
            for cell in cells:
                # Get clean cell text
                cell_text = cell.get_text(strip=True)
                # Remove reference numbers and clean
                cell_text = re.sub(r'\[\d+\]', '', cell_text)
                cell_text = ' '.join(cell_text.split())
                
                # Handle special characters and encoding issues
                cell_text = cell_text.replace('\u2013', '-').replace('\u2014', '-')  # em/en dashes
                cell_text = cell_text.replace('\xa0', ' ')  # non-breaking space
                
                row_data.append(cell_text)
            
            # Only add rows with meaningful data
            if any(cell.strip() for cell in row_data):
                # Pad row to match header length
                while len(row_data) < len(headers):
                    row_data.append('')
                # Trim row if it's too long
                row_data = row_data[:len(headers)]
                data_rows.append(row_data)
        
        # Create DataFrame
        if not data_rows:
            raise Exception("No data rows found in table")
        
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Clean the dataframe
        df = self._post_process_dataframe(df)
        
        print(f"📊 Final DataFrame shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        return df
    
    def _post_process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the dataframe to clean up common issues"""
        # Remove rows that are duplicates of headers
        if len(df) > 1:
            # Check if any row contains header-like values
            header_like_rows = []
            for idx, row in df.iterrows():
                header_similarity = sum(1 for col in df.columns 
                                      if str(row[col]).strip().lower() == col.strip().lower())
                if header_similarity > len(df.columns) * 0.6:  # 60% similarity threshold
                    header_like_rows.append(idx)
            
            if header_like_rows:
                print(f"📊 Removing {len(header_like_rows)} header-like rows")
                df = df.drop(header_like_rows).reset_index(drop=True)
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').reset_index(drop=True)  # Remove empty rows
        df = df.loc[:, ~(df == '').all()]  # Remove empty columns
        
        # Remove rows with mostly empty cells
        threshold = max(1, len(df.columns) // 3)  # At least 1/3 of columns should have data
        df = df.dropna(thresh=threshold).reset_index(drop=True)
        
        return df
    
    async def fetch_webpage_with_session(self, url: str) -> str:
        """Fetch webpage using session method with enhanced cookie collection"""
        try:
            print("🍪 Getting cookies from base domain...")
            
            # Extract base domain for cookie collection
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # First visit homepage to get cookies
            homepage_response = self.session.get(base_url, timeout=30)
            print(f"Homepage status: {homepage_response.status_code}")
            print(f"Cookies received: {len(self.session.cookies)}")
            
            # Wait a bit between requests to seem more human-like
            time.sleep(2)
            
            # Now fetch the actual URL
            print(f"🌐 Fetching {url} with session and cookies...")
            response = self.session.get(url, timeout=30)
            print(f"Target page status: {response.status_code}")
            
            if response.status_code == 200:
                # Check if we got blocked or access denied (more specific detection)
                content_lower = response.text.lower()
                
                # More specific blocking indicators that are less likely to be false positives
                blocked_indicators = [
                    'access denied', 
                    'you don\'t have permission to access',
                    '403 forbidden',
                    'this site is blocked',
                    'your request has been blocked',
                    'cloudflare ray id',
                    'please enable javascript and cookies',
                    'human verification',
                    'please complete the security check'
                ]
                
                # Check for blocking only if we have very specific indicators
                # AND the content is suspiciously short (likely an error page)
                is_blocked = False
                if len(response.text) < 1000:  # Error pages are usually short
                    is_blocked = any(indicator in content_lower for indicator in blocked_indicators)
                else:
                    # For longer content, only check for very specific blocking messages
                    specific_blocks = [
                        'access denied',
                        'you don\'t have permission to access',
                        '403 forbidden',
                        'cloudflare ray id'
                    ]
                    is_blocked = any(indicator in content_lower for indicator in specific_blocks)
                
                if is_blocked:
                    print("❌ Access denied with session method - got blocked page")
                    raise Exception("Access denied - session method blocked by server")
                
                print("✅ Successfully fetched webpage with session method")
                return response.text
            else:
                raise Exception(f"HTTP {response.status_code}: {response.reason}")
                
        except Exception as e:
            print(f"❌ Session method failed: {e}")
            # Let the caller handle fallback strategy
            raise e

class ImprovedWebScraper:
    """Main class that coordinates web scraping and numeric formatting"""
    
    def __init__(self):
        self.numeric_formatter = NumericFieldFormatter()
        self.web_scraper = WebScraper()
    
    async def extract_data(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to extract data from web sources - single table only"""
        # Handle both URL string and config dict formats
        if isinstance(source_config, str):
            url = source_config
        else:
            url = source_config.get("url", "")
            if not url:
                raise Exception("No URL provided in source config")
        
        print(f"🚀 Starting data extraction for: {url}")
        
        # Smart fetch method selection based on domain
        html_content = await self._smart_fetch_webpage(url)
        
        # Extract a single best table
        df = await self.web_scraper.extract_table_from_html(html_content)
        if df is None or df.empty:
            raise Exception(f"No data extracted from {url}")

        print(f"📊 Raw data extracted: 1 table ({df.shape})")

        # Clean numeric fields using LLM
        cleaned_df, formatting_results = await self.numeric_formatter.format_dataframe_numerics(df)

        table_name = "Main_Table"

        processed_tables = [{
            "table_name": table_name,
            "dataframe": cleaned_df,
            "shape": cleaned_df.shape,
            "columns": list(cleaned_df.columns),
            "data_types": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
            "numeric_formatting": formatting_results
        }]

        print("✅ Data cleaning complete: 1 table processed")

        return {
            "tables": processed_tables,
            "metadata": {
                "source_type": "web_scrape",
                "source_url": url,
                "extraction_method": "single_table_detection",
                "total_tables": 1,
                "table_names": [table_name]
            }
        }
    
    async def _smart_fetch_webpage(self, url: str) -> str:
        """Try cookie-based session method first, fallback to Playwright if it fails"""
        print("🍪 Trying cookie-based session scraping first...")
        
        try:
            # Try session method first (faster and less resource intensive)
            html_content = await self.web_scraper.fetch_webpage_with_session(url)
            print("✅ Cookie-based session scraping successful")
            return html_content
            
        except Exception as e:
            print(f"❌ Session method failed: {e}")
            print("🔄 Falling back to Playwright scraping...")
            
            try:
                # Fallback to Playwright method
                html_content = await self.web_scraper.fetch_webpage(url)
                print("✅ Playwright scraping successful")
                return html_content
                
            except Exception as e2:
                print(f"❌ Playwright method also failed: {e2}")
                # Re-raise the original error
                raise Exception(f"Both scraping methods failed. Session: {e}, Playwright: {e2}")
        
    # Unreachable fallback removed
    
    async def scrape_and_clean(self, url: str) -> Dict[str, Any]:
        """Alias method for backward compatibility"""
        return await self.extract_data(url)
    
    async def extract_multiple_tables(self, url: str, max_tables: int = 3) -> Dict[str, Any]:
        """Deprecated: extracts a single table and saves as one CSV file"""
        print(f"🚀 Starting single table extraction for: {url}")
        
        try:
            # Fetch webpage using smart method
            html_content = await self._smart_fetch_webpage(url)
            
            # Extract single table
            df = await self.web_scraper.extract_table_from_html(html_content)
            if df is None or df.empty:
                print("❌ No tables found on the webpage")
                return {"success": False, "message": "No tables found", "tables_found": 0, "files_saved": []}

            # Clean numeric fields
            cleaned_df, formatting_results = await self.numeric_formatter.format_dataframe_numerics(df)

            # Generate filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_url = url.split('//')[-1].replace('/', '_').replace('.', '_')[:30]
            table_name = "Main_Table"
            safe_table_name = re.sub(r'[^\w\s-]', '', table_name).replace(' ', '_')[:40]
            filename = f"{safe_table_name}_{safe_url}_{timestamp}.csv"

            cleaned_df.to_csv(filename, index=False)

            table_info = {
                "table_number": 1,
                "table_name": table_name,
                "filename": filename,
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "data_types": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
                "sample_data": cleaned_df.head(3).to_dict('records'),
                "numeric_formatting": formatting_results
            }

            print(f"\n✅ Successfully processed 1 table")
            print(f"📁 Saved file: {filename}")

            return {
                "success": True,
                "source_url": url,
                "tables_found": 1,
                "tables_processed": 1,
                "files_saved": [filename],
                "tables_info": [table_info],
                "metadata": {
                    "extraction_method": "single_table_detection",
                    "timestamp": datetime.now().isoformat(),
                    "max_tables_requested": 1
                }
            }
            
        except Exception as e:
            print(f"❌ Error in multiple table extraction: {e}")
            return {
                "success": False,
                "error": str(e),
                "tables_found": 0,
                "files_saved": []
            }


class DataScraper:
    """Compatibility wrapper class that provides the methods expected by app.py"""
    
    def __init__(self):
        self.numeric_formatter = NumericFieldFormatter()
        self.web_scraper = WebScraper()
    
    def _basic_clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame cleaning - delegates to WebScraper"""
        return self.web_scraper._basic_clean_dataframe(df)
    
    def _clean_currency_column(self, series: pd.Series) -> pd.Series:
        """Clean currency values - delegates to NumericFieldFormatter"""
        return self.numeric_formatter._clean_currency_column(series)
    
    def _clean_percentage_column(self, series: pd.Series) -> pd.Series:
        """Clean percentage values - delegates to NumericFieldFormatter"""
        return self.numeric_formatter._clean_percentage_column(series)
    
    def _clean_generic_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean generic numeric values - delegates to NumericFieldFormatter"""
        return self.numeric_formatter._clean_generic_numeric_column(series)


# ============================================================================
# ENHANCED SCRAPING CAPABILITIES - QUESTION-AWARE EXTRACTION
# ============================================================================

class QuestionAnalyzer:
    """Lightweight question analyzer for determining extraction focus"""
    
    def __init__(self):
        # Keep lists small for memory efficiency
        self.financial_keywords = ['revenue', 'profit', 'earnings', 'sales', 'price', 'cost']
        self.sports_keywords = ['score', 'match', 'game', 'player', 'team', 'statistics']
        self.temporal_keywords = ['quarterly', 'annual', 'monthly', 'latest', 'recent']
        self.comparison_keywords = ['compare', 'vs', 'versus', 'difference']
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Quick analysis of question to guide extraction - memory efficient"""
        if not question:
            return {"domain": "general", "keywords": [], "intent": "general"}
        
        question_lower = question.lower()
        
        # Extract key information efficiently
        keywords = [word for word in question_lower.split() 
                   if len(word) > 2 and word not in {'the', 'and', 'for', 'are', 'what', 'how'}][:10]  # Limit to 10 keywords
        
        # Determine domain
        domain = "general"
        if any(kw in question_lower for kw in self.financial_keywords):
            domain = "finance"
        elif any(kw in question_lower for kw in self.sports_keywords):
            domain = "sports"
        
        # Determine intent
        intent = "information"
        if any(kw in question_lower for kw in self.comparison_keywords):
            intent = "comparison"
        elif "latest" in question_lower or "recent" in question_lower:
            intent = "current_data"
        
        return {
            "domain": domain,
            "keywords": keywords,
            "intent": intent,
            "has_temporal": any(kw in question_lower for kw in self.temporal_keywords)
        }

class EnhancedContentExtractor:
    """Memory-efficient enhanced content extractor"""
    
    def __init__(self):
        self.question_analyzer = QuestionAnalyzer()
    
    def extract_relevant_content(self, html_content: str, question: str = "", max_sources: int = 3) -> List[Dict[str, Any]]:
        """
        Extract only the most relevant content sources - memory efficient
        Returns list of content dictionaries instead of storing all in memory
        """
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        question_analysis = self.question_analyzer.analyze_question(question)
        
        content_sources = []
        
        # Extract and score content sources on-the-fly
        # 1. HTML Tables (existing functionality enhanced)
        tables = self._extract_relevant_tables(soup, question_analysis, max_tables=max_sources)
        content_sources.extend(tables)
        
        # 2. Only extract other content types if we have space and question suggests it
        if len(content_sources) < max_sources:
            # Card layouts for product/item data
            if question_analysis["domain"] in ["general", "finance"] or "price" in question_analysis["keywords"]:
                cards = self._extract_card_data(soup, question_analysis, max_cards=max_sources - len(content_sources))
                content_sources.extend(cards)
        
        if len(content_sources) < max_sources:
            # JSON-LD for structured data
            json_data = self._extract_json_ld(soup, question_analysis, max_items=max_sources - len(content_sources))
            content_sources.extend(json_data)
        
        if len(content_sources) < max_sources:
            # Key-value pairs from text
            kv_pairs = self._extract_key_value_text(soup, question_analysis)
            if kv_pairs:
                content_sources.append(kv_pairs)
        
        # Sort by relevance and return top results
        content_sources.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return content_sources[:max_sources]
    
    def _extract_relevant_tables(self, soup: BeautifulSoup, question_analysis: Dict, max_tables: int = 3) -> List[Dict[str, Any]]:
        """Extract and score HTML tables for relevance"""
        tables = soup.find_all('table')
        relevant_tables = []
        
        for i, table in enumerate(tables[:10]):  # Limit processing to first 10 tables for memory
            try:
                # Quick relevance check before expensive DataFrame conversion
                table_text = table.get_text().lower()
                
                # Skip clearly irrelevant tables
                if self._is_navigation_table(table, table_text):
                    continue
                
                # Convert to DataFrame only if it passes initial checks
                df = self._table_to_dataframe_efficient(table)
                if df is None or df.empty or len(df) < 2:
                    continue
                
                # Score relevance
                relevance_score = self._score_table_relevance(df, table_text, question_analysis)
                
                if relevance_score > 0.2:  # Only keep tables with decent relevance
                    relevant_tables.append({
                        "content_type": "table",
                        "content": df,
                        "relevance_score": relevance_score,
                        "source_info": f"table_{i}",
                        "metadata": {
                            "rows": len(df),
                            "columns": len(df.columns),
                            "table_classes": table.get('class', [])
                        }
                    })
                
                # Clean up immediately to save memory
                del df
                
                if len(relevant_tables) >= max_tables:
                    break
                    
            except Exception as e:
                print(f"Error processing table {i}: {e}")
                continue
        
        return relevant_tables
    
    def _extract_card_data(self, soup: BeautifulSoup, question_analysis: Dict, max_cards: int = 2) -> List[Dict[str, Any]]:
        """Extract data from card layouts - memory efficient"""
        card_selectors = ['.card', '[class*="card"]', '.product', '[class*="item"]']
        
        for selector in card_selectors:
            cards = soup.select(selector)
            if len(cards) >= 3:  # Only process if we have multiple cards
                try:
                    # Limit processing to avoid memory overload
                    cards_to_process = cards[:20]  # Max 20 cards
                    
                    card_data = []
                    for card in cards_to_process:
                        data = self._extract_card_info(card)
                        if data:
                            card_data.append(data)
                    
                    if len(card_data) >= 2:
                        df = pd.DataFrame(card_data)
                        relevance_score = self._score_content_relevance(df, question_analysis)
                        
                        if relevance_score > 0.1:
                            return [{
                                "content_type": "cards",
                                "content": df,
                                "relevance_score": relevance_score,
                                "source_info": f"cards_{selector}",
                                "metadata": {"card_count": len(card_data)}
                            }]
                
                except Exception as e:
                    print(f"Error processing cards {selector}: {e}")
                    continue
        
        return []
    
    def _extract_json_ld(self, soup: BeautifulSoup, question_analysis: Dict, max_items: int = 2) -> List[Dict[str, Any]]:
        """Extract JSON-LD structured data - memory efficient"""
        json_scripts = soup.find_all('script', type='application/ld+json')
        json_sources = []
        
        for i, script in enumerate(json_scripts[:5]):  # Limit to first 5 JSON-LD scripts
            try:
                json_data = json.loads(script.string)
                
                # Convert to DataFrame efficiently
                if isinstance(json_data, list):
                    df = pd.json_normalize(json_data[:10])  # Limit to 10 items
                elif isinstance(json_data, dict):
                    df = pd.json_normalize([json_data])
                else:
                    continue
                
                if not df.empty:
                    relevance_score = self._score_content_relevance(df, question_analysis)
                    
                    if relevance_score > 0.1:
                        json_sources.append({
                            "content_type": "json_ld",
                            "content": df,
                            "relevance_score": relevance_score,
                            "source_info": f"json_ld_{i}",
                            "metadata": {"json_type": json_data.get('@type', 'Unknown')}
                        })
                
                if len(json_sources) >= max_items:
                    break
                    
            except Exception as e:
                print(f"Error processing JSON-LD {i}: {e}")
                continue
        
        return json_sources
    
    def _extract_key_value_text(self, soup: BeautifulSoup, question_analysis: Dict) -> Optional[Dict[str, Any]]:
        """Extract key-value pairs from text - lightweight"""
        text_content = soup.get_text()
        
        # Pattern for key-value pairs (limit search to avoid memory issues)
        kv_pattern = r'([A-Za-z][A-Za-z\s]{2,30}):\s*([^\n\r]{1,100})'
        matches = re.findall(kv_pattern, text_content)
        
        if len(matches) >= 3:  # Only if we found multiple pairs
            # Limit to first 20 matches to avoid memory overload
            limited_matches = matches[:20]
            
            try:
                df = pd.DataFrame(limited_matches, columns=['Key', 'Value'])
                relevance_score = self._score_content_relevance(df, question_analysis)
                
                if relevance_score > 0.1:
                    return {
                        "content_type": "key_value",
                        "content": df,
                        "relevance_score": relevance_score,
                        "source_info": "text_kv_pairs",
                        "metadata": {"pair_count": len(limited_matches)}
                    }
            except Exception as e:
                print(f"Error processing key-value pairs: {e}")
        
        return None
    
    def _table_to_dataframe_efficient(self, table) -> Optional[pd.DataFrame]:
        """Memory-efficient table to DataFrame conversion"""
        try:
            rows = table.find_all('tr')
            if not rows or len(rows) > 100:  # Skip very large tables to save memory
                return None
            
            # Get headers efficiently
            first_row = rows[0]
            headers = [cell.get_text().strip() for cell in first_row.find_all(['th', 'td'])]
            
            # Get data rows (limit to 50 rows for memory efficiency)
            data = []
            data_rows = rows[1:] if headers else rows
            for row in data_rows[:50]:  # Limit rows
                row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                if row_data and len(row_data) <= 20:  # Limit columns too
                    data.append(row_data)
            
            if not data:
                return None
            
            # Create DataFrame with proper column handling
            max_cols = max(len(row) for row in data) if data else 0
            if headers and len(headers) >= max_cols:
                df = pd.DataFrame(data, columns=headers[:max_cols])
            else:
                df = pd.DataFrame(data)
            
            return df if not df.empty else None
            
        except Exception as e:
            print(f"Error converting table to DataFrame: {e}")
            return None
    
    def _extract_card_info(self, card) -> Optional[Dict[str, Any]]:
        """Extract info from a single card - memory efficient"""
        try:
            card_data = {}
            
            # Get title
            title_elem = card.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', '.title'])
            if title_elem:
                card_data['title'] = title_elem.get_text().strip()[:100]  # Limit length
            
            # Get price
            price_elem = card.find(['span', 'div'], string=re.compile(r'[\$€£¥]')) or \
                        card.find(['span', 'div'], class_=re.compile(r'price'))
            if price_elem:
                card_data['price'] = price_elem.get_text().strip()[:50]
            
            # Get first number found (could be rating, count, etc.)
            numbers = re.findall(r'\d+(?:\.\d+)?', card.get_text())
            if numbers:
                card_data['numeric_value'] = numbers[0]
            
            # Only return if we found useful data
            return card_data if len(card_data) > 1 else None
            
        except Exception:
            return None
    
    def _is_navigation_table(self, table, table_text: str) -> bool:
        """Quick check to skip navigation tables"""
        nav_indicators = ['home', 'about', 'contact', 'menu', 'navigation', 'footer', 'header']
        return (len(table_text) < 100 and 
                any(indicator in table_text for indicator in nav_indicators))
    
    def _score_table_relevance(self, df: pd.DataFrame, table_text: str, question_analysis: Dict) -> float:
        """Quick relevance scoring for tables"""
        if not question_analysis or not question_analysis.get("keywords"):
            return 0.5  # Default score
        
        score = 0.0
        content_text = (table_text + ' ' + ' '.join(df.columns)).lower()
        
        # Keyword matching (simplified)
        keywords = question_analysis.get("keywords", [])
        if keywords:
            matches = sum(1 for kw in keywords if kw in content_text)
            score += (matches / len(keywords)) * 0.6
        
        # Domain bonus
        domain = question_analysis.get("domain", "general")
        if domain == "finance":
            financial_terms = ['revenue', 'profit', 'price', '$', '%']
            if any(term in content_text for term in financial_terms):
                score += 0.3
        elif domain == "sports":
            sports_terms = ['score', 'points', 'match', 'player']
            if any(term in content_text for term in sports_terms):
                score += 0.3
        
        # Table size bonus (prefer substantial tables)
        if len(df) > 3 and len(df.columns) > 2:
            score += 0.1
        
        return min(1.0, score)
    
    def _score_content_relevance(self, df: pd.DataFrame, question_analysis: Dict) -> float:
        """Quick relevance scoring for any content"""
        if not question_analysis or not question_analysis.get("keywords"):
            return 0.3
        
        content_text = ' '.join([str(val) for val in df.values.flatten()]).lower()
        keywords = question_analysis.get("keywords", [])
        
        if keywords:
            matches = sum(1 for kw in keywords if kw in content_text)
            return min(0.8, (matches / len(keywords)) * 0.7)
        
        return 0.3


class EnhancedWebScraper(ImprovedWebScraper):
    """
    Enhanced version of ImprovedWebScraper with question-aware capabilities
    Extends existing functionality without breaking anything
    """
    
    def __init__(self):
        super().__init__()
        self.enhanced_extractor = EnhancedContentExtractor()
    
    async def extract_data_with_question(self, source_config: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        """
        Enhanced data extraction guided by question
        Falls back to original method if enhanced extraction fails
        """
        try:
            # If it's a URL, handle it with question awareness
            if isinstance(source_config, str) or source_config.get('url'):
                url = source_config if isinstance(source_config, str) else source_config.get('url')
                return await self._extract_from_url_with_question(url, question)
            else:
                # Fall back to original method for non-URL sources
                return await super().extract_data(source_config)
                
        except Exception as e:
            print(f"Enhanced extraction failed, falling back to original: {e}")
            return await super().extract_data(source_config)
    
    async def _extract_from_url_with_question(self, url: str, question: str) -> Dict[str, Any]:
        """Extract data from URL with question guidance"""
        print(f"🎯 Enhanced extraction for: {question[:50]}..." if question else "🔍 Enhanced extraction (no question)")
        
        try:
            # Get HTML content using existing smart fetch
            html_content = await self._smart_fetch_webpage(url)
            
            if not question:
                # No question provided, fall back to original table extraction
                print("No question provided, using original table extraction")
                df = await self.web_scraper.extract_table_from_html(html_content)
                if df is None or df.empty:
                    return {"success": False, "message": "No tables found"}
                
                # Apply numeric formatting
                cleaned_df, formatting_results = await self.numeric_formatter.format_dataframe_numerics(df)
                
                return {
                    "success": True,
                    "source_url": url,
                    "extraction_method": "original_table_extraction",
                    "final_data": cleaned_df,
                    "shape": cleaned_df.shape,
                    "columns": list(cleaned_df.columns),
                    "numeric_formatting": formatting_results
                }
            
            # Extract relevant content using enhanced method
            content_sources = self.enhanced_extractor.extract_relevant_content(html_content, question, max_sources=3)
            
            if not content_sources:
                print("No relevant content found with enhanced method, trying original")
                # Fall back to original method
                df = await self.web_scraper.extract_table_from_html(html_content)
                if df is None or df.empty:
                    return {"success": False, "message": "No relevant content found"}
                
                cleaned_df, formatting_results = await self.numeric_formatter.format_dataframe_numerics(df)
                return {
                    "success": True,
                    "source_url": url,
                    "extraction_method": "fallback_table_extraction",
                    "final_data": cleaned_df,
                    "shape": cleaned_df.shape,
                    "columns": list(cleaned_df.columns),
                    "numeric_formatting": formatting_results
                }
            
            # Combine the most relevant sources
            best_source = content_sources[0]  # Top relevance
            final_df = best_source["content"]
            
            # Apply numeric formatting to final DataFrame
            cleaned_df, formatting_results = await self.numeric_formatter.format_dataframe_numerics(final_df)
            
            # Prepare additional sources info (without storing full DataFrames)
            additional_sources = []
            for source in content_sources[1:]:
                additional_sources.append({
                    "content_type": source["content_type"],
                    "relevance_score": source["relevance_score"],
                    "shape": source["content"].shape,
                    "source_info": source["source_info"]
                })
            
            print(f"✅ Enhanced extraction found {len(content_sources)} relevant sources")
            print(f"📊 Using best source: {best_source['content_type']} (relevance: {best_source['relevance_score']:.3f})")
            
            return {
                "success": True,
                "source_url": url,
                "question": question,
                "extraction_method": "enhanced_question_aware",
                "final_data": cleaned_df,
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "best_source_type": best_source["content_type"],
                "best_source_relevance": best_source["relevance_score"],
                "total_sources_found": len(content_sources),
                "additional_sources": additional_sources,
                "numeric_formatting": formatting_results
            }
            
        except Exception as e:
            print(f"Error in enhanced extraction: {e}")
            # Final fallback to original method
            return await super().extract_data(url)
    
    async def get_relevant_tables_only(self, url: str, question: str, max_tables: int = 3) -> Dict[str, Any]:
        """
        Get only tables relevant to the question - memory efficient
        """
        try:
            html_content = await self._smart_fetch_webpage(url)
            soup = BeautifulSoup(html_content, 'html.parser')
            
            question_analysis = self.enhanced_extractor.question_analyzer.analyze_question(question)
            relevant_tables = self.enhanced_extractor._extract_relevant_tables(soup, question_analysis, max_tables)
            
            if not relevant_tables:
                return {
                    "success": False,
                    "message": "No relevant tables found",
                    "question": question,
                    "tables_found": 0
                }
            
            # Process each table and apply numeric formatting
            processed_tables = []
            for i, table_source in enumerate(relevant_tables):
                df = table_source["content"]
                cleaned_df, formatting_results = await self.numeric_formatter.format_dataframe_numerics(df)
                
                table_info = {
                    "table_number": i + 1,
                    "content_type": table_source["content_type"],
                    "relevance_score": table_source["relevance_score"],
                    "data": cleaned_df,
                    "shape": cleaned_df.shape,
                    "columns": list(cleaned_df.columns),
                    "source_info": table_source["source_info"],
                    "metadata": table_source["metadata"],
                    "numeric_formatting": formatting_results
                }
                processed_tables.append(table_info)
            
            return {
                "success": True,
                "source_url": url,
                "question": question,
                "tables_found": len(processed_tables),
                "relevant_tables": processed_tables,
                "message": f"Found {len(processed_tables)} relevant tables"
            }
            
        except Exception as e:
            print(f"Error getting relevant tables: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "tables_found": 0
            }