import os
import requests
import re
import time
import random
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

def try_font_formats(font_name):
    """Generate different font name formats to try with Google Fonts API"""
    formats = [
        font_name,  # Original
        font_name.replace(' ', '+'),  # Spaces to +
        font_name.lower(),  # Lowercase
        font_name.lower().replace(' ', '+'),  # Lowercase, spaces to +
        font_name.title(),  # Title case
        font_name.title().replace(' ', '+'),  # Title case, spaces to +
        font_name.replace(' ', ''),  # No spaces
        font_name.lower().replace(' ', ''),  # Lowercase, no spaces
        font_name.title().replace(' ', ''),  # Title case, no spaces
    ]
    return formats

def download_font(font_name, output_dir):
    """Download a font from Google Fonts"""
    # Create font directory
    font_dir = Path(output_dir) / font_name.replace(' ', '_')
    font_dir.mkdir(parents=True, exist_ok=True)
    
    # Browser user-agent to request all font formats
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    # Try different font name formats
    for formatted_name in try_font_formats(font_name):
        # Request font with regular and bold weights
        css_url = f"https://fonts.googleapis.com/css2?family={formatted_name}:wght@400;700&display=swap"
        
        try:
            # Small delay to avoid rate limiting
            time.sleep(random.uniform(0.2, 0.5))
            
            response = requests.get(css_url, headers=headers)
            
            if response.status_code != 200:
                continue  # Try next format
            
            # Extract font URLs from the CSS
            font_urls = re.findall(r'src:\s*url\((https://[^)]+)\)', response.text)
            
            if not font_urls:
                continue  # Try next format
            
            # Extract style blocks to get font weights
            style_blocks = re.findall(r'@font-face\s*{([^}]+)}', response.text, re.DOTALL)
            
            # Download each font file
            downloaded = 0
            
            for i, block in enumerate(style_blocks):
                if i >= len(font_urls):
                    break
                
                url = font_urls[i]
                
                # Extract weight information
                weight_match = re.search(r'font-weight:\s*([^;]+)', block)
                weight = weight_match.group(1).strip() if weight_match else '400'
                
                # Create filename based on weight
                file_ext = '.ttf'  # Default extension
                url_path = url.split('/')[-1].split('?')[0]
                url_ext = os.path.splitext(url_path)[1]
                if url_ext:
                    file_ext = url_ext
                
                filename = f"{font_name.replace(' ', '_')}_{weight}{file_ext}"
                file_path = font_dir / filename
                
                # Skip if file already exists
                if file_path.exists():
                    downloaded += 1
                    continue
                
                # Download the font file
                font_response = requests.get(url)
                
                if font_response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(font_response.content)
                    
                    downloaded += 1
            
            if downloaded > 0:
                # Save the Google Fonts name format that worked
                with open(output_dir / 'google_name.txt', 'a') as f:
                    f.write(formatted_name)
                
                return True, f"Downloaded {downloaded} files"
            
        except Exception:
            continue
    
    # If CSS2 API failed, try older CSS API as fallback
    for formatted_name in try_font_formats:
        # If CSS2 API failed, try older CSS API as fallback
        try:
            fallback_url = f"https://fonts.googleapis.com/css?family={font_name.replace(' ', '+')}"
            response = requests.get(fallback_url, headers=headers)
            
            if response.status_code == 200:
                font_urls = re.findall(r'src:\s*url\((https://[^)]+)\)', response.text)
                
                if font_urls:
                    # Download the font
                    for i, url in enumerate(font_urls):
                        file_ext = '.ttf'
                        url_path = url.split('/')[-1].split('?')[0]
                        url_ext = os.path.splitext(url_path)[1]
                        if url_ext:
                            file_ext = url_ext
                        
                        filename = f"{font_name.replace(' ', '_')}_400{file_ext}"
                        file_path = font_dir / filename
                        
                        if not file_path.exists():
                            font_response = requests.get(url)
                            if font_response.status_code == 200:
                                with open(file_path, 'wb') as f:
                                    f.write(font_response.content)
                                
                                with open(font_dir / 'google_name.txt', 'w') as f:
                                    f.write(font_name.replace(' ', '+'))
                                
                                return True, "Downloaded with fallback method"
        except Exception:
            pass

        return False, "Failed to download font"

def download_all_fonts(fonts_file, output_dir, max_workers=10):
    """Download all fonts listed in the fonts.txt file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read font names from file
    with open(fonts_file, 'r', encoding='utf-8') as f:
        font_names = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(font_names)} fonts to download")
    
    # Create failed fonts log file
    failed_log = output_dir / 'failed_fonts.txt'
    with open(failed_log, 'w', encoding='utf-8') as f:
        f.write(f"Failed fonts log created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    
    results = []
    
    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_font = {
            executor.submit(download_font, font_name, output_dir): font_name
            for font_name in font_names
        }
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_font), total=len(font_names), desc="Downloading fonts"):
            font_name = future_to_font[future]
            try:
                success, message = future.result()
                results.append((font_name, success, message))
                
                if not success:
                    with open(failed_log, 'a', encoding='utf-8') as f:
                        f.write(f"{font_name}: {message}\n")
            except Exception as e:
                results.append((font_name, False, str(e)))
                with open(failed_log, 'a', encoding='utf-8') as f:
                    f.write(f"{font_name}: Exception - {str(e)}\n")
    
    # Summarize results
    succeeded = sum(1 for _, success, _ in results if success)
    print(f"\nDownloaded {succeeded} out of {len(font_names)} fonts")
    
    if succeeded < len(font_names):
        print(f"Failed to download {len(font_names) - succeeded} fonts")
        print(f"See {failed_log} for details")
    else:
        print("All fonts downloaded successfully")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download fonts from Google Fonts")
    parser.add_argument("--fonts-file", default="full_fonts_list.txt", help="File containing list of font names")
    parser.add_argument("--output-dir", default="fonts", help="Directory to save downloaded fonts")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of concurrent downloads")
    
    args = parser.parse_args()
    
    download_all_fonts(args.fonts_file, args.output_dir, args.max_workers)