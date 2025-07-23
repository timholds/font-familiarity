import os
import glob
import re
import json

# Directory containing all font subfolders
FONTS_DIR = os.path.join(os.path.dirname(__file__), '../fonts')

# Pattern to extract category from METADATA.pb
CATEGORY_PATTERN = re.compile(r'category:\s*"([A-Z_]+)"')
# Pattern to extract name from METADATA.pb
NAME_PATTERN = re.compile(r'name:\s*"([^"]+)"')

def parse_metadata_pb(metadata_path):
    """Parse METADATA.pb and return (font_name, category)"""
    with open(metadata_path, 'r') as f:
        content = f.read()
    # Get category
    cat_match = CATEGORY_PATTERN.search(content)
    name_match = NAME_PATTERN.search(content)
    if cat_match and name_match:
        return name_match.group(1), cat_match.group(1)
    return None, None

def build_font_category_mapping():
    mapping = {}
    for font_dir in os.listdir(FONTS_DIR):
        meta_path = os.path.join(FONTS_DIR, font_dir, 'METADATA.pb')
        if os.path.isfile(meta_path):
            font_name, category = parse_metadata_pb(meta_path)
            if font_name and category:
                mapping[font_name] = category
    return mapping

def main():
    mapping = build_font_category_mapping()
    out_path = os.path.join(os.path.dirname(__file__), '../data/font_class_to_category.json')
    with open(out_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote mapping for {len(mapping)} fonts to {out_path}")

if __name__ == "__main__":
    main()
