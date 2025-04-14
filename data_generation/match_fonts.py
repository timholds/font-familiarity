import os
import glob
import logging
from pathlib import Path
from PIL import ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FontMatcher:
    """Match font names from a list to font files in a directory."""
    
    def __init__(self, font_list_file, fonts_dir="fonts"):
        """
        Initialize the font matcher.
        
        Args:
            font_list_file: Path to a file containing a list of font names
            fonts_dir: Path to the directory containing font files
        """
        self.font_list_file = font_list_file
        self.fonts_dir = Path(fonts_dir)
        self.font_mappings = {}  # Maps font names to primary font files
        self.font_cache = {}     # Cache for loaded fonts
        
        # Load all font files
        self._discover_font_files()
        
        # Match font names to font files
        self._match_fonts()
    
    def normalize_font_name(self, name):
        """Normalize a font name by removing spaces and converting to lowercase."""
        return name.replace(' ', '').lower()
    
    def _discover_font_files(self):
        """Discover all font files in the fonts directory."""
        self.all_font_files = {}
        
        # Find all .ttf files using glob
        ttf_files = glob.glob(str(self.fonts_dir / "**" / "*.ttf"), recursive=True)
        
        # Group font files by directory
        for file_path in ttf_files:
            dir_name = os.path.basename(os.path.dirname(file_path))
            if dir_name not in self.all_font_files:
                self.all_font_files[dir_name] = []
            self.all_font_files[dir_name].append(file_path)
        
        logger.info(f"Found {len(ttf_files)} font files in {len(self.all_font_files)} directories")
    
    def _match_fonts(self):
        """Match font names from the list to font files."""
        # Read font list
        with open(self.font_list_file, 'r', encoding='utf-8') as f:
            font_names = [line.strip() for line in f if line.strip()]
        
        # Try to match each font name to a directory
        for font_name in font_names:
            normalized_name = self.normalize_font_name(font_name)
            
            # Try exact directory name match
            if normalized_name in self.all_font_files:
                # Found exact match - use the first font file in this directory
                font_files = self.all_font_files[normalized_name]
                # Prefer regular/normal fonts if available
                regular_files = [f for f in font_files if 'regular' in f.lower()]
                if regular_files:
                    self.font_mappings[font_name] = regular_files[0]
                else:
                    # Just use the first font file
                    self.font_mappings[font_name] = font_files[0]
                continue
            
            # Try to find a directory that contains this font name
            found = False
            for dir_name, font_files in self.all_font_files.items():
                if normalized_name in dir_name.lower():
                    # Found a partial match
                    regular_files = [f for f in font_files if 'regular' in f.lower()]
                    if regular_files:
                        self.font_mappings[font_name] = regular_files[0]
                    else:
                        self.font_mappings[font_name] = font_files[0]
                    found = True
                    break
            
            if not found:
                logger.warning(f"No match found for font: {font_name}, {normalized_name}")
        
        logger.info(f"Matched {len(self.font_mappings)} out of {len(font_names)} fonts")
    
    def get_available_fonts(self):
        """Get a list of available fonts that have matching files."""
        return list(self.font_mappings.keys())
    
    def get_font_file(self, font_name):
        """
        Get the file path for a font.
        
        Args:
            font_name: Font name as it appears in the font list file
            
        Returns:
            Path to the font file
            
        Raises:
            ValueError: If the font is not found
        """
        if font_name in self.font_mappings:
            return self.font_mappings[font_name]
        else:
            raise ValueError(f"Font '{font_name}' not found in available fonts")
    
    def load_font(self, font_name, font_size):
        """
        Load a font with the specified name and size.
        
        Args:
            font_name: Font name as it appears in the font list file
            font_size: Size of the font to load
            
        Returns:
            PIL ImageFont object
            
        Raises:
            ValueError: If the font is not found
        """
        cache_key = f"{font_name}_{font_size}"

        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        # Get the font file path
        font_path = self.get_font_file(font_name)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
            self.font_cache[cache_key] = font
            return font
        except Exception as e:
            raise ValueError(f"Failed to load font '{font_name}' from {font_path}: {e}")
    
    def get_unmatched_fonts(self):
        """Get a list of fonts from the font list that don't have matching files."""
        with open(self.font_list_file, 'r', encoding='utf-8') as f:
            all_fonts = [line.strip() for line in f if line.strip()]
        
        return [font for font in all_fonts if font not in self.font_mappings]
    
    def generate_pruned_font_list(self, output_file):
        """
        Generate a new font list containing only the fonts that have matching files.
        
        Args:
            output_file: Path to the output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for font_name in self.get_available_fonts():
                f.write(f"{font_name}\n")
        
        logger.info(f"Generated pruned font list with {len(self.font_mappings)} fonts")

# Example usage
if __name__ == "__main__":
    font_matcher = FontMatcher("data_generation/full_fonts_list.txt", "fonts")
    
    # Print some stats
    available_fonts = font_matcher.get_available_fonts()
    unmatched_fonts = font_matcher.get_unmatched_fonts()
    
    print(f"Found {len(available_fonts)} available fonts out of {len(available_fonts) + len(unmatched_fonts)} in the list")
    
    # Print a few examples
    print("\nSample available fonts:")
    for font_name in available_fonts[:5]:
        font_file = font_matcher.get_font_file(font_name)
        print(f"Font '{font_name}' -> {font_file}")
    
    print("\nUnmatched fonts:")
    for font_name in unmatched_fonts:
        print(f"Font '{font_name}' not found")
    # Generate a pruned font list
    font_matcher.generate_pruned_font_list("available_fonts.txt")
    
    # Test loading a font
    try:
        font = font_matcher.load_font(available_fonts[0], 24)
        print(f"\nSuccessfully loaded {available_fonts[0]} at size 24")
    except ValueError as e:
        print(f"\nError loading font: {e}")