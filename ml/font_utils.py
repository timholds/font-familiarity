"""
Font name normalization utilities for consistent font naming across the codebase.

This module provides centralized font name normalization to solve the inconsistency
between JavaScript (spaces), Python (underscores), and category mapping (proper case).
"""

import os
import json
import logging
from typing import Dict, Optional, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FontNameNormalizer:
    """Centralized font name normalization utility."""
    
    def __init__(self, available_fonts_path: Optional[str] = None, 
                 category_mapping_path: Optional[str] = None):
        """
        Initialize the font name normalizer.
        
        Args:
            available_fonts_path: Path to available_fonts.txt file
            category_mapping_path: Path to font_class_to_category.json file
        """
        # Set default paths relative to this file
        if available_fonts_path is None:
            available_fonts_path = os.path.join(
                os.path.dirname(__file__), '../static/available_fonts.txt'
            )
        if category_mapping_path is None:
            category_mapping_path = os.path.join(
                os.path.dirname(__file__), '../data/font_class_to_category.json'
            )
            
        self.available_fonts_path = available_fonts_path
        self.category_mapping_path = category_mapping_path
        
        # Mapping dictionaries
        self._canonical_to_proper: Dict[str, str] = {}
        self._proper_to_canonical: Dict[str, str] = {}
        self._category_mapping: Dict[str, str] = {}
        
        # Load mappings
        self._load_font_mappings()
        self._load_category_mappings()
        
    def _load_font_mappings(self):
        """Load font name mappings from available_fonts.txt."""
        if not os.path.exists(self.available_fonts_path):
            logger.warning(f"Available fonts file not found: {self.available_fonts_path}")
            return
            
        try:
            with open(self.available_fonts_path, 'r', encoding='utf-8') as f:
                for line in f:
                    proper_name = line.strip()
                    if proper_name:
                        canonical = self.to_canonical(proper_name)
                        self._canonical_to_proper[canonical] = proper_name
                        self._proper_to_canonical[proper_name] = canonical
                        
            logger.info(f"Loaded {len(self._canonical_to_proper)} font name mappings")
            
        except Exception as e:
            logger.error(f"Error loading font mappings: {e}")
            
    def _load_category_mappings(self):
        """Load category mappings from font_class_to_category.json."""
        if not os.path.exists(self.category_mapping_path):
            logger.warning(f"Category mapping file not found: {self.category_mapping_path}")
            return
            
        try:
            with open(self.category_mapping_path, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
                
            # Normalize the category mapping keys to canonical format
            for proper_name, category in raw_mapping.items():
                canonical = self.to_canonical(proper_name)
                self._category_mapping[canonical] = category
                
            logger.info(f"Loaded {len(self._category_mapping)} category mappings")
            
        except Exception as e:
            logger.error(f"Error loading category mappings: {e}")
    
    @staticmethod
    def to_canonical(font_name: str) -> str:
        """
        Convert any font name format to canonical format (lowercase with underscores).
        
        Args:
            font_name: Font name in any format
            
        Returns:
            Canonical format: lowercase with underscores
            
        Examples:
            "Abril Fatface" -> "abril_fatface"
            "abril fatface" -> "abril_fatface"  
            "abril_fatface" -> "abril_fatface"
        """
        if not font_name:
            return ""
        return font_name.replace(' ', '_').lower().strip()
    
    def to_proper(self, font_name: str) -> str:
        """
        Convert any font name format to proper capitalization.
        
        Args:
            font_name: Font name in any format
            
        Returns:
            Proper capitalization format or original if not found
            
        Examples:
            "abril_fatface" -> "Abril Fatface"
            "abril fatface" -> "Abril Fatface"
        """
        canonical = self.to_canonical(font_name)
        return self._canonical_to_proper.get(canonical, font_name)
    
    def to_spaces_lowercase(self, font_name: str) -> str:
        """
        Convert any font name format to lowercase with spaces (JavaScript format).
        
        Args:
            font_name: Font name in any format
            
        Returns:
            Lowercase with spaces format
            
        Examples:
            "Abril Fatface" -> "abril fatface"
            "abril_fatface" -> "abril fatface"
        """
        canonical = self.to_canonical(font_name)
        return canonical.replace('_', ' ')
    
    def to_google_fonts_url(self, font_name: str) -> str:
        """
        Convert font name to Google Fonts URL format (spaces to plus signs).
        
        Args:
            font_name: Font name in any format
            
        Returns:
            Google Fonts URL format
            
        Examples:
            "Abril Fatface" -> "Abril+Fatface"
        """
        proper = self.to_proper(font_name)
        return proper.replace(' ', '+')
    
    def get_category(self, font_name: str) -> Optional[str]:
        """
        Get the category for a font name.
        
        Args:
            font_name: Font name in any format
            
        Returns:
            Category string or None if not found
        """
        canonical = self.to_canonical(font_name)
        return self._category_mapping.get(canonical)
    
    def get_canonical_fonts(self) -> Set[str]:
        """Get all available font names in canonical format."""
        return set(self._canonical_to_proper.keys())
    
    def get_proper_fonts(self) -> Set[str]:
        """Get all available font names in proper format."""
        return set(self._canonical_to_proper.values())
    
    def is_valid_font(self, font_name: str) -> bool:
        """Check if a font name is valid (exists in available fonts)."""
        canonical = self.to_canonical(font_name)
        return canonical in self._canonical_to_proper
    
    def debug_lookup(self, font_name: str) -> Dict[str, str]:
        """
        Debug a font name lookup to see all formats and mappings.
        
        Args:
            font_name: Font name to debug
            
        Returns:
            Dictionary with all format variants and lookup results
        """
        canonical = self.to_canonical(font_name)
        
        return {
            'input': font_name,
            'canonical': canonical,
            'proper': self.to_proper(font_name),
            'spaces_lowercase': self.to_spaces_lowercase(font_name),
            'google_fonts_url': self.to_google_fonts_url(font_name),
            'category': self.get_category(font_name),
            'is_valid': self.is_valid_font(font_name),
            'in_canonical_map': canonical in self._canonical_to_proper,
            'in_category_map': canonical in self._category_mapping,
        }


# Global instance for easy access
_global_normalizer: Optional[FontNameNormalizer] = None

def get_normalizer() -> FontNameNormalizer:
    """Get the global font name normalizer instance."""
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = FontNameNormalizer()
    return _global_normalizer

# Convenience functions for common operations
def normalize_font_name(font_name: str) -> str:
    """Convert font name to canonical format."""
    return get_normalizer().to_canonical(font_name)

def get_proper_font_name(font_name: str) -> str:
    """Convert font name to proper capitalization."""
    return get_normalizer().to_proper(font_name)

def get_font_category(font_name: str) -> Optional[str]:
    """Get category for font name."""
    return get_normalizer().get_category(font_name)

def is_valid_font_name(font_name: str) -> bool:
    """Check if font name is valid."""
    return get_normalizer().is_valid_font(font_name)