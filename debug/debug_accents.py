#!/usr/bin/env python3
"""
Debug Accent Normalization
"""

import unicodedata

def test_accent_normalization():
    print("🔤 TESTING ACCENT NORMALIZATION")
    print("=" * 40)
    
    def normalize_team_name(team: str) -> str:
        """Normalize team name by removing accents and common football terms"""
        print(f"Input: '{team}'")
        
        # Remove accents using Unicode normalization
        normalized = unicodedata.normalize('NFD', team.lower().strip())
        print(f"After NFD: '{normalized}'")
        
        ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        print(f"After removing diacritics: '{ascii_name}'")
        
        # Additional manual accent replacements for common cases
        accent_map = {
            'ə': 'e', 'ğ': 'g', 'ş': 's', 'ı': 'i', 'ç': 'c', 'ö': 'o', 'ü': 'u',
            'â': 'a', 'î': 'i', 'û': 'u', 'ê': 'e', 'ô': 'o'
        }
        
        for accented, plain in accent_map.items():
            if accented in ascii_name:
                print(f"Replacing '{accented}' with '{plain}'")
                ascii_name = ascii_name.replace(accented, plain)
        
        print(f"After manual replacements: '{ascii_name}'")
        
        # Remove common football terms
        common_terms = ['fk', 'fc', 'united', 'city', 'town', 'football', 'club', 'sporting', 
                       'athletic', 'rovers', 'wanderers', 'belediyespor', 'belediye', 'spor',
                       'futbol', 'kulübü', '76', 'sc', 'ac', 'cf', 'afc', 'fbc']
        
        words = [word for word in ascii_name.split() if word not in common_terms and len(word) > 1]
        
        result = ' '.join(words) if words else ascii_name
        print(f"Final result: '{result}'")
        return result
    
    # Test the problematic case
    test_cases = [
        ('Mingəçevir', 'Expected: mingechevir or similar'),
        ('Mingachevir', 'Expected: mingachevir'),
        ('Iğdır', 'Expected: igdir'),
        ('76 Iğdır Belediyespor', 'Expected: igdir')
    ]
    
    for team, expected in test_cases:
        print(f"\n--- Testing: {team} ---")
        print(f"{expected}")
        normalize_team_name(team)
        print()
    
    # Test character analysis
    print("\n📊 CHARACTER ANALYSIS FOR 'ə'")
    print("-" * 30)
    char = 'ə'
    print(f"Character: {char}")
    print(f"Unicode name: {unicodedata.name(char)}")
    print(f"Unicode category: {unicodedata.category(char)}")
    print(f"NFD normalized: {repr(unicodedata.normalize('NFD', char))}")

if __name__ == "__main__":
    test_accent_normalization()