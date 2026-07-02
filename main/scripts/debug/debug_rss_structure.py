#!/usr/bin/env python3
"""
Debug RSS Structure
Debug the RSS XML structure to understand the issue
"""

import requests
import xml.etree.ElementTree as ET

def debug_rss_structure():
    """Debug RSS XML structure"""
    try:
        print("🔍 Debugging RSS Structure...")
        
        response = requests.get('https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms', timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            
            print(f"Found {len(items)} items")
            
            if items:
                item = items[0]
                print("\nFirst item structure:")
                for child in item:
                    text = child.text[:50] if child.text else "None"
                    print(f"  {child.tag}: {text}...")
                
                print("\nLooking for title and link elements:")
                title_elem = item.find('title')
                link_elem = item.find('link')
                
                print(f"Title element: {title_elem}")
                print(f"Link element: {link_elem}")
                
                if title_elem is not None:
                    print(f"Title text: {title_elem.text}")
                if link_elem is not None:
                    print(f"Link text: {link_elem.text}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_rss_structure()
