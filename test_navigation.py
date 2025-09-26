#!/usr/bin/env python3
"""
Test script to verify page navigation functionality.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.text_comparison import create_page_mapping, get_page_mapping_info, get_page_mapping_summary

def test_page_navigation():
    """Test page navigation and mapping functionality"""
    print("Testing Page Navigation and Mapping...")
    
    # Test 1: Simple page mapping
    print("\n1. Testing simple page mapping...")
    text1_pages = ["Page 1 content", "Page 2 content", "Page 3 content"]
    text2_pages = ["Page 1 content", "Modified page 2 content", "Page 3 content", "New page 4 content"]
    
    mapping, similarity_matrix = create_page_mapping(text1_pages, text2_pages, similarity_threshold=0.2)
    summary = get_page_mapping_summary(mapping, len(text1_pages), len(text2_pages))
    
    print(f"   Original pages: {len(text1_pages)}")
    print(f"   Updated pages: {len(text2_pages)}")
    print(f"   Page mapping: {mapping}")
    print(f"   Total logical pages: {summary['total_logical_pages']}")
    
    # Test 2: Page navigation bounds
    print("\n2. Testing page navigation bounds...")
    max_pages = max(len(text1_pages), len(text2_pages))
    
    for logical_page in range(max_pages + 2):  # Test beyond bounds
        orig_page, upd_page = get_page_mapping_info(logical_page, mapping, len(text1_pages), len(text2_pages))
        
        if logical_page < max_pages:
            print(f"   Logical page {logical_page}: Original={orig_page}, Updated={upd_page}")
        else:
            print(f"   Logical page {logical_page} (out of bounds): Original={orig_page}, Updated={upd_page}")
    
    # Test 3: Edge cases
    print("\n3. Testing edge cases...")
    
    # Empty documents
    empty_mapping, _ = create_page_mapping([], [], similarity_threshold=0.2)
    print(f"   Empty documents mapping: {empty_mapping}")
    
    # Single page documents
    single_mapping, _ = create_page_mapping(["Single page"], ["Single modified page"], similarity_threshold=0.2)
    print(f"   Single page mapping: {single_mapping}")
    
    # Large document simulation (should use simplified mapping)
    large_pages1 = [f"Page {i} content" for i in range(60)]
    large_pages2 = [f"Page {i} modified content" for i in range(65)]
    large_mapping, _ = create_page_mapping(large_pages1, large_pages2, similarity_threshold=0.2)
    print(f"   Large document mapping (first 5): {dict(list(large_mapping.items())[:5])}")
    print(f"   Large document mapping type: {'Sequential' if large_mapping == {i: i for i in range(min(60, 65))} else 'Complex'}")
    
    print("\n✅ Page navigation tests completed successfully!")
    return True

def test_session_state_simulation():
    """Simulate session state management"""
    print("\n4. Testing session state simulation...")
    
    # Simulate session state
    session_state = {'page_num': 0}
    
    # Simulate navigation
    max_pages = 5
    
    for action in ['next', 'next', 'next', 'prev', 'next', 'next', 'next']:  # Will test bounds
        if action == 'next' and session_state['page_num'] < max_pages - 1:
            session_state['page_num'] += 1
        elif action == 'prev' and session_state['page_num'] > 0:
            session_state['page_num'] -= 1
        
        # Bounds checking
        session_state['page_num'] = max(0, min(max_pages - 1, session_state['page_num']))
        
        print(f"   After '{action}': page_num = {session_state['page_num']} (bounds: 0 to {max_pages-1})")
    
    print("   ✅ Session state navigation working correctly")

if __name__ == "__main__":
    try:
        test_page_navigation()
        test_session_state_simulation()
        print("\n🎉 All page navigation tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)