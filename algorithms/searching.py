"""
Searching algorithms for the algorithm visualizer.
This module implements various searching algorithms with visualization support.
Each algorithm is implemented as a generator function that yields after each step
to allow for visualization of the search process.
"""

import math

def linear_search(data, target, highlight, increment_counters):
    """
    Implementation of linear search algorithm.
    
    Time Complexity:
        Best: O(1) when target is first element
        Average: O(n)
        Worst: O(n) when target is last element or not found
    Space Complexity: O(1) - constant extra space
    
    Args:
        data (list): List of integers to search through
        target (int): Value to search for
        highlight (list): List to store search state and indices:
            - highlight[0]: Current element being checked
            - highlight[1]: Not used in linear search
            - highlight[2]: Search state ("search", "found", "not_found")
            - highlight[3]: List of recently checked indices
        increment_counters (function): Function to increment comparison counter
    
    Yields:
        None: After each comparison for visualization
    """
    found = False
    trail = []
    
    for i in range(len(data)):
        increment_counters(comparison=True)
        highlight[0] = i
        highlight[1] = -1
        highlight[2] = "search"
        highlight[3] = trail[-5:]  # Show last 5 checked elements
        yield
        if data[i] == target:
            highlight[0] = i
            highlight[1] = -1
            highlight[2] = "found"
            highlight[3] = []
            for _ in range(5):  # Blink effect on found element
                yield
            found = True
            break
        trail.append(i)
    
    if not found:
        highlight[0] = -1
        highlight[1] = -1
        highlight[2] = "not_found"
        highlight[3] = []
        for _ in range(3):  # Red flash for not found
            yield

def binary_search(data, target, highlight, increment_counters):
    """
    Implementation of binary search algorithm.
    
    Time Complexity:
        Best: O(1) when target is middle element
        Average: O(log n)
        Worst: O(log n) when target is at either end
    Space Complexity: O(1) - iterative implementation
        Note: Recursive implementation would use O(log n) for call stack
    
    Args:
        data (list): Sorted list of integers to search through
        target (int): Value to search for
        highlight (list): List to store search state and indices:
            - highlight[0]: Current middle element being checked
            - highlight[1]: Not used in binary search
            - highlight[2]: Search state ("search", "found", "not_found")
            - highlight[3]: List of indices in current search range
        increment_counters (function): Function to increment comparison counter
    
    Yields:
        None: After each comparison for visualization
    """
    data.sort()  # Ensure data is sorted
    low = 0
    high = len(data) - 1
    found = False
    search_range = []
    
    while low <= high:
        mid = (low + high) // 2
        search_range = list(range(low, high+1))
        highlight[0] = mid
        highlight[1] = -1
        highlight[2] = "search"
        highlight[3] = search_range
        increment_counters(comparison=True)
        yield
        
        if data[mid] == target:
            highlight[0] = mid
            highlight[1] = -1
            highlight[2] = "found"
            highlight[3] = []
            for _ in range(5):  # Success animation
                yield
            found = True
            break
        elif data[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    if not found:
        highlight[0] = -1
        highlight[1] = -1
        highlight[2] = "not_found"
        highlight[3] = []
        for _ in range(3):  # Failure animation
            yield

