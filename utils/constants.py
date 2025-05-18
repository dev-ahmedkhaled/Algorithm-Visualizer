"""
Constants and configuration settings for the algorithm visualizer.
This module contains all the global settings, colors, and algorithm categories.
"""

import pyray as pr
from screeninfo import get_monitors

# Get primary monitor dimensions
monitor_details = get_monitors()
SCREEN_WIDTH = monitor_details[0].width
SCREEN_HEIGHT = monitor_details[0].height

# Array visualization settings
ARRAY_SIZE = 50  # Number of elements in the array
BAR_WIDTH = 20   # Width of each bar in pixels
BAR_SPACING = 2  # Space between bars in pixels

# Color definitions for visualization
custom_dark_gray = pr.Color(40, 40, 40, 255)    # Background color
custom_light_gray = pr.Color(100, 100, 100, 255) # UI elements
custom_blue = pr.Color(0, 121, 241, 255)        # Default node/bar color
custom_white = pr.Color(255, 255, 255, 255)     # Text color
custom_green = pr.Color(0, 228, 48, 255)        # Success/current node color
custom_red = pr.Color(230, 41, 55, 255)         # Error/visited node color
custom_yellow = pr.Color(255, 203, 0, 255)      # Highlight color
custom_purple = pr.Color(200, 122, 255, 255)    # Special state color

# Layout constants for visualization
HORIZONTAL_PADDING = 20  # Padding from screen edges
MIN_BAR_WIDTH = 8       # Minimum width of array bars

# Algorithm categories and their implementations
CATEGORIES = {
    "Sorting": [
        "BUBBLE_SORT",
        "INSERTION_SORT",
        "SELECTION_SORT",
        "QUICK_SORT",
        "MERGE_SORT",
        "HEAP_SORT"
    ],
    "Searching": [
        "LINEAR_SEARCH",
        "BINARY_SEARCH"
    ],
    "Trees": [
        "BINARY_TREE"
    ],
    "Graph": [
        "BFS",
        "DFS",
        "DIJKSTRA",
        "A_STAR"
    ]
}

# Time and space complexity information for each algorithm
COMPLEXITY_INFO = {
    "BUBBLE_SORT": {
        "time": {
            "best": "O(n)",
            "average": "O(n²)",
            "worst": "O(n²)"
        },
        "space": "O(1)"
    },
    "INSERTION_SORT": {
        "time": {
            "best": "O(n)",
            "average": "O(n²)",
            "worst": "O(n²)"
        },
        "space": "O(1)"
    },
    "SELECTION_SORT": {
        "time": {
            "best": "O(n²)",
            "average": "O(n²)",
            "worst": "O(n²)"
        },
        "space": "O(1)"
    },
    "QUICK_SORT": {
        "time": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n²)"
        },
        "space": "O(log n)"
    },
    "MERGE_SORT": {
        "time": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)"
        },
        "space": "O(n)"
    },
    "HEAP_SORT": {
        "time": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n)"
        },
        "space": "O(1)"
    },
    "LINEAR_SEARCH": {
        "time": {
            "best": "O(1)",
            "average": "O(n)",
            "worst": "O(n)"
        },
        "space": "O(1)"
    },
    "BINARY_SEARCH": {
        "time": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(log n)"
        },
        "space": "O(1)"
    },
    "BINARY_TREE": {
        "time": {
            "best": "O(1)",
            "average": "O(log n)",
            "worst": "O(n)"
        },
        "space": "O(1)"
    },
    "BFS": {
        "time": {
            "best": "O(1)",
            "average": "O(V + E)",
            "worst": "O(V + E)"
        },
        "space": "O(V)"
    },
    "DFS": {
        "time": {
            "best": "O(1)",
            "average": "O(V + E)",
            "worst": "O(V + E)"
        },
        "space": "O(V)"
    },
    "DIJKSTRA": {
        "time": {
            "best": "O(1)",
            "average": "O((V + E)log V)",
            "worst": "O((V + E)log V)"
        },
        "space": "O(V)"
    },
    "A_STAR": {
        "time": {
            "best": "O(1)",
            "average": "O(E log V)",
            "worst": "O(E log V)"
        },
        "space": "O(V)"
    }
} 