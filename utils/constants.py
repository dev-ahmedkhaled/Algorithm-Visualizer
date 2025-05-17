import pyray as pr

from screeninfo import get_monitors

# Window settings
monitor_details = get_monitors()
SCREEN_WIDTH = monitor_details[0].width
SCREEN_HEIGHT = monitor_details[0].height

# Array settings
ARRAY_SIZE = 50
BAR_WIDTH = 20
BAR_SPACING = 2

# Colors
custom_dark_gray = pr.Color(40, 40, 40, 255)
custom_light_gray = pr.Color(100, 100, 100, 255)
custom_blue = pr.Color(0, 121, 241, 255)
custom_white = pr.Color(255, 255, 255, 255)
custom_green = pr.Color(0, 228, 48, 255)
custom_red = pr.Color(230, 41, 55, 255)
custom_yellow = pr.Color(255, 203, 0, 255)
custom_purple = pr.Color(200, 122, 255, 255)

# Layout constants
HORIZONTAL_PADDING = 20
MIN_BAR_WIDTH = 8

# Algorithm categories
CATEGORIES = {
    "Sorting": ["BUBBLE_SORT", "INSERTION_SORT", "SELECTION_SORT", "QUICK_SORT", "MERGE_SORT", "HEAP_SORT"],
    "Searching": ["LINEAR_SEARCH", "BINARY_SEARCH"],
    "Trees": ["BINARY_TREE", "INORDER", "PREORDER", "POSTORDER", "LEVEL_ORDER"],
    "Graph": ["BFS", "DFS", "DIJKSTRA", "A_STAR"]
} 