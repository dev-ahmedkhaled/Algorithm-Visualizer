"""
Visualization functions for the algorithm visualizer.
This module handles all the drawing operations for different data structures and algorithms.
"""

import pyray as pr
from utils.constants import (
    custom_dark_gray, custom_light_gray, custom_blue, custom_white,
    custom_green, custom_red, custom_yellow, custom_purple,
    HORIZONTAL_PADDING, MIN_BAR_WIDTH, BAR_SPACING, SCREEN_WIDTH, SCREEN_HEIGHT,
    CATEGORIES, COMPLEXITY_INFO
)

def draw_array_bars(data, highlight):
    """
    Draw rectangles representing array elements with different colors based on their state.
    
    Args:
        data (list): List of integer values representing heights of bars
        highlight (tuple): Tuple containing highlighting information:
            - highlight[0] (int): Index of first highlighted element (red)
            - highlight[1] (int): Index of second highlighted element (green)
            - highlight[2] (str): Special state indicator:
                - "search": Search operation in progress
                - "found": Element was found
                - "not_found": Element was not found
            - highlight[3] (list): List of indices in search range when highlight[2]=="search"
    
    Color Scheme:
        - Red: First highlighted element
        - Green: Second highlighted element or found element
        - Yellow: Elements in search range
        - Blue: Default color for non-highlighted elements
    """
    screen_width = SCREEN_WIDTH - 2 * HORIZONTAL_PADDING
    bar_width = max(MIN_BAR_WIDTH, (screen_width / len(data)) - BAR_SPACING)
    
    for i, height in enumerate(data):
        x = HORIZONTAL_PADDING + i * (bar_width + BAR_SPACING)
        
        # Different colors for highlighted elements
        if i == highlight[0]:
            color = custom_red
        elif i == highlight[1]:
            color = custom_green
        elif len(highlight) > 2 and highlight[2] == "search":
            if len(highlight) > 3 and i in highlight[3]:
                color = custom_yellow  # Search range
            else:
                color = custom_blue
        elif len(highlight) > 2 and highlight[2] == "found" and i == highlight[0]:
            color = custom_green  # Found element
        elif len(highlight) > 2 and highlight[2] == "not_found":
            color = custom_blue
        else:
            color = custom_blue
            
        pr.draw_rectangle(int(x), SCREEN_HEIGHT - height, int(bar_width), height, color)

def draw_tree(tree_root, tree_nodes, tree_visualization_mode, tree_current_node, tree_path):
    """
    Visualize a binary tree structure with various visualization modes and color-coding.
    
    Args:
        tree_root (Node): The root node of the binary tree
        tree_nodes (list): List of all nodes in the tree
        tree_visualization_mode (str): Current visualization mode:
            - "insertion": Node insertion in progress
            - "deletion": Node deletion in progress
            - "search": Node search in progress
        tree_current_node (Node): Currently selected/active node
        tree_path (list): List of nodes in the current path being visualized
    
    Color Scheme:
        - Blue: Default node color
        - Yellow: Nodes in path during insertion/search
        - Red: Nodes in path during deletion
        - Green: Current node
    """
    if not tree_root:
        return
        
    # Draw edges first (so they appear behind nodes)
    for node in tree_nodes:
        if node.left:
            pr.draw_line(
                int(node.x), int(node.y), 
                int(node.left.x), int(node.left.y), 
                custom_blue
            )
        if node.right:
            pr.draw_line(
                int(node.x), int(node.y), 
                int(node.right.x), int(node.right.y), 
                custom_blue
            )
    
    # Then draw nodes
    for node in tree_nodes:
        node_color = custom_blue
        
        # Color based on visualization mode
        if tree_visualization_mode == "insertion" and node in tree_path:
            node_color = custom_yellow
        elif tree_visualization_mode == "deletion" and node in tree_path:
            node_color = custom_red
        elif tree_visualization_mode == "search" and node in tree_path:
            node_color = custom_yellow
        
        # Special cases
        if node == tree_current_node:
            node_color = custom_green
        
        # For Red-Black trees
        if hasattr(node, 'color') and node.color == "RED":
            node_color = custom_red
        
        # Draw the node
        pr.draw_circle(int(node.x), int(node.y), 25, node_color)
        pr.draw_circle(int(node.x), int(node.y), 26, custom_dark_gray)  # Border
        pr.draw_circle(int(node.x), int(node.y), 25, node_color)  # Node
        
        # Draw the value
        text_size = 20
        text_width = pr.measure_text(str(node.value), text_size)
        pr.draw_text(
            str(node.value), 
            int(node.x - text_width / 2), 
            int(node.y - text_size / 2), 
            text_size, 
            custom_white
        )

def draw_graph(graph_nodes, graph_edges, graph_selected_node, graph_start_node, graph_end_node):
    """
    Visualize a graph structure with nodes, edges, and weights.
    
    Args:
        graph_nodes (list): List of GraphNode objects
        graph_edges (list): List of (start_id, end_id, weight) tuples
        graph_selected_node (int): ID of currently selected node
        graph_start_node (int): ID of start node for path finding
        graph_end_node (int): ID of end node for path finding
    
    Color Scheme:
        - Blue: Default node color
        - Green: Start node
        - Red: End node
        - Purple: Selected node
    """
    # Draw edges first
    for i, j, weight in graph_edges:
        x1, y1 = graph_nodes[i].x, graph_nodes[i].y
        x2, y2 = graph_nodes[j].x, graph_nodes[j].y
        
        # Draw edge
        pr.draw_line(int(x1), int(y1), int(x2), int(y2), custom_white)
        
        # Draw weight
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        text_size = 20
        text_width = pr.measure_text(str(weight), text_size)
        
        # Background for text
        pr.draw_circle(int(mid_x), int(mid_y), text_width/2 + 5, custom_dark_gray)
        
        # Text
        pr.draw_text(
            str(weight), 
            int(mid_x - text_width / 2), 
            int(mid_y - text_size / 2), 
            text_size, 
            custom_white
        )
    
    # Draw nodes
    for i, node in enumerate(graph_nodes):
        node_color = node.color
        
        # Highlight selected/start/end nodes
        if i == graph_selected_node:
            node_color = custom_purple
        if i == graph_start_node:
            node_color = custom_green
        if i == graph_end_node:
            node_color = custom_red
        
        # Draw the node
        pr.draw_circle(int(node.x), int(node.y), 20, node_color)
        pr.draw_circle(int(node.x), int(node.y), 21, custom_dark_gray)  # Border
        pr.draw_circle(int(node.x), int(node.y), 20, node_color)  # Node
        
        # Draw the ID
        text_size = 20
        text_width = pr.measure_text(str(i), text_size)
        pr.draw_text(
            str(i), 
            int(node.x - text_width / 2), 
            int(node.y - text_size / 2), 
            text_size, 
            custom_white
        )

def draw_ui_elements(categories, current_category_idx, current_alg_idx, sorting, steps_counter, comparison_counter, swap_counter):
    """
    Draw the user interface elements including header, controls, and statistics.
    
    Args:
        categories (list): List of available algorithm categories
        current_category_idx (int): Index of current category
        current_alg_idx (int): Index of current algorithm
        sorting (bool): Whether current category is sorting
        steps_counter (int): Number of steps performed
        comparison_counter (int): Number of comparisons performed
        swap_counter (int): Number of swaps performed
    """
    # Draw header
    pr.draw_rectangle(0, 0, SCREEN_WIDTH, 60, custom_dark_gray)
    
    # Draw category title
    category = categories[current_category_idx]
    alg_list = CATEGORIES[category]
    algorithm = alg_list[current_alg_idx]
    
    header_text = f"{category} - {algorithm.replace('_', ' ')}"
    text_size = 30
    text_width = pr.measure_text(header_text, text_size)
    pr.draw_text(
        header_text, 
        int(SCREEN_WIDTH / 2 - text_width / 2), 
        15, 
        text_size, 
        custom_white
    )
    
    # Draw complexity information
    if algorithm in COMPLEXITY_INFO:
        complexity = COMPLEXITY_INFO[algorithm]
        complexity_text = f"Time Complexity: Best: {complexity['time']['best']} | Average: {complexity['time']['average']} | Worst: {complexity['time']['worst']} | Space: {complexity['space']}"
        complexity_size = 18
        complexity_width = pr.measure_text(complexity_text, complexity_size)
        
        # Draw background for complexity info
        pr.draw_rectangle(
            int(SCREEN_WIDTH / 2 - complexity_width / 2 - 10),
            70,
            complexity_width + 20,
            30,
            custom_light_gray
        )
        
        # Draw complexity text
        pr.draw_text(
            complexity_text,
            int(SCREEN_WIDTH / 2 - complexity_width / 2),
            75,
            complexity_size,
            custom_white
        )
    
    # Draw counters for sorting algorithms
    if category == "Sorting" and sorting:
        counter_text = f"Steps: {steps_counter} | Comparisons: {comparison_counter} | Swaps: {swap_counter}"
        counter_size = 20
        counter_width = pr.measure_text(counter_text, counter_size)
        pr.draw_rectangle(10, 110, counter_width + 20, 30, custom_light_gray)
        pr.draw_text(
            counter_text, 
            20, 
            115, 
            counter_size, 
            custom_white
        )
    
    # Draw controls
    controls_text = "Controls: LEFT/RIGHT - Change Algorithm | UP/DOWN - Change Category | SPACE - Start/Stop | R - Reset | +/- - Speed"
    if category == "Searching":
        controls_text += " | T - Set Target"
    elif category == "Trees":
        if algorithm == "BINARY_TREE":
            controls_text += " | I - Insert | D - Delete | S - Search"
    elif category == "Graph":
        controls_text += " | S - Reset Start/End | Click - Select Start/End Node"
    
    controls_size = 18
    controls_width = pr.measure_text(controls_text, controls_size)
    
    pr.draw_rectangle(0, SCREEN_HEIGHT - 30, SCREEN_WIDTH, 30, custom_dark_gray)
    pr.draw_text(
        controls_text, 
        int(SCREEN_WIDTH / 2 - controls_width / 2), 
        SCREEN_HEIGHT - 25, 
        controls_size, 
        custom_white
    ) 