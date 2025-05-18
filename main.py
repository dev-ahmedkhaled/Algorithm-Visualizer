"""
Main module for the Algorithm Visualizer application.
This module handles the main game loop, user input, and coordinates the visualization
of various algorithms including sorting, searching, tree operations, and graph algorithms.
"""

import pyray as pr
import random
import time
from utils.constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, CATEGORIES,
    custom_dark_gray, custom_light_gray, custom_blue, custom_white,
    custom_green, custom_red, custom_yellow, custom_purple
)
from utils.data_structures import TreeNode
from algorithms.sorting import (
    bubble_sort, insertion_sort, selection_sort,
    quick_sort, merge_sort, heap_sort
)
from algorithms.searching import (
    linear_search, binary_search
)
from algorithms.tree import (
    create_random_tree, insert_node, delete_node, search_node,
    calculate_tree_positions
)
from algorithms.graph import (
    create_random_graph, breadth_first_search, depth_first_search,
    dijkstra_algorithm, a_star_algorithm
)
from visualization.draw import (
    draw_array_bars, draw_tree, draw_ui_elements
)

def main():
    """
    Main function that initializes and runs the algorithm visualizer.
    Handles window creation, state management, user input, and visualization updates.
    """
    # Initialize window
    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Algorithm Visualizer")
    pr.set_target_fps(60)
    
    # Initialize state
    categories = list(CATEGORIES.keys())
    current_category_idx = 0
    current_alg_idx = 0
    running = False
    reset = True
    delay = 0.2  
    min_delay = 0.05  
    max_delay = 1.0  
    
    # Data state for array-based algorithms (sorting and searching)
    data = []
    highlight = [0, 0, None, None]  # [current, next, state, trail]
    steps_counter = 0
    comparison_counter = 0
    swap_counter = 0
    target = None
    
    # Algorithm state
    current_algorithm = None
    
    # Tree state
    tree_root = None
    tree_nodes = []
    tree_visualization_mode = None
    tree_current_node = None
    tree_path = []
    tree_selected_node = None
    tree_operation = None  # 'insert', 'delete', 'search', or None
    
    # Graph state
    graph_nodes = []
    graph_edges = []
    graph_start_node = None
    graph_end_node = None
    selecting_start = True
    
    def increment_counters(comparison=False, swap=False):
        """
        Increment the comparison and swap counters.
        
        Args:
            comparison (bool): Whether to increment comparison counter
            swap (bool): Whether to increment swap counter
        """
        nonlocal comparison_counter, swap_counter
        if comparison:
            comparison_counter += 1
        if swap:
            swap_counter += 1
    
    def reset_data():
        """
        Reset all visualization data based on the current category and algorithm.
        This includes initializing arrays, trees, or graphs as needed.
        """
        nonlocal data, highlight, steps_counter, comparison_counter, swap_counter, target
        nonlocal tree_root, tree_nodes, tree_visualization_mode, tree_current_node, tree_path
        nonlocal tree_selected_node, tree_operation
        nonlocal graph_nodes, graph_edges, graph_start_node, graph_end_node
        nonlocal current_algorithm
        
        category = categories[current_category_idx]
        algorithm = CATEGORIES[category][current_alg_idx]
        
        if category == "Sorting":
            data = [random.randint(50, SCREEN_HEIGHT - 100) for _ in range(50)]
            highlight = [0, 0, None, None]
            steps_counter = 0
            comparison_counter = 0
            swap_counter = 0
            current_algorithm = None
        elif category == "Searching":
            data = sorted([random.randint(50, SCREEN_HEIGHT - 100) for _ in range(50)])
            highlight = [0, 0, None, None]
            steps_counter = 0
            comparison_counter = 0
            swap_counter = 0
            target = random.choice(data)
            current_algorithm = None
        elif category == "Trees":
            tree_root, tree_nodes = create_random_tree(10, SCREEN_WIDTH, SCREEN_HEIGHT)
            tree_visualization_mode = None
            tree_current_node = None
            tree_path = []
            tree_selected_node = None
            tree_operation = None
            current_algorithm = None
        elif category == "Graph":
            graph_nodes, graph_edges = create_random_graph(15, 0.3, SCREEN_WIDTH, SCREEN_HEIGHT)
            # Initialize node colors
            for node in graph_nodes:
                node.color = custom_blue
            graph_start_node = random.choice(range(len(graph_nodes)))
            graph_end_node = random.choice([i for i in range(len(graph_nodes)) if i != graph_start_node])
            current_algorithm = None
    
    def get_algorithm_generator():
        """
        Get the appropriate algorithm generator based on current category and algorithm.
        
        Returns:
            generator: The algorithm generator function
        """
        category = categories[current_category_idx]
        algorithm = CATEGORIES[category][current_alg_idx]
        
        if category == "Sorting":
            if algorithm == "BUBBLE_SORT":
                return bubble_sort(data, highlight, increment_counters)
            elif algorithm == "INSERTION_SORT":
                return insertion_sort(data, highlight, increment_counters)
            elif algorithm == "SELECTION_SORT":
                return selection_sort(data, highlight, increment_counters)
            elif algorithm == "QUICK_SORT":
                return quick_sort(data, highlight, increment_counters)
            elif algorithm == "MERGE_SORT":
                return merge_sort(data, highlight, increment_counters)
            elif algorithm == "HEAP_SORT":
                return heap_sort(data, highlight, increment_counters)
        elif category == "Searching":
            if algorithm == "LINEAR_SEARCH":
                return linear_search(data, target, highlight, increment_counters)
            elif algorithm == "BINARY_SEARCH":
                return binary_search(data, target, highlight, increment_counters)
        elif category == "Graph":
            if algorithm == "BFS":
                return breadth_first_search(graph_nodes, graph_start_node, graph_end_node)
            elif algorithm == "DFS":
                return depth_first_search(graph_nodes, graph_start_node, graph_end_node)
            elif algorithm == "DIJKSTRA":
                return dijkstra_algorithm(graph_nodes, graph_start_node, graph_end_node)
            elif algorithm == "A_STAR":
                return a_star_algorithm(graph_nodes, graph_start_node, graph_end_node)
        return None
    
    reset_data()
    
    # Main game loop
    while not pr.window_should_close():
        # Handle input
        if pr.is_key_pressed(pr.KeyboardKey.KEY_RIGHT):
            current_alg_idx = (current_alg_idx + 1) % len(CATEGORIES[categories[current_category_idx]])
            reset_data()
        elif pr.is_key_pressed(pr.KeyboardKey.KEY_LEFT):
            current_alg_idx = (current_alg_idx - 1) % len(CATEGORIES[categories[current_category_idx]])
            reset_data()
        elif pr.is_key_pressed(pr.KeyboardKey.KEY_UP):
            current_category_idx = (current_category_idx - 1) % len(categories)
            current_alg_idx = 0
            reset_data()
        elif pr.is_key_pressed(pr.KeyboardKey.KEY_DOWN):
            current_category_idx = (current_category_idx + 1) % len(categories)
            current_alg_idx = 0
            reset_data()
        elif pr.is_key_pressed(pr.KeyboardKey.KEY_SPACE):
            running = not running
            if running and current_algorithm is None:
                current_algorithm = get_algorithm_generator()
        elif pr.is_key_pressed(pr.KeyboardKey.KEY_R):
            reset_data()
        elif pr.is_key_pressed(pr.KeyboardKey.KEY_EQUAL):
            delay = max(delay - 0.1, min_delay)  # Changed to decrease delay (speed up)
        elif pr.is_key_pressed(pr.KeyboardKey.KEY_MINUS):
            delay = min(delay + 0.1, max_delay)  # Changed to increase delay (slow down)
        
        current_time = time.time()
        
        # Handle category-specific input
        category = categories[current_category_idx]
        algorithm = CATEGORIES[category][current_alg_idx]
        
        if category == "Searching" and pr.is_key_pressed(pr.KeyboardKey.KEY_T):
            target = random.choice(data)
            highlight = [0, 0, None, None]
            current_algorithm = None
        elif category == "Trees":
            if algorithm == "BINARY_TREE":
                if pr.is_key_pressed(pr.KeyboardKey.KEY_I):
                    value = random.randint(1, 100)
                    tree_root, tree_nodes, tree_path = insert_node(tree_root, value, tree_nodes)
                    tree_visualization_mode = "insertion"
                    tree_current_node = tree_path[-1] if tree_path else None
                    if tree_root:
                        calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)
                elif pr.is_key_pressed(pr.KeyboardKey.KEY_D):
                    if tree_nodes:
                        node = random.choice(tree_nodes)
                        tree_root, tree_nodes, tree_path = delete_node(tree_root, node.value, tree_nodes)
                        tree_visualization_mode = "deletion"
                        tree_current_node = node
                        if tree_root:
                            calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)
                elif pr.is_key_pressed(pr.KeyboardKey.KEY_S):
                    if tree_nodes:
                        node = random.choice(tree_nodes)
                        tree_current_node, tree_path = search_node(tree_root, node.value)
                        tree_visualization_mode = "search"
        elif category == "Graph":
            if pr.is_key_pressed(pr.KeyboardKey.KEY_S):
                # Reset all node colors to blue
                for node in graph_nodes:
                    node.color = custom_blue
                graph_start_node = -1
                graph_end_node = -1
                selecting_start = True
                current_algorithm = None
            elif pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_LEFT):
                mouse_x = pr.get_mouse_x()
                mouse_y = pr.get_mouse_y()
                
                # Check if clicked on a node
                for i, node in enumerate(graph_nodes):
                    if ((mouse_x - node.x) ** 2 + (mouse_y - node.y) ** 2) <= 400:  # 20^2 = 400
                        if selecting_start:
                            # Reset previous start node color
                            if graph_start_node is not None and graph_start_node != -1:
                                graph_nodes[graph_start_node].color = custom_blue
                            graph_start_node = i
                            graph_nodes[i].color = custom_green
                            selecting_start = False
                        else:
                            # Reset previous end node color
                            if graph_end_node is not None and graph_end_node != -1:
                                graph_nodes[graph_end_node].color = custom_blue
                            graph_end_node = i
                            graph_nodes[i].color = custom_red
                            selecting_start = True
                        break
        
        # Handle mouse clicks for tree node selection
        elif category == "Trees" and not running and tree_operation is not None:
            if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_LEFT):
                mouse_x = pr.get_mouse_x()
                mouse_y = pr.get_mouse_y()
                
                # Check if clicked on a node
                for node in tree_nodes:
                    if ((mouse_x - node.x) ** 2 + (mouse_y - node.y) ** 2) <= 400:  # 20^2 = 400
                        tree_selected_node = node
                        
                        # Perform the selected operation
                        if tree_operation == 'insert':
                            value = random.randint(1, 100)
                            tree_root, tree_nodes, tree_path = insert_node(tree_root, value, tree_nodes)
                            tree_visualization_mode = "insertion"
                            tree_current_node = tree_path[-1] if tree_path else None
                        elif tree_operation == 'delete':
                            tree_root, tree_nodes, tree_path = delete_node(tree_root, node.value, tree_nodes)
                            tree_visualization_mode = "deletion"
                            tree_current_node = node
                        elif tree_operation == 'search':
                            tree_current_node, tree_path = search_node(tree_root, node.value)
                            tree_visualization_mode = "search"
                        
                        tree_operation = None
                        break
        
        # Update
        if running and current_algorithm is not None:
            try:
                next(current_algorithm)
                time.sleep(delay)  # Add delay between algorithm steps
            except StopIteration:
                current_algorithm = None
                running = False
        
        # Draw
        pr.begin_drawing()
        pr.clear_background(custom_dark_gray)
        
        if category == "Sorting" or category == "Searching":
            draw_array_bars(data, highlight)
        elif category == "Trees":
            draw_tree(tree_root, tree_nodes, tree_visualization_mode, tree_current_node, tree_path)
            
            # Draw operation instructions
            if tree_operation is not None:
                operation_text = f"Select a node to {tree_operation}"
                pr.draw_text(operation_text, 10, SCREEN_HEIGHT - 30, 20, custom_white)
            
            # Draw selected node highlight
            if tree_selected_node is not None:
                pr.draw_circle(tree_selected_node.x, tree_selected_node.y, 25, custom_yellow)
        elif category == "Graph":
            # Draw edges
            for start, end, weight in graph_edges:
                start_node = graph_nodes[start]
                end_node = graph_nodes[end]
                pr.draw_line(start_node.x, start_node.y, end_node.x, end_node.y, custom_light_gray)
                # Draw weight
                mid_x = (start_node.x + end_node.x) // 2
                mid_y = (start_node.y + end_node.y) // 2
                pr.draw_text(str(weight), mid_x, mid_y, 20, custom_white)
            
            # Draw nodes
            for node in graph_nodes:
                pr.draw_circle(node.x, node.y, 20, node.color)
                pr.draw_text(str(node.id), node.x - 5, node.y - 5, 20, custom_white)
            
            # Draw start and end nodes
            if graph_start_node is not None:
                pr.draw_circle(graph_nodes[graph_start_node].x, graph_nodes[graph_start_node].y, 25, custom_green)
            if graph_end_node is not None:
                pr.draw_circle(graph_nodes[graph_end_node].x, graph_nodes[graph_end_node].y, 25, custom_red)
        
        draw_ui_elements(categories, current_category_idx, current_alg_idx, category == "Sorting", steps_counter, comparison_counter, swap_counter)
        
        if category == "Searching" and target is not None:
            target_text = f"Target: {target}"
            pr.draw_rectangle(10, 110, 150, 30, custom_light_gray)
            pr.draw_text(target_text, 20, 115, 20, custom_white)
        
        pr.end_drawing()
        
    pr.close_window()

if __name__ == "__main__":
    main()