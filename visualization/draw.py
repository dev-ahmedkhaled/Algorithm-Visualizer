import pyray as pr
from utils.constants import (
    custom_dark_gray, custom_light_gray, custom_blue, custom_white,
    custom_green, custom_red, custom_yellow, custom_purple,
    HORIZONTAL_PADDING, MIN_BAR_WIDTH, BAR_SPACING, SCREEN_WIDTH, SCREEN_HEIGHT,
    CATEGORIES
)

def draw_array_bars(data, highlight):
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
        elif tree_visualization_mode == "traversal":
            if node in tree_path:
                path_index = tree_path.index(node)
                # Gradual color change based on position in path
                color_factor = path_index / max(1, len(tree_path) - 1)
                r = int(custom_blue.r + (custom_green.r - custom_blue.r) * color_factor)
                g = int(custom_blue.g + (custom_green.g - custom_blue.g) * color_factor)
                b = int(custom_blue.b + (custom_green.b - custom_blue.b) * color_factor)
                node_color = pr.Color(r, g, b, 255)
        
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
    
    # Draw counters for sorting algorithms
    if category == "Sorting" and sorting:
        counter_text = f"Steps: {steps_counter} | Comparisons: {comparison_counter} | Swaps: {swap_counter}"
        counter_size = 20
        counter_width = pr.measure_text(counter_text, counter_size)
        pr.draw_rectangle(10, 70, counter_width + 20, 30, custom_light_gray)
        pr.draw_text(
            counter_text, 
            20, 
            75, 
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
        elif algorithm in ["INORDER", "PREORDER", "POSTORDER", "LEVEL_ORDER"]:
            controls_text += " | S - Start Traversal"
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