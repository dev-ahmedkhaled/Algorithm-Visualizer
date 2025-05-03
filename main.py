import pyray as pr
import random
import math
from screeninfo import get_monitors

monitor_details = get_monitors()

SCREEN_WIDTH = monitor_details[0].width
SCREEN_HEIGHT = monitor_details[0].height
pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, b"Algorithm Visualizer")
cont = pr.get_monitor_count()
fps = pr.get_monitor_refresh_rate(0)
print(f"Monitor count: {cont}")
print(f"Monitor refresh rate: {fps}")
if fps == 0:
    fps = 60 
else:
    fps = int(fps)

pr.set_target_fps(fps)
pr.set_window_state(pr.ConfigFlags.FLAG_FULLSCREEN_MODE)
if cont > 1:
    pr.set_window_position(monitor_details[0].width, monitor_details[0].height)
    pr.set_window_size(monitor_details[0].width, monitor_details[0].height)

# Color palette
custom_dark_gray = pr.Color(34, 40, 49, 255)
custom_light_gray = pr.Color(49, 54, 63, 255)
custom_blue = pr.Color(118, 171, 174, 255)
custom_white = pr.Color(238, 238, 238, 255)
custom_green = pr.Color(106, 190, 138, 255)
custom_red = pr.Color(231, 111, 81, 255)
custom_yellow = pr.Color(233, 196, 106, 255)
custom_purple = pr.Color(150, 111, 214, 255)

HORIZONTAL_PADDING = 20
MIN_BAR_WIDTH = 8
BAR_SPACING = 1

# Algorithm categories
CATEGORIES = {
    "Sorting": ["BUBBLE", "INSERTION", "SELECTION", "QUICK", "MERGE", "HEAP"],
    "Searching": ["LINEAR", "BINARY", "JUMP", "EXPONENTIAL"],
    "Trees": ["BST", "AVL", "RED_BLACK", "TRAVERSALS"],
    "Graph": ["BFS", "DFS", "DIJKSTRA", "A_STAR"]
}
current_category_idx = 0
current_alg_idx = 0
categories = list(CATEGORIES.keys())
sorting = False
data = []
generator = None
highlight = [-1, -1]
speed_factor = 1.0  # Animation speed control
steps_counter = 0
comparison_counter = 0
swap_counter = 0

# Tree visualization variables
tree_root = None
tree_nodes = []
tree_visualization_mode = "normal"  # normal, insertion, deletion, search
tree_current_node = None
tree_path = []

# Graph visualization variables
graph_nodes = []
graph_edges = []
graph_selected_node = -1
graph_start_node = -1
graph_end_node = -1

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  # For AVL tree
        self.color = "RED"  # For Red-Black tree (RED or BLACK)
        self.x = 0  # For visualization
        self.y = 0  # For visualization
        self.parent = None  # For some algorithms

class GraphNode:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.connections = []  # List of (node_id, weight) tuples
        self.color = custom_blue
    
    def connect(self, other_id, weight=1):
        # Don't add duplicate connections
        for conn, _ in self.connections:
            if conn == other_id:
                return
        self.connections.append((other_id, weight))

def reset_data(size=100):
    global data, steps_counter, comparison_counter, swap_counter
    data = [random.randint(50, 500) for _ in range(size)]
    steps_counter = 0
    comparison_counter = 0
    swap_counter = 0

def increment_counters(comparison=False, swap=False):
    global steps_counter, comparison_counter, swap_counter
    steps_counter += 1
    if comparison:
        comparison_counter += 1
    if swap:
        swap_counter += 1

#########################################
# SORTING ALGORITHMS
#########################################

def bubble_sort(): 
    global highlight
    n = len(data)
    for i in range(n):
        for j in range(0, n-i-1):
            highlight = [j, j+1]
            increment_counters(comparison=True)
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                increment_counters(swap=True)
            yield
    highlight = [-1, -1]  

def insertion_sort():
    global highlight
    for i in range(1, len(data)):
        key = data[i]
        j = i-1
        highlight = [j, i]
        yield
        while j >= 0 and key < data[j]:
            highlight = [j, j+1]
            increment_counters(comparison=True)
            data[j + 1] = data[j]
            increment_counters(swap=True)
            j -= 1
            yield
        data[j + 1] = key
        increment_counters(swap=True)
    highlight = [-1, -1]

def selection_sort():
    global highlight
    n = len(data)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            highlight = [j, min_idx]
            increment_counters(comparison=True)
            if data[j] < data[min_idx]:
                min_idx = j
            yield
        data[i], data[min_idx] = data[min_idx], data[i]
        increment_counters(swap=True)
        highlight = [i, min_idx]
        yield
    highlight = [-1, -1] 

def quick_sort():
    global highlight
    def partition(low, high):
        global highlight
        pivot = data[high]
        i = low - 1
        for j in range(low, high):
            highlight = [j, high]
            increment_counters(comparison=True)
            if data[j] < pivot:
                i += 1
                data[i], data[j] = data[j], data[i]
                increment_counters(swap=True)
            yield
        data[i + 1], data[high] = data[high], data[i + 1]
        increment_counters(swap=True)
        return i + 1

    def quick_sort_helper(low, high):
        if low < high:
            pi = yield from partition(low, high)
            yield from quick_sort_helper(low, pi - 1)
            yield from quick_sort_helper(pi + 1, high)

    yield from quick_sort_helper(0, len(data) - 1)
    highlight = [-1, -1]

def merge_sort():
    global highlight, data
    
    def merge(start, mid, end):
        global highlight, data
        left = data[start:mid+1]
        right = data[mid+1:end+1]
        
        i = j = 0
        k = start
        
        while i < len(left) and j < len(right):
            highlight = [start + i, mid + 1 + j]
            increment_counters(comparison=True)
            yield
            
            if left[i] <= right[j]:
                data[k] = left[i]
                i += 1
            else:
                data[k] = right[j]
                j += 1
            k += 1
            increment_counters(swap=True)
        
        while i < len(left):
            highlight = [k, -1]
            data[k] = left[i]
            i += 1
            k += 1
            increment_counters(swap=True)
            yield
            
        while j < len(right):
            highlight = [k, -1]
            data[k] = right[j]
            j += 1
            k += 1
            increment_counters(swap=True)
            yield
    
    def merge_sort_helper(start, end):
        if start < end:
            mid = (start + end) // 2
            yield from merge_sort_helper(start, mid)
            yield from merge_sort_helper(mid + 1, end)
            yield from merge(start, mid, end)
    
    yield from merge_sort_helper(0, len(data) - 1)
    highlight = [-1, -1]

def heap_sort():
    global highlight, data
    n = len(data)
    
    def heapify(n, i, start):
        global highlight, data
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        highlight = [start + i, -1]
        yield
        
        if left < n:
            highlight = [start + largest, start + left]
            increment_counters(comparison=True)
            yield
            if data[start + left] > data[start + largest]:
                largest = left
        
        if right < n:
            highlight = [start + largest, start + right]
            increment_counters(comparison=True)
            yield
            if data[start + right] > data[start + largest]:
                largest = right
        
        if largest != i:
            highlight = [start + i, start + largest]
            data[start + i], data[start + largest] = data[start + largest], data[start + i]
            increment_counters(swap=True)
            yield
            yield from heapify(n, largest, start)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(n, i, 0)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        highlight = [0, i]
        data[0], data[i] = data[i], data[0]
        increment_counters(swap=True)
        yield
        yield from heapify(i, 0, 0)
    
    highlight = [-1, -1]

#########################################
# SEARCHING ALGORITHMS
#########################################

def linear_search(target):
    global highlight
    found = False
    trail = []
    
    for i in range(len(data)):
        increment_counters(comparison=True)
        highlight = [i, -1, "search", trail[-5:]]  # Current index, -1, status, trail
        yield
        if data[i] == target:
            highlight = [i, -1, "found", []]
            for _ in range(5):  # Blink effect on found element
                yield
            found = True
            break
        trail.append(i)
    
    if not found:
        highlight = [-1, -1, "not_found", []]
        for _ in range(3):  # Red flash for not found
            yield

def binary_search(target):
    global highlight
    data.sort()  # Sort the data first
    low = 0
    high = len(data) - 1
    found = False
    search_range = []
    
    while low <= high:
        mid = (low + high) // 2
        search_range = list(range(low, high+1))
        highlight = [mid, -1, "search", search_range]
        increment_counters(comparison=True)
        yield
        
        if data[mid] == target:
            highlight = [mid, -1, "found", []]
            for _ in range(5):  # Success animation
                yield
            found = True
            break
        elif data[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    if not found:
        highlight = [-1, -1, "not_found", []]
        for _ in range(3):  # Failure animation
            yield

def jump_search(target):
    global highlight
    data.sort()  # Sort the data first
    n = len(data)
    step = int(math.sqrt(n))
    
    prev = 0
    search_block = []
    
    # Finding the block
    while prev < n and data[min(step, n)-1] < target:
        search_block = list(range(prev, min(step, n)))
        highlight = [min(step, n)-1, -1, "search", search_block]
        increment_counters(comparison=True)
        yield
        
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            highlight = [-1, -1, "not_found", []]
            for _ in range(3):
                yield
            return
    
    # Linear search in the block
    while prev < min(step, n):
        search_block = list(range(prev, min(step, n)))
        highlight = [prev, -1, "search", search_block]
        increment_counters(comparison=True)
        yield
        
        if data[prev] == target:
            highlight = [prev, -1, "found", []]
            for _ in range(5):
                yield
            return
        
        prev += 1
    
    highlight = [-1, -1, "not_found", []]
    for _ in range(3):
        yield

def exponential_search(target):
    global highlight
    data.sort()  # Sort the data first
    n = len(data)
    
    if data[0] == target:
        highlight = [0, -1, "found", []]
        for _ in range(5):
            yield
        return
    
    # Find range for binary search
    i = 1
    while i < n and data[i] <= target:
        highlight = [i, -1, "search", list(range(0, i+1))]
        increment_counters(comparison=True)
        yield
        i = i * 2
    
    # Binary search in the range
    lo = i // 2
    hi = min(i, n-1)
    
    while lo <= hi:
        mid = (lo + hi) // 2
        search_range = list(range(lo, hi+1))
        highlight = [mid, -1, "search", search_range]
        increment_counters(comparison=True)
        yield
        
        if data[mid] == target:
            highlight = [mid, -1, "found", []]
            for _ in range(5):
                yield
            return
        elif data[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    
    highlight = [-1, -1, "not_found", []]
    for _ in range(3):
        yield

#########################################
# TREE ALGORITHMS
#########################################

def create_bst():
    global tree_root, tree_nodes
    tree_root = None
    tree_nodes = []
    
    # Create a balanced BST from sorted data
    sorted_data = sorted(list(set(random.sample(range(10, 99), 15))))
    
    def build_balanced_bst(arr, start, end):
        if start > end:
            return None
            
        mid = (start + end) // 2
        node = TreeNode(arr[mid])
        tree_nodes.append(node)
        
        node.left = build_balanced_bst(arr, start, mid - 1)
        if node.left:
            node.left.parent = node
            
        node.right = build_balanced_bst(arr, mid + 1, end)
        if node.right:
            node.right.parent = node
            
        return node
    
    tree_root = build_balanced_bst(sorted_data, 0, len(sorted_data) - 1)
    calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)

def calculate_tree_positions(node, x_min, x_max, y, level_height):
    if not node:
        return
        
    # Calculate x position (center of the range)
    node.x = (x_min + x_max) // 2
    node.y = y
    
    # Calculate positions for children
    mid_x = (x_min + x_max) // 2
    if node.left:
        calculate_tree_positions(node.left, x_min, mid_x, y + level_height, level_height)
    if node.right:
        calculate_tree_positions(node.right, mid_x, x_max, y + level_height, level_height)

def bst_insert(value):
    global tree_root, tree_nodes, tree_visualization_mode, tree_current_node, tree_path
    
    tree_visualization_mode = "insertion"
    tree_path = []
    new_node = TreeNode(value)
    
    if tree_root is None:
        tree_root = new_node
        tree_nodes.append(new_node)
        calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)
        tree_visualization_mode = "normal"
        return
    
    def insert_helper(root, value):
        global tree_current_node, tree_path
        
        if not root:
            return new_node
            
        tree_current_node = root
        tree_path.append(root)
        yield
        
        if value < root.value:
            if root.left is None:
                root.left = new_node
                new_node.parent = root
                tree_nodes.append(new_node)
                calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)
            else:
                root.left = yield from insert_helper(root.left, value)
        elif value > root.value:
            if root.right is None:
                root.right = new_node
                new_node.parent = root
                tree_nodes.append(new_node)
                calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)
            else:
                root.right = yield from insert_helper(root.right, value)
                
        return root
    
    def bst_insert_animator():
        yield from insert_helper(tree_root, value)
        tree_path = []
        tree_current_node = None
    
    return bst_insert_animator()

def bst_search(value):
    global tree_root, tree_visualization_mode, tree_current_node, tree_path
    
    tree_visualization_mode = "search"
    tree_path = []
    tree_current_node = None
    found = False
    
    def search_helper(root, value):
        global tree_current_node, tree_path, found
        
        if not root:
            return None
            
        tree_current_node = root
        tree_path.append(root)
        yield
        
        if root.value == value:
            found = True
            for _ in range(5):  # Highlight the found node
                yield
        elif value < root.value:
            yield from search_helper(root.left, value)
        else:
            yield from search_helper(root.right, value)
    
    def bst_search_animator():
        yield from search_helper(tree_root, value)
        if not found:
            # Not found animation
            for _ in range(3):
                yield
        tree_path = []
        tree_current_node = None
        tree_visualization_mode = "normal"
    
    return bst_search_animator()

def bst_delete(value):
    global tree_root, tree_nodes, tree_visualization_mode, tree_current_node, tree_path
    
    tree_visualization_mode = "deletion"
    tree_path = []
    tree_current_node = None
    node_to_delete = None
    
    def find_node(root, value):
        global tree_current_node, tree_path, node_to_delete
        
        if not root:
            return None
            
        tree_current_node = root
        tree_path.append(root)
        yield
        
        if root.value == value:
            node_to_delete = root
        elif value < root.value:
            yield from find_node(root.left, value)
        else:
            yield from find_node(root.right, value)
    
    def delete_node():
        global tree_root, tree_nodes
        
        if not node_to_delete:
            tree_visualization_mode = "normal"
            return
        
        # Case 1: No children
        if not node_to_delete.left and not node_to_delete.right:
            if node_to_delete == tree_root:
                tree_root = None
            else:
                parent = node_to_delete.parent
                if parent.left == node_to_delete:
                    parent.left = None
                else:
                    parent.right = None
            
            tree_nodes.remove(node_to_delete)
            
        # Case 2: One child
        elif not node_to_delete.left:
            if node_to_delete == tree_root:
                tree_root = node_to_delete.right
                tree_root.parent = None
            else:
                parent = node_to_delete.parent
                if parent.left == node_to_delete:
                    parent.left = node_to_delete.right
                else:
                    parent.right = node_to_delete.right
                node_to_delete.right.parent = parent
                
            tree_nodes.remove(node_to_delete)
            
        elif not node_to_delete.right:
            if node_to_delete == tree_root:
                tree_root = node_to_delete.left
                tree_root.parent = None
            else:
                parent = node_to_delete.parent
                if parent.left == node_to_delete:
                    parent.left = node_to_delete.left
                else:
                    parent.right = node_to_delete.left
                node_to_delete.left.parent = parent
                
            tree_nodes.remove(node_to_delete)
            
        # Case 3: Two children
        else:
            # Find inorder successor (smallest node in right subtree)
            successor = node_to_delete.right
            while successor.left:
                successor = successor.left
                
            # Copy successor value to the node to delete
            node_to_delete.value = successor.value
            
            # Delete the successor
            if successor.parent.left == successor:
                successor.parent.left = successor.right
            else:
                successor.parent.right = successor.right
                
            if successor.right:
                successor.right.parent = successor.parent
                
            tree_nodes.remove(successor)
        
        calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)
    
    def bst_delete_animator():
        yield from find_node(tree_root, value)
        delete_node()
        tree_path = []
        tree_current_node = None
        tree_visualization_mode = "normal"
        yield  # Final frame to show the result
    
    return bst_delete_animator()

def avl_tree():
    global tree_root, tree_nodes
    tree_root = None
    tree_nodes = []
    
    # Create nodes with random values
    values = random.sample(range(10, 99), 10)
    
    def get_height(node):
        if not node:
            return 0
        return node.height
    
    def get_balance(node):
        if not node:
            return 0
        return get_height(node.left) - get_height(node.right)
    
    def right_rotate(y):
        x = y.left
        T2 = x.right
        
        # Rotation
        x.right = y
        y.left = T2
        
        # Update parent references
        x.parent = y.parent
        y.parent = x
        if T2:
            T2.parent = y
            
        # Update parent's child reference
        if x.parent:
            if x.parent.left == y:
                x.parent.left = x
            else:
                x.parent.right = x
        
        # Update heights
        y.height = max(get_height(y.left), get_height(y.right)) + 1
        x.height = max(get_height(x.left), get_height(x.right)) + 1
        
        return x
    
    def left_rotate(x):
        y = x.right
        T2 = y.left
        
        # Rotation
        y.left = x
        x.right = T2
        
        # Update parent references
        y.parent = x.parent
        x.parent = y
        if T2:
            T2.parent = x
            
        # Update parent's child reference
        if y.parent:
            if y.parent.left == x:
                y.parent.left = y
            else:
                y.parent.right = y
        
        # Update heights
        x.height = max(get_height(x.left), get_height(x.right)) + 1
        y.height = max(get_height(y.left), get_height(y.right)) + 1
        
        return y
    
    def insert(root, node):
        # Standard BST insert
        if not root:
            return node
            
        if node.value < root.value:
            root.left = insert(root.left, node)
            root.left.parent = root
        elif node.value > root.value:
            root.right = insert(root.right, node)
            root.right.parent = root
        else:
            # Equal values not allowed
            return root
            
        # Update height
        root.height = max(get_height(root.left), get_height(root.right)) + 1
        
        # Get balance factor
        balance = get_balance(root)
        
        # Left Left Case
        if balance > 1 and node.value < root.left.value:
            return right_rotate(root)
            
        # Right Right Case
        if balance < -1 and node.value > root.right.value:
            return left_rotate(root)
            
        # Left Right Case
        if balance > 1 and node.value > root.left.value:
            root.left = left_rotate(root.left)
            return right_rotate(root)
            
        # Right Left Case
        if balance < -1 and node.value < root.right.value:
            root.right = right_rotate(root.right)
            return left_rotate(root)
            
        return root
    
    # Insert nodes to build AVL tree
    for value in values:
        node = TreeNode(value)
        tree_nodes.append(node)
        if not tree_root:
            tree_root = node
        else:
            tree_root = insert(tree_root, node)
            
    calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)

def red_black_tree():
    global tree_root, tree_nodes
    tree_root = None
    tree_nodes = []
    
    # Create nodes with random values
    values = random.sample(range(10, 99), 10)
    
    def rotate_left(x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if not x.parent:
            tree_root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
    
    def rotate_right(x):
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if not x.parent:
            tree_root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
    
    def fix_insert(k):
        while k != tree_root and k.parent.color == "RED":
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u and u.color == "RED":
                    u.color = "BLACK"
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        rotate_right(k)
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    rotate_left(k.parent.parent)
            else:
                u = k.parent.parent.right
                if u and u.color == "RED":
                    u.color = "BLACK"
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        rotate_left(k)
                    k.parent.color = "BLACK"
                    k.parent.parent.color = "RED"
                    rotate_right(k.parent.parent)
            if k == tree_root:
                break
        tree_root.color = "BLACK"
    
    def insert(value):
        node = TreeNode(value)
        tree_nodes.append(node)
        
        # Standard BST insert
        y = None
        x = tree_root
        
        while x:
            y = x
            if node.value < x.value:
                x = x.left
            else:
                x = x.right
                
        node.parent = y
        if not y:
            tree_root = node
        elif node.value < y.value:
            y.left = node
        else:
            y.right = node
            
        # Red-Black tree fix
        fix_insert(node)
    
    # Insert nodes to build Red-Black tree
    for value in values:
        insert(value)
            
    calculate_tree_positions(tree_root, 0, SCREEN_WIDTH, 80, 120)

def tree_traversals():
    global tree_visualization_mode, tree_current_node, tree_path
    tree_visualization_mode = "traversal"
    tree_path = []
    tree_current_node = None
    
    def inorder_traversal(root):
        global tree_current_node, tree_path
        if not root:
            return
            
        yield from inorder_traversal(root.left)
        
        tree_current_node = root
        tree_path.append(root)
        yield
        
        yield from inorder_traversal(root.right)
    
    def preorder_traversal(root):
        global tree_current_node, tree_path
        if not root:
            return
            
        tree_current_node = root
        tree_path.append(root)
        yield
        
        yield from preorder_traversal(root.left)
        yield from preorder_traversal(root.right)
        
    def postorder_traversal(root):
        global tree_current_node, tree_path
        if not root:
            return
            
        yield from postorder_traversal(root.left)
        yield from postorder_traversal(root.right)
        
        tree_current_node = root
        tree_path.append(root)
        yield
    
    def level_order_traversal(root):
        global tree_current_node, tree_path
        if not root:
            return
            
        queue = [root]
        while queue:
            node = queue.pop(0)
            tree_current_node = node
            tree_path.append(node)
            yield
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    # Start with inorder traversal by default
    traversal_types = ["inorder", "preorder", "postorder", "level_order"]
    current_traversal = 0
    
    def run_traversal():
        global tree_path
        tree_path = []
        
        if traversal_types[current_traversal] == "inorder":
            yield from inorder_traversal(tree_root)
        elif traversal_types[current_traversal] == "preorder":
            yield from preorder_traversal(tree_root)
        elif traversal_types[current_traversal] == "postorder":
            yield from postorder_traversal(tree_root)
        elif traversal_types[current_traversal] == "level_order":
            yield from level_order_traversal(tree_root)
            
        # Reset after traversal
        tree_path = []
        tree_current_node = None
        tree_visualization_mode = "normal"
    
    return traversal_types, current_traversal, run_traversal()

#########################################
# GRAPH ALGORITHMS
#########################################

def create_random_graph(num_nodes=10, connectivity=0.3):
    global graph_nodes, graph_edges
    graph_nodes = []
    graph_edges = []
    
    # Create nodes
    for i in range(num_nodes):
        x = random.randint(100, SCREEN_WIDTH - 100)
        y = random.randint(100, SCREEN_HEIGHT - 100)
        graph_nodes.append(GraphNode(i, x, y))
    
    # Create edges with random weights
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Avoid self-loops and duplicate edges
            if random.random() < connectivity:
                weight = random.randint(1, 10)
                graph_nodes[i].connect(j, weight)
                graph_nodes[j].connect(i, weight)  # For undirected graph
                graph_edges.append((i, j, weight))

def breadth_first_search():
    global graph_nodes, graph_selected_node, graph_start_node, graph_end_node
    
    if graph_start_node == -1:
        return
    
    # Reset colors
    for node in graph_nodes:
        node.color = custom_blue
    
    queue = [(graph_start_node, [])]  # (node_id, path)
    visited = set([graph_start_node])
    found = False
    
    def bfs_generator():
        nonlocal queue, visited, found
        
        while queue and not found:
            current, path = queue.pop(0)
            graph_nodes[current].color = custom_yellow  # Current node being explored
            yield
            
            if current == graph_end_node:
                found = True
                # Highlight the path
                for node_id in path + [current]:
                    graph_nodes[node_id].color = custom_green
                yield
                break
                
            for neighbor, _ in graph_nodes[current].connections:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [current]))
                    graph_nodes[neighbor].color = custom_purple  # Enqueued
            
            if not found:
                graph_nodes[current].color = custom_blue  # Mark as visited
            yield
        
        if not found and graph_end_node != -1:
            graph_nodes[graph_start_node].color = custom_red
            if graph_end_node != -1:
                graph_nodes[graph_end_node].color = custom_red
            yield
    
    return bfs_generator()

def depth_first_search():
    global graph_nodes, graph_selected_node, graph_start_node, graph_end_node
    
    if graph_start_node == -1:
        return
    
    # Reset colors
    for node in graph_nodes:
        node.color = custom_blue
    
    stack = [(graph_start_node, [])]  # (node_id, path)
    visited = set()
    found = False
    
    def dfs_generator():
        nonlocal stack, visited, found
        
        while stack and not found:
            current, path = stack.pop()
            
            if current in visited:
                continue
                
            visited.add(current)
            graph_nodes[current].color = custom_yellow  # Current node being explored
            yield
            
            if current == graph_end_node:
                found = True
                # Highlight the path
                for node_id in path + [current]:
                    graph_nodes[node_id].color = custom_green
                yield
                break
                
            # Add neighbors to stack in reverse order for proper DFS
            neighbors = [(n, w) for n, w in graph_nodes[current].connections]
            neighbors.reverse()
            
            for neighbor, _ in neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, path + [current]))
                    graph_nodes[neighbor].color = custom_purple  # Added to stack
            
            if not found:
                graph_nodes[current].color = custom_blue  # Mark as visited
            yield
        
        if not found and graph_end_node != -1:
            graph_nodes[graph_start_node].color = custom_red
            if graph_end_node != -1:
                graph_nodes[graph_end_node].color = custom_red
            yield
    
    return dfs_generator()

def dijkstra_algorithm():
    global graph_nodes, graph_selected_node, graph_start_node, graph_end_node
    
    if graph_start_node == -1:
        return
    
    # Reset colors
    for node in graph_nodes:
        node.color = custom_blue
    
    # Initialize distances
    distances = {i: float('infinity') for i in range(len(graph_nodes))}
    distances[graph_start_node] = 0
    previous = {i: None for i in range(len(graph_nodes))}
    
    unvisited = set(range(len(graph_nodes)))
    visited = set()
    found = False
    current_path = []
    
    def get_min_distance_node():
        min_dist = float('infinity')
        min_node = None
        for node_id in unvisited:
            if distances[node_id] < min_dist:
                min_dist = distances[node_id]
                min_node = node_id
        return min_node
    
    def reconstruct_path(end):
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        return path[::-1]  # Reverse to get start->end
    
    def dijkstra_generator():
        nonlocal unvisited, visited, found, current_path
        
        while unvisited and not found:
            current = get_min_distance_node()
            
            if current is None:  # No path exists
                break
                
            if current == graph_end_node:
                found = True
                current_path = reconstruct_path(graph_end_node)
                # Highlight the path
                for node_id in current_path:
                    graph_nodes[node_id].color = custom_green
                yield
                break
            
            unvisited.remove(current)
            visited.add(current)
            graph_nodes[current].color = custom_yellow  # Current node being explored
            yield
            
            # Check all neighbors
            for neighbor, weight in graph_nodes[current].connections:
                if neighbor in visited:
                    continue
                    
                # Calculate new distance
                new_distance = distances[current] + weight
                
                # If new distance is shorter, update
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
                    graph_nodes[neighbor].color = custom_purple  # Updated
            
            if not found:
                graph_nodes[current].color = custom_blue  # Mark as visited
            yield
        
        if not found and graph_end_node != -1:
            graph_nodes[graph_start_node].color = custom_red
            if graph_end_node != -1:
                graph_nodes[graph_end_node].color = custom_red
            yield
    
    return dijkstra_generator()

def a_star_algorithm():
    global graph_nodes, graph_selected_node, graph_start_node, graph_end_node
    
    if graph_start_node == -1 or graph_end_node == -1:
        return
    
    # Reset colors
    for node in graph_nodes:
        node.color = custom_blue
    
    # Heuristic function (Euclidean distance)
    def heuristic(node1, node2):
        x1, y1 = graph_nodes[node1].x, graph_nodes[node1].y
        x2, y2 = graph_nodes[node2].x, graph_nodes[node2].y
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # A* algorithm
    open_set = {graph_start_node}
    closed_set = set()
    
    g_score = {i: float('infinity') for i in range(len(graph_nodes))}
    g_score[graph_start_node] = 0
    
    f_score = {i: float('infinity') for i in range(len(graph_nodes))}
    f_score[graph_start_node] = heuristic(graph_start_node, graph_end_node)
    
    previous = {i: None for i in range(len(graph_nodes))}
    found = False
    
    def get_lowest_f_score_node():
        return min(open_set, key=lambda x: f_score[x])
    
    def reconstruct_path(end):
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        return path[::-1]  # Reverse to get start->end
    
    def a_star_generator():
        nonlocal open_set, closed_set, found
        
        while open_set and not found:
            current = get_lowest_f_score_node()
            
            if current == graph_end_node:
                found = True
                path = reconstruct_path(graph_end_node)
                # Highlight the path
                for node_id in path:
                    graph_nodes[node_id].color = custom_green
                yield
                break
            
            open_set.remove(current)
            closed_set.add(current)
            graph_nodes[current].color = custom_yellow  # Current node being explored
            yield
            
            # Check all neighbors
            for neighbor, weight in graph_nodes[current].connections:
                if neighbor in closed_set:
                    continue
                    
                # Calculate tentative g score
                tentative_g_score = g_score[current] + weight
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                    graph_nodes[neighbor].color = custom_purple  # Added to open set
                elif tentative_g_score >= g_score[neighbor]:
                    continue
                
                # This path is better, record it
                previous[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, graph_end_node)
            
            if not found:
                graph_nodes[current].color = custom_blue  # Mark as visited
            yield
        
        if not found:
            graph_nodes[graph_start_node].color = custom_red
            graph_nodes[graph_end_node].color = custom_red
            yield
    
    return a_star_generator()

#########################################
# UI COMPONENTS AND FUNCTIONS
#########################################

def draw_array_bars():
    global data, highlight
    
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

def draw_tree():
    global tree_root, tree_nodes, tree_visualization_mode, tree_current_node, tree_path
    
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
        elif tree_visualization_mode == "traversal" and node in tree_path:
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

def draw_graph():
    global graph_nodes, graph_edges, graph_selected_node, graph_start_node, graph_end_node
    
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

def draw_ui_elements():
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
        if algorithm == "BST":
            controls_text += " | I - Insert | D - Delete | S - Search"
        elif algorithm == "TRAVERSALS":
            controls_text += " | T - Change Traversal Type"
    elif category == "Graph":
        controls_text += " | Click - Select Node | S - Set Start | E - Set End"
    
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

#########################################
# MAIN LOOP
#########################################

def main():
    global current_category_idx, current_alg_idx, sorting, data, generator
    global highlight, speed_factor, tree_root, tree_visualization_mode
    global graph_nodes, graph_selected_node, graph_start_node, graph_end_node
    
    # Initialize data
    reset_data()
    create_bst()  # Initialize tree
    create_random_graph()  # Initialize graph
    
    # Tree traversal variables
    tree_traversal_types = None
    tree_traversal_idx = 0
    tree_traversal_gen = None
    
    # Search target
    search_target = 250  # Default target value
    
    # Main game loop
    while not pr.window_should_close():
        # Process input
        if pr.is_key_pressed(pr.KeyboardKey.KEY_RIGHT):
            current_alg_idx = (current_alg_idx + 1) % len(CATEGORIES[categories[current_category_idx]])
            sorting = False
            generator = None
            highlight = [-1, -1]
            reset_data()
            
        if pr.is_key_pressed(pr.KeyboardKey.KEY_LEFT):
            current_alg_idx = (current_alg_idx - 1) % len(CATEGORIES[categories[current_category_idx]])
            sorting = False
            generator = None
            highlight = [-1, -1]
            reset_data()
            
        if pr.is_key_pressed(pr.KeyboardKey.KEY_UP):
            current_category_idx = (current_category_idx - 1) % len(categories)
            current_alg_idx = 0
            sorting = False
            generator = None
            highlight = [-1, -1]
            reset_data()
            
            # Initialize based on category
            if categories[current_category_idx] == "Trees":
                create_bst()
            elif categories[current_category_idx] == "Graph":
                create_random_graph()
            
        if pr.is_key_pressed(pr.KeyboardKey.KEY_DOWN):
            current_category_idx = (current_category_idx + 1) % len(categories)
            current_alg_idx = 0
            sorting = False
            generator = None
            highlight = [-1, -1]
            reset_data()
            
            # Initialize based on category
            if categories[current_category_idx] == "Trees":
                create_bst()
            elif categories[current_category_idx] == "Graph":
                create_random_graph()
            
        if pr.is_key_pressed(pr.KeyboardKey.KEY_SPACE):
            if not sorting:
                category = categories[current_category_idx]
                algorithm = CATEGORIES[category][current_alg_idx]
                
                # Initialize generators based on algorithm
                if category == "Sorting":
                    if algorithm == "BUBBLE":
                        generator = bubble_sort()
                    elif algorithm == "INSERTION":
                        generator = insertion_sort()
                    elif algorithm == "SELECTION":
                        generator = selection_sort()
                    elif algorithm == "QUICK":
                        generator = quick_sort()
                    elif algorithm == "MERGE":
                        generator = merge_sort()
                    elif algorithm == "HEAP":
                        generator = heap_sort()
                elif category == "Searching":
                    if algorithm == "LINEAR":
                        generator = linear_search(search_target)
                    elif algorithm == "BINARY":
                        generator = binary_search(search_target)
                    elif algorithm == "JUMP":
                        generator = jump_search(search_target)
                    elif algorithm == "EXPONENTIAL":
                        generator = exponential_search(search_target)
                elif category == "Trees":
                    if algorithm == "TRAVERSALS":
                        tree_traversal_types, tree_traversal_idx, tree_traversal_gen = tree_traversals()
                elif category == "Graph":
                    if algorithm == "BFS":
                        generator = breadth_first_search()
                    elif algorithm == "DFS":
                        generator = depth_first_search()
                    elif algorithm == "DIJKSTRA":
                        generator = dijkstra_algorithm()
                    elif algorithm == "A_STAR":
                        generator = a_star_algorithm()
                
                sorting = True
            else:
                sorting = False
                
        if pr.is_key_pressed(pr.KeyboardKey.KEY_R):
            sorting = False
            generator = None
            highlight = [-1, -1]
            
            category = categories[current_category_idx]
            if category == "Sorting" or category == "Searching":
                reset_data()
            elif category == "Trees":
                algorithm = CATEGORIES[category][current_alg_idx]
                if algorithm == "BST":
                    create_bst()
                elif algorithm == "AVL":
                    avl_tree()
                elif algorithm == "RED_BLACK":
                    red_black_tree()
                tree_visualization_mode = "normal"
            elif category == "Graph":
                create_random_graph()
                graph_selected_node = -1
                graph_start_node = -1
                graph_end_node = -1
        
        if pr.is_key_pressed(pr.KeyboardKey.KEY_EQUAL):  # + key
            speed_factor = min(5.0, speed_factor * 1.2)
            
        if pr.is_key_pressed(pr.KeyboardKey.KEY_MINUS):
            speed_factor = max(0.1, speed_factor / 1.2)
            
        # Special keys for different categories
        category = categories[current_category_idx]
        algorithm = CATEGORIES[category][current_alg_idx]
        
        if category == "Searching" and pr.is_key_pressed(pr.KeyboardKey.KEY_T):
            search_target = random.randint(50, 500)
            sorting = False
            generator = None
            highlight = [-1, -1]
            
        elif category == "Trees" and algorithm == "BST":
            if pr.is_key_pressed(pr.KeyboardKey.KEY_I) and not sorting:
                new_value = random.randint(10, 99)
                generator = bst_insert(new_value)
                sorting = True
            elif pr.is_key_pressed(pr.KeyboardKey.KEY_D) and not sorting:
                if tree_nodes:
                    delete_value = random.choice(tree_nodes).value
                    generator = bst_delete(delete_value)
                    sorting = True
            elif pr.is_key_pressed(pr.KeyboardKey.KEY_S) and not sorting:
                if tree_nodes:
                    search_value = random.choice(tree_nodes).value
                    generator = bst_search(search_value)
                    sorting = True
                    
        elif category == "Trees" and algorithm == "TRAVERSALS":
            if pr.is_key_pressed(pr.KeyboardKey.KEY_T) and not sorting:
                tree_traversal_idx = (tree_traversal_idx + 1) % len(tree_traversal_types)
                tree_traversal_types, _, tree_traversal_gen = tree_traversals()
                generator = tree_traversal_gen
                sorting = True
                
        elif category == "Graph":
            # Check for mouse clicks to select nodes
            if pr.is_mouse_button_pressed(pr.MouseButton.MOUSE_BUTTON_LEFT):
                mouse_pos = pr.get_mouse_position()
                
                # Check if clicked on any node
                for i, node in enumerate(graph_nodes):
                    if math.sqrt((mouse_pos.x - node.x)**2 + (mouse_pos.y - node.y)**2) <= 20:
                        graph_selected_node = i
                        break
                        
            if pr.is_key_pressed(pr.KeyboardKey.KEY_S) and graph_selected_node != -1:
                graph_start_node = graph_selected_node
                
            if pr.is_key_pressed(pr.KeyboardKey.KEY_E) and graph_selected_node != -1:
                graph_end_node = graph_selected_node
        
        # Algorithm animation step
        if sorting and generator:
            # Adjust speed based on speed_factor
            steps_per_frame = max(1, int(5 * speed_factor))
            
            try:
                for _ in range(steps_per_frame):
                    next(generator)
            except StopIteration:
                sorting = False
                if category == "Trees" and algorithm == "TRAVERSALS":
                    generator = None
        
        # Drawing
        pr.begin_drawing()
        pr.clear_background(custom_dark_gray)
        
        # Draw based on current category
        if category == "Sorting" or category == "Searching":
            draw_array_bars()
        elif category == "Trees":
            draw_tree()
        elif category == "Graph":
            draw_graph()
        
        draw_ui_elements()
        
        # Draw info for search target
        if category == "Searching":
            target_text = f"Target: {search_target}"
            pr.draw_rectangle(10, 110, 150, 30, custom_light_gray)
            pr.draw_text(target_text, 20, 115, 20, custom_white)
            
            # Draw a marker for the target value
            max_height = 500
            target_ratio = search_target / max_height
            target_x = HORIZONTAL_PADDING + int(target_ratio * (SCREEN_WIDTH - 2 * HORIZONTAL_PADDING))
            pr.draw_triangle(
                pr.Vector2(target_x, 70),
                pr.Vector2(target_x - 10, 50),
                pr.Vector2(target_x + 10, 50),
                custom_red
            )
        
        # Draw tree traversal info
        if category == "Trees" and algorithm == "TRAVERSALS" and tree_traversal_types:
            info_text = f"Traversal: {tree_traversal_types[tree_traversal_idx].upper()}"
            pr.draw_rectangle(10, 70, 250, 30, custom_light_gray)
            pr.draw_text(info_text, 20, 75, 20, custom_white)
            
        # Draw graph info   
        if category == "Graph":
            info_text = f"Selected: {graph_selected_node}, Start: {graph_start_node}, End: {graph_end_node}"
            pr.draw_rectangle(10, 70, 350, 30, custom_light_gray)
            pr.draw_text(info_text, 20, 75, 20, custom_white)
        
        pr.end_drawing()
    
    pr.close_window()

if __name__ == "__main__":
    main()



