"""
Tree algorithms and operations for the algorithm visualizer.
This module implements various binary tree operations with visualization support.
"""

import random
import pyray as pr
from utils.constants import (
    custom_blue, custom_green, custom_red, custom_yellow, custom_purple
)

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1  
        self.color = "RED" 
        self.x = 0  # For visualization
        self.y = 0  # For visualization
        self.parent = None  # For some algorithms

def create_random_tree(num_nodes=10, screen_width=800, screen_height=600):
    """
    Create a random balanced binary search tree.
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(n) for storing nodes and recursion stack
    
    Args:
        num_nodes: Number of nodes to create
        screen_width: Width of the visualization area
        screen_height: Height of the visualization area
    
    Returns:
        tuple: (root_node, list_of_nodes)
    """
    tree_root = None
    tree_nodes = []
    
    # Create a balanced BST from sorted data
    sorted_data = sorted(list(set(random.sample(range(10, 99), num_nodes))))
    
    def build_balanced_bst(arr, start, end):
        """Helper function to build a balanced BST recursively."""
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
    calculate_tree_positions(tree_root, 0, screen_width, 80, 120)
    
    return tree_root, tree_nodes

def calculate_tree_positions(node, x_min, x_max, y, level_height):
    """
    Calculate x,y coordinates for tree visualization.
    
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(h) where h is height of tree for recursion stack
    
    Args:
        node: Current node to position
        x_min: Minimum x coordinate for this level
        x_max: Maximum x coordinate for this level
        y: Current y coordinate (level)
        level_height: Height between levels
    """
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

def insert_node(root, value, tree_nodes):
    """
    Insert a new node into the binary search tree.
    
    Time Complexity:
        Best: O(1) when tree is empty
        Average: O(log n) for balanced tree
        Worst: O(n) for skewed tree
    Space Complexity: O(1) - iterative implementation
        Note: Recursive implementation would use O(h) where h is height
    
    Args:
        root: Root node of the tree
        value: Value to insert
        tree_nodes: List of all nodes in the tree
    
    Returns:
        tuple: (updated_root, updated_nodes, path_taken)
    """
    tree_path = []
    new_node = TreeNode(value)
    if root is None:
        tree_nodes.append(new_node)
        return new_node, tree_nodes, tree_path
    node = root
    while node:
        tree_path.append(node)
        if value < node.value:
            if node.left is None:
                node.left = new_node
                new_node.parent = node
                tree_nodes.append(new_node)
                break
            node = node.left
        elif value > node.value:
            if node.right is None:
                node.right = new_node
                new_node.parent = node
                tree_nodes.append(new_node)
                break
            node = node.right
        else:
            # Value already exists
            break
    return root, tree_nodes, tree_path

def delete_node(root, value, tree_nodes):
    """
    Delete a node from the binary search tree.
    
    Time Complexity:
        Best: O(1) when tree is empty
        Average: O(log n) for balanced tree
        Worst: O(n) for skewed tree
    Space Complexity: O(1) - iterative implementation
        Note: Recursive implementation would use O(h) where h is height
    
    Args:
        root: Root node of the tree
        value: Value to delete
        tree_nodes: List of all nodes in the tree
    
    Returns:
        tuple: (updated_root, updated_nodes, path_taken)
    """
    tree_path = []
    node = root
    parent = None
    while node and node.value != value:
        tree_path.append(node)
        parent = node
        if value < node.value:
            node = node.left
        else:
            node = node.right
    if node is None:
        return root, tree_nodes, tree_path  # Not found
    tree_path.append(node)
    
    # Case 1: No children
    if node.left is None and node.right is None:
        if parent is None:
            root = None
        elif parent.left == node:
            parent.left = None
        else:
            parent.right = None
        tree_nodes.remove(node)
    # Case 2: One child
    elif node.left is None or node.right is None:
        child = node.left if node.left else node.right
        if parent is None:
            root = child
        elif parent.left == node:
            parent.left = child
        else:
            parent.right = child
        child.parent = parent
        tree_nodes.remove(node)
    # Case 3: Two children
    else:
        succ_parent = node
        succ = node.right
        while succ.left:
            succ_parent = succ
            succ = succ.left
        node.value = succ.value
        if succ_parent.left == succ:
            succ_parent.left = succ.right
        else:
            succ_parent.right = succ.right
        if succ.right:
            succ.right.parent = succ_parent
        tree_nodes.remove(succ)
    return root, tree_nodes, tree_path

def search_node(root, value):
    """
    Search for a value in the binary search tree.
    
    Time Complexity:
        Best: O(1) when root is target
        Average: O(log n) for balanced tree
        Worst: O(n) for skewed tree
    Space Complexity: O(1) - iterative implementation
        Note: Recursive implementation would use O(h) where h is height
    
    Args:
        root: Root node of the tree
        value: Value to search for
    
    Returns:
        tuple: (found_node, path_taken)
    """
    tree_path = []
    node = root
    while node:
        tree_path.append(node)
        if value == node.value:
            return node, tree_path
        elif value < node.value:
            node = node.left
        else:
            node = node.right
    return None, tree_path 