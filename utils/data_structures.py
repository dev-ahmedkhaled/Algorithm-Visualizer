"""
Data structure classes used throughout the algorithm visualizer.
These classes define the fundamental structures used for tree and graph algorithms.
"""

class TreeNode:
    """
    A node in a binary tree structure.
    
    Attributes:
        value: The value stored in the node
        left: Reference to the left child node
        right: Reference to the right child node
        height: Height of the node (used for AVL trees)
        color: Color of the node (used for Red-Black trees)
        x: X-coordinate for visualization
        y: Y-coordinate for visualization
        parent: Reference to the parent node
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
        self.color = "RED"
        self.x = 0  # For visualization
        self.y = 0  # For visualization
        self.parent = None  # For some algorithms

class GraphNode:
    """
    A node in a graph structure.
    
    Attributes:
        id: Unique identifier for the node
        x: X-coordinate for visualization
        y: Y-coordinate for visualization
        connections: List of (node_id, weight) tuples representing edges
        color: Color of the node for visualization
    """
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.connections = []  # List of (node_id, weight) tuples
        self.color = None  # Will be set by the visualization module
    
    def connect(self, other_id, weight=1):
        """
        Add a connection to another node.
        
        Args:
            other_id: ID of the node to connect to
            weight: Weight of the edge (default: 1)
        """
        if not any(conn[0] == other_id for conn in self.connections):
            self.connections.append((other_id, weight)) 