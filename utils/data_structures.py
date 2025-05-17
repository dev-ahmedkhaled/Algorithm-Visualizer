class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
        self.color = "RED"
        self.x = 0
        self.y = 0
        self.parent = None

class GraphNode:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.connections = []
        self.color = None  # Will be set by the visualization module
    
    def connect(self, other_id, weight=1):
        if not any(conn[0] == other_id for conn in self.connections):
            self.connections.append((other_id, weight)) 