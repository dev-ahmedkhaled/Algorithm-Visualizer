import random
import math
import pyray as pr
from utils.constants import (
    custom_blue, custom_green, custom_red, custom_yellow, custom_purple
)

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

def create_random_graph(num_nodes=10, connectivity=0.3, screen_width=800, screen_height=600):
    graph_nodes = []
    graph_edges = []
    
    # Create nodes
    for i in range(num_nodes):
        x = random.randint(100, screen_width - 100)
        y = random.randint(100, screen_height - 100)
        graph_nodes.append(GraphNode(i, x, y))
    
    # Create edges with random weights
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Avoid self-loops and duplicate edges
            if random.random() < connectivity:
                weight = random.randint(1, 10)
                graph_nodes[i].connect(j, weight)
                graph_nodes[j].connect(i, weight)  # For undirected graph
                graph_edges.append((i, j, weight))
    
    return graph_nodes, graph_edges

def breadth_first_search(graph_nodes, start_node, end_node):
    if start_node == -1:
        return
    
    # Reset colors
    for node in graph_nodes:
        node.color = custom_blue
    
    queue = [(start_node, [])]  # (node_id, path)
    visited = set([start_node])
    found = False
    
    while queue and not found:
        current, path = queue.pop(0)
        graph_nodes[current].color = custom_yellow  # Current node being explored
        yield
        
        if current == end_node:
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
    
    if not found and end_node != -1:
        graph_nodes[start_node].color = custom_red
        if end_node != -1:
            graph_nodes[end_node].color = custom_red
        yield

def depth_first_search(graph_nodes, start_node, end_node):
    if start_node == -1:
        return
    
    # Reset colors
    for node in graph_nodes:
        node.color = custom_blue
    
    stack = [(start_node, [])]  # (node_id, path)
    visited = set()
    found = False
    
    while stack and not found:
        current, path = stack.pop()
        
        if current in visited:
            continue
            
        visited.add(current)
        graph_nodes[current].color = custom_yellow  # Current node being explored
        yield
        
        if current == end_node:
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
    
    if not found and end_node != -1:
        graph_nodes[start_node].color = custom_red
        if end_node != -1:
            graph_nodes[end_node].color = custom_red
        yield

def dijkstra_algorithm(graph_nodes, start_node, end_node):
    if start_node == -1:
        return
    
    # Reset colors
    for node in graph_nodes:
        node.color = custom_blue
    
    # Initialize distances
    distances = {i: float('infinity') for i in range(len(graph_nodes))}
    distances[start_node] = 0
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
    
    while unvisited and not found:
        current = get_min_distance_node()
        
        if current is None:  # No path exists
            break
            
        if current == end_node:
            found = True
            current_path = reconstruct_path(end_node)
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
    
    if not found and end_node != -1:
        graph_nodes[start_node].color = custom_red
        if end_node != -1:
            graph_nodes[end_node].color = custom_red
        yield

def a_star_algorithm(graph_nodes, start_node, end_node):
    if start_node == -1 or end_node == -1:
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
    open_set = {start_node}
    closed_set = set()
    
    g_score = {i: float('infinity') for i in range(len(graph_nodes))}
    g_score[start_node] = 0
    
    f_score = {i: float('infinity') for i in range(len(graph_nodes))}
    f_score[start_node] = heuristic(start_node, end_node)
    
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
    
    while open_set and not found:
        current = get_lowest_f_score_node()
        
        if current == end_node:
            found = True
            path = reconstruct_path(end_node)
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
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end_node)
        
        if not found:
            graph_nodes[current].color = custom_blue  # Mark as visited
        yield
    
    if not found:
        graph_nodes[start_node].color = custom_red
        graph_nodes[end_node].color = custom_red
        yield 