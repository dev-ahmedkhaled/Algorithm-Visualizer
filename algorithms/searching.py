import math

def linear_search(data, target, highlight, increment_counters):
    found = False
    trail = []
    
    for i in range(len(data)):
        increment_counters(comparison=True)
        highlight[0] = i
        highlight[1] = -1
        highlight[2] = "search"
        highlight[3] = trail[-5:]
        yield
        if data[i] == target:
            highlight[0] = i
            highlight[1] = -1
            highlight[2] = "found"
            highlight[3] = []
            for _ in range(5):  # Blink effect on found element
                yield
            found = True
            break
        trail.append(i)
    
    if not found:
        highlight[0] = -1
        highlight[1] = -1
        highlight[2] = "not_found"
        highlight[3] = []
        for _ in range(3):  # Red flash for not found
            yield

def binary_search(data, target, highlight, increment_counters):
    data.sort()  # Sort the data first
    low = 0
    high = len(data) - 1
    found = False
    search_range = []
    
    while low <= high:
        mid = (low + high) // 2
        search_range = list(range(low, high+1))
        highlight[0] = mid
        highlight[1] = -1
        highlight[2] = "search"
        highlight[3] = search_range
        increment_counters(comparison=True)
        yield
        
        if data[mid] == target:
            highlight[0] = mid
            highlight[1] = -1
            highlight[2] = "found"
            highlight[3] = []
            for _ in range(5):  # Success animation
                yield
            found = True
            break
        elif data[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    
    if not found:
        highlight[0] = -1
        highlight[1] = -1
        highlight[2] = "not_found"
        highlight[3] = []
        for _ in range(3):  # Failure animation
            yield

