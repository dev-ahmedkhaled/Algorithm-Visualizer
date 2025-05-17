def bubble_sort(data, highlight, increment_counters):
    n = len(data)
    for i in range(n):
        for j in range(0, n-i-1):
            highlight[0] = j
            highlight[1] = j+1
            increment_counters(comparison=True)
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                increment_counters(swap=True)
            yield
    highlight[0] = -1
    highlight[1] = -1

def insertion_sort(data, highlight, increment_counters):
    for i in range(1, len(data)):
        key = data[i]
        j = i-1
        highlight[0] = j
        highlight[1] = i
        yield
        while j >= 0 and key < data[j]:
            highlight[0] = j
            highlight[1] = j+1
            increment_counters(comparison=True)
            data[j + 1] = data[j]
            increment_counters(swap=True)
            j -= 1
            yield
        data[j + 1] = key
        increment_counters(swap=True)
    highlight[0] = -1
    highlight[1] = -1

def selection_sort(data, highlight, increment_counters):
    n = len(data)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            highlight[0] = j
            highlight[1] = min_idx
            increment_counters(comparison=True)
            if data[j] < data[min_idx]:
                min_idx = j
            yield
        data[i], data[min_idx] = data[min_idx], data[i]
        increment_counters(swap=True)
        highlight[0] = i
        highlight[1] = min_idx
        yield
    highlight[0] = -1
    highlight[1] = -1

def quick_sort(data, highlight, increment_counters):
    def partition(low, high):
        pivot = data[high]
        i = low - 1
        for j in range(low, high):
            highlight[0] = j
            highlight[1] = high
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
    highlight[0] = -1
    highlight[1] = -1

def merge_sort(data, highlight, increment_counters):
    def merge(start, mid, end):
        left = data[start:mid+1]
        right = data[mid+1:end+1]
        
        i = j = 0
        k = start
        
        while i < len(left) and j < len(right):
            highlight[0] = start + i
            highlight[1] = mid + 1 + j
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
            highlight[0] = k
            highlight[1] = -1
            data[k] = left[i]
            i += 1
            k += 1
            increment_counters(swap=True)
            yield
            
        while j < len(right):
            highlight[0] = k
            highlight[1] = -1
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
    highlight[0] = -1
    highlight[1] = -1

def heap_sort(data, highlight, increment_counters):
    n = len(data)
    
    def heapify(n, i, start):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        highlight[0] = start + i
        highlight[1] = -1
        yield
        
        if left < n:
            highlight[0] = start + largest
            highlight[1] = start + left
            increment_counters(comparison=True)
            yield
            if data[start + left] > data[start + largest]:
                largest = left
        
        if right < n:
            highlight[0] = start + largest
            highlight[1] = start + right
            increment_counters(comparison=True)
            yield
            if data[start + right] > data[start + largest]:
                largest = right
        
        if largest != i:
            highlight[0] = start + i
            highlight[1] = start + largest
            data[start + i], data[start + largest] = data[start + largest], data[start + i]
            increment_counters(swap=True)
            yield
            yield from heapify(n, largest, start)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(n, i, 0)
    
    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        highlight[0] = 0
        highlight[1] = i
        data[0], data[i] = data[i], data[0]
        increment_counters(swap=True)
        yield
        yield from heapify(i, 0, 0)
    
    highlight[0] = -1
    highlight[1] = -1 