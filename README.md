# Algorithm Visualizer

A cross‑platform, full‑screen algorithm visualizer using [Raylib](https://www.raylib.com/), demonstrating common sorting, searching, tree, and graph algorithms with step‑by‑step animations and live performance counters.

## Features

- **Sorting**: Bubble, Insertion, Selection, Quick, Merge, Heap
- **Searching**: Linear, Binary, Jump, Exponential
- **Trees**: BST (with insert/delete/search), AVL, Red‑Black, and traversals (in‑order, pre‑order, post‑order, level‑order)
- **Graphs**: Random graph generation and visualization of BFS, DFS, Dijkstra, and A* pathfinding
- Live **Steps**, **Comparisons**, and **Swaps** counters for sorting
- Dynamic **Target** visualization for searching
- Interactive **Node** selection, **Start/End** marking for graph algorithms
- Adjustable **Animation Speed** and **Full‑screen** support

## Installation

1. **Clone the repository**
- git clone https://github.com/dev-ahmedkhaled/Algorithm-Visualizer.git
- cd algorithm‑visualizer
   
3. Create and activate a virtual environment

**Using conda:**

- conda create -n name_of_env
- conda activate name_of_env


**Using pip:**


- python3 -m venv venv
- source venv/bin/activate


## 4. Install dependencies

**Using pip:**

- cd <the_file_location>
- pip install -r requirements.txt



**Using conda:**

- conda install --yes --file requirements.txt


## Usage
- **← / →** : Change algorithm  
- **↑ / ↓** : Change category (Sorting, Searching, Trees, Graph)  
- **SPACE**   : Start / Pause animation  
- **R**       : Reset data or structure  
- **+ / –**   : Increase / Decrease animation speed  
- **I / D / S** : (BST only) Insert / Delete / Search a random value  
- **Click**   : (Graph only) Select a node  
- **S / E**   : (Graph only) Set Start / End node for pathfinding 
