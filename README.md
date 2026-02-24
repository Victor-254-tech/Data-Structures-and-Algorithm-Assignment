BIT2203 - DATA STRUCTURES AND ALGORITHMS GROUP ASSIGNMENT
GROUP MEMBERS
1. ISAAC OTWORI - BSCCS/2025/53541
2. BONIFACE MUTUNGA - BBIT/2017/83097.
3. JACQUES KAMUNTU -  BBIT/2025/42777
4. ⁠EMMANUEL CHERUIYOT CHOROREI - BSCCS/2025/55718 
5. COLLINS MURANGI -BSCCS/2025/54525.
6. AMINA - BBIT/2025/53280
7. ⁠ETON OTIENO - BSCCS/2025/54479 
8. ⁠GIFT MWADIGA -BBIT/2025/31150
9. VICTOR KIVINDA BIT/2022/51502

**Question 1**
Using a programming language of your choice, in groups, write code to represent each of the data structure classification and types. 
Data structures are generally classified into two main groups, primitive and non-primitive. Non-primitive structures then branch into linear and non-linear. 

1.	Primitive Data Structures 
These are the building blocks. They store a single value at a specific memory address. 
Types - Integer, Float, Character, Boolean. 
# Primitive types are built-in age = 25          # Integer price = 19.99     # Float is_valid = True   # Boolean initial = 'A'     # Character (String in Python) 

2.	Non-Primitive: Linear Data Structures 
Data elements are arranged in a sequence. Each element except the first and last has a 
"previous" and a "next." 
•	Static (Array): Fixed size. Fast access but hard to resize. 
•	Dynamic (Linked List, Stack, Queue): Size changes at runtime. 
# 1. Array  arr = [10, 20, 30, 40] 
 
# 2. Linked List  class Node:     def __init__(self, data): 
        self.data = data         self.next = None 
 
# 3. Stack (LIFO - Last In First Out) stack = [] 
stack.append('A') # Push stack.pop()       # Pop 
 
# 4. Queue (FIFO - First In First Out) 
 
from collections import deque queue = deque(["User1", "User2"]) queue.append("User3") # Enqueue queue.popleft()       # Dequeue 


3. Non-Primitive: Non-Linear Data Structures Data is attached hierarchically or interconnected. 
•	Trees: Hierarchical (Parent-Child). 
•	Graphs: Network-based (Nodes and Edges). 
# 1. Binary Tree Node class TreeNode:     def __init__(self, value): 
        self.left = None         self.right = None         self.val = value 
 
# 2. Graph (using an Adjacency List) graph = { 
    'A': ['B', 'C'], 
    'B': ['D'],     'C': ['D'], 
    'D': [] } 
 
**Question 2**
Research where the data structures types are applied and give reasons why. 
1.	Arrays  
a.	Application: Image Processing 
b.	Reason: Pixels are stored in contiguous memory for fast sequential access. 

2.	Linked Lists 
a.	Application: Music Player Playlists 
b.	Reason: Easy to insert or delete songs without shifting the entire list in memory. 

3.	Stacks 
a.	Application: "Undo" in Word/Photoshop 
b.	Reason: LIFO allows you to revert the most recent action first. 

4.	Queues 
a.	Application: Printer Spooling 
b.	Reason: FIFO ensures the first document sent is the first document printed. 

5.	Hash Tables 
a.	Application: Database Indexing 
b.	Reason: Provides fast lookups in databases. 

6.	Graphs 
a.	Application: Google Maps 
b.	Reason: Locations are nodes and roads are edges; allows for pathfinding algorithms. 


**Question 3**
Give examples of applications that are using the data structure type and algorithm. Give reasons why. 

1: Web Browsers (Stack & Queue) 
When navigating the web, the browser uses a stack to manage the history. The "Back" button pops the top URL off the stack. Meanwhile, it uses a queue to manage network requests (fetching images, scripts) so they load in the order they were requested. 

2: Social Media (Graphs) 
Facebook or LinkedIn doesn't use a list to store users. They use a graph. The algorithm they use is called Breadth-First Search (BFS). It is used to find "Suggested Friends" or "2nddegree connections." It traverses the graph from your node to see who your friends are connected to. 


**Question 4**
Research how data structures and algorithms work within systems. 
Data Structures are the organization and Algorithms as the strategy.  
1.	Memory Management: Systems use the Heap (for dynamic structures like Linked Lists) and the Stack (for function calls and primitive local variables). 
2.	CPU Efficiency: An efficient algorithm (like QuickSort vs. Bubble Sort) reduces the number of clock cycles the CPU spends on a task. 
3.	Data Persistence: When saving data to a disk, the Operating System uses B-Trees (nodes can have more than two children and store multiple keys) to organize files so the hard drive head does not have to do an intensive search for the file. 
4.	Network Throughput: Routers use Queues (buffers) to handle spikes in traffic. If the queue fills up, packets are dropped—this is why an online game lags. 
 
