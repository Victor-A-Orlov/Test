from sklearn.datasets import load_digits, make_blobs, make_classification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


n_samples = 100
random_state = 170
X, y = make_classification(
    n_features=2, 
    n_redundant=0, 
    n_informative=2, 
    random_state=1, 
    n_clusters_per_class=1,
    n_samples=n_samples,
    scale=10
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

for x in X_test:
    plt.scatter(x[0], x[1], c='b')

DIM = 2

class Node:
    def __init__(self, X, y, alignment_axis=0):
        self.left = None
        self.right = None
        self.axis = 0
        self.X = X
        self.y = y
        
    def __call__(self):
        return self.data

class Tree:
    def __init__(self, val=None):
        self.root = None
        
    def insert(self, root=None, node=None):
        if self.root is None:
            self.root = node
        else:
            if root is None:
                root = self.root
            node.axis = (root.axis + 1) % DIM
            if node.X[root.axis] < root.X[root.axis]:
                if root.left is None:
                    root.left = node
                    print('ok')
                else:
                    self.insert(root.left, node)
            else:
                if root.right is None:
                    root.right = node
                    print('ok')
                else:
                    self.insert(root.right, node)
                

t = Tree()
for x_sample, y_sample in zip(X_train, y_train):
    t.insert(node=Node(x_sample, y_sample))
    
def query_targets(root, Q):
    results = []
    for q in Q:
        _, y = find_one_neighbor(root, q)
        results.append(y)
    return np.array(results)
        
        
def find_one_neighbor(root, q):
    best_distance = 10000
    best_node = root
    node = root
    while True:
        d = np.linalg.norm(node.X - q)
        if d < best_distance:
            best_distance = d
            best_node = node
        if q[node.axis] < node.X[node.axis]:
            node = node.left
            if node is None:
                break
        else:
            node = node.right
            if node is None:
                break
    
    return best_node.X, best_node.y

y_pred = query_targets(t.root, X_test)
