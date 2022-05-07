from sklearn.datasets import load_digits, make_blobs, make_classification
import matplotlib.pyplot as plt
import numpy as np

n_samples = 10
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
X = X.astype('int')

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])

for i, txt in enumerate(y):
    ax.annotate((X[i, 0], X[i, 1]), (X[i, 0], X[i, 1]))
    
q = np.array([10,34])
ax.scatter(q[0], q[1])

DIM = 2

class Node:
    def __init__(self, val, alignment_axis=0):
        self.left = None
        self.right = None
        self.axis = 0
        self.data = val
        
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
            if node.data[root.axis] < root.data[root.axis]:
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
for x in X:
    t.insert(node=Node(x))
    
def find(root, q):
    best_distance = 10000
    best_node = root
    node = root
    while True:
        d = np.linalg.norm(node.data - q)
        if d < best_distance:
            best_distance = d
            best_node = node
        if q[node.axis] < node.data[node.axis]:
            node = node.left
            if node is None:
                break
        else:
            node = node.right
            if node is None:
                break
    
    return best_node.data


b = find(t.root, q)
print(b)
