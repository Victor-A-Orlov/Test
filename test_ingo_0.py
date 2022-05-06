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
# X = X + np.array([1, 2])

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])

for i, txt in enumerate(y):
    ax.annotate((X[i, 0], X[i, 1]), (X[i, 0], X[i, 1]))
    
    
class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.data = val

def insert(root, node):
    if root is None:
        root = node
    else:
        if root.data > node.data:
            print('root.data > node.data')
            if root.left is None:
                print('root.left is None')
                root.left = node
            else:
                print('root.left is NOT None')
                insert(root.left, node)
        else:
            print('root.data <= node.data')
            if root.right is None:
                print('root.right is None')
                root.right = node
            else:
                print('root.right is NOT None')
                insert(root.right, node)

t = Node(3)