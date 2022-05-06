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

def insert(root, node):
    if root is None:
        root = node
    else:
        node.axis = (root.axis + 1) % DIM
        print('node axis now ', node.axis)
        if node.data[root.axis] < root.data[root.axis]:
            print('node.data[root.axis] < root.data[root.axis] ')
            if root.left is None:
                print('root.left is None ')
                root.left = node
            else:
                print('root.left is NOT None ')
                insert(root.left, node)
        else:
            print('node.data[root.axis] >= root.data[root.axis] ')
            if root.right is None:
                print('root.right is None ')
                root.right = node
            else:
                print('root.right is NOT None ')
                insert(root.right, node)
                

best_distance = 1e5
best_node = None
def find(node, q):
    if node.
    print('data: ', node.data)
    print('axis: ', node.axis)
    if q[node.axis] < node.data[node.axis]:
        print('q[node.axis] < node.data[node.axis] ')
        if node.left is None:
            pass
        else:
            print('root.left is NOT None ')
            find(node.left, q)
    else:
        print('q[node.axis] >= node.data[node.axis]')
        if node.right is None:
            pass
        else:
            print('node.right is NOT None ')
            find(node.right, q)

t = Node(X[0])
for x in X[1:]:
    insert(t, Node(x))
    

