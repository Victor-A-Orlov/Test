from sklearn.datasets import load_digits, make_blobs, make_classification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

n_samples = 1000
random_state = 170
X, y = make_classification(
    n_features=2, 
    n_redundant=0, 
    n_informative=2, 
    random_state=1, 
    n_clusters_per_class=1,
    n_samples=n_samples,
    scale=10,
    n_classes=2
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
                else:
                    self.insert(root.left, node)
            else:
                if root.right is None:
                    root.right = node
                else:
                    self.insert(root.right, node)
                

t = Tree()
for x_sample, y_sample in zip(X_train, y_train):
    t.insert(node=Node(x_sample, y_sample))
    
def query_targets(start_node, Q):
    results = []
    for q in Q:
        node = find_one_neighbor(start_node, q)
        results.append(node.y)
    return np.array(results)
        
        
def find_one_neighbor(start_node, q):
    best_distance = 10000
    best_node = start_node
    node = start_node
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
    
    return best_node

class Classifier:
    def __init__(self) -> None:
        self.tree = Tree()

    def fit(self, X, y):
        for x_sample, y_sample in zip(X, y):
            self.tree.insert(node=Node(x_sample, y_sample))

    def predict(self, query):
        return query_targets(self.tree.root, query)

y_pred = query_targets(t.root, X_test)
print(accuracy_score(y_test, y_pred))

clf = Classifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))