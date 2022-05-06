from sklearn.datasets import load_digits, make_blobs, make_classification
import matplotlib.pyplot as plt

n_samples = 10
random_state = 170
X, y = make_classification(n_samples=n_samples, random_state=random_state, scale=10)
X = X.astype('int')
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1])

for i, txt in enumerate(y):
    ax.annotate((X[i, 0], X[i, 1]), (X[i, 0], X[i, 1]))
    
