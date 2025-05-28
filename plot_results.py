import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("results.csv", header=None, names=["x", "y", "cluster"])

plt.figure(figsize=(8, 6))

num_clusters = data['cluster'].nunique()
colors = plt.cm.get_cmap('tab10', num_clusters)

for i, label in enumerate(sorted(data['cluster'].unique())):
    cluster = data[data["cluster"] == label]
    plt.scatter(cluster["x"], cluster["y"], label=f"Cluster {label}", s=10, color=colors(i))

plt.title("CUDA KMeans Clustering Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()