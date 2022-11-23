import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df1 = pd.read_csv('output/mnist_fedavg_iid.csv')
df2 = pd.read_csv('output/mnist_fedavg_noniid.csv')
df3 = pd.read_csv('output/mnist_kcenter_noniid.csv')
df4 = pd.read_csv('output/mnist_kmeans_noniid.csv')

# Plot data using accuracy vs. round
fig, ax = plt.subplots()
sns.lineplot(x='round', y='accuracy', data=df1, label='FedAvg (IID)', ax=ax)
sns.lineplot(x='round', y='accuracy', data=df2, label='FedAvg (Non-IID)', ax=ax)
sns.lineplot(x='round', y='accuracy', data=df3, label='KCenter (Non-IID)', ax=ax)
sns.lineplot(x='round', y='accuracy', data=df4, label='KMeans (Non-IID)', ax=ax)
#ax.set_title('MNIST')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Testing Accuracy (%)')
# turn on grid lines
ax.grid(True)
ax.set_ylim(94, 100)
ax.set_xlim(0, 250)
#plt.show()

# save the figure as a PNG
fig.savefig('output/fig_1.png')