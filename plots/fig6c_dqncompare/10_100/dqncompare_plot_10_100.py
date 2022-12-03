import pandas as pd
import matplotlib.pyplot as plt
import os


cwd = os.getcwd()
files = os.listdir(cwd)
df1 = pd.read_csv(files[1])
df2 = pd.read_csv(files[2])
df3 = pd.read_csv(files[3])
df4 = pd.read_csv(files[4])
df5 = pd.read_csv(files[5])
df6 = pd.read_csv(files[6])


round1 = df1['round']
accuracy1 = df1['accuracy']
round2 = df2['round']
accuracy2 = df2['accuracy']
round3 = df3['round']
accuracy3 = df3['accuracy']
round4 = df4['round']
accuracy4 = df4['accuracy']
round5 = df5['round']
accuracy5 = df5['accuracy']
round6 = df6['round']
accuracy6 = df6['accuracy']

plt.plot(round1, accuracy1, c = 'black', label="dqn_noniid")
plt.plot(round2, accuracy2, c = 'b', label="dqn_noniid_difference")
plt.plot(round3, accuracy3, c = 'y', label="fedavg_iid")
plt.plot(round4, accuracy4, c = 'r', label="fedavg_noniid")
plt.plot(round5, accuracy5, c = 'hotpink', label="kcenter_noniid")
plt.plot(round6, accuracy6, c = 'c', label="kmeans_noniid")

plt.xlabel('Round(s)')
plt.ylabel('Accuracy')
plt.grid()
plt.title('Compare DQN (10_100) with traditional method with accuracy and communication rounds')
plt.legend()
plt.ylim([94,100])

plt.show()
plt.savefig("dqn_compare_10_100.png")