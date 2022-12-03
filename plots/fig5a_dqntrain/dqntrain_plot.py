import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("dqn_10-100_difference_rewards.csv")

episode1 = df1['Episode']
reward1 = df1['Reward']

plt.plot(episode1, reward1)
plt.xlabel('Episode(s)')
plt.ylabel('Total Return')
plt.grid()
plt.title('DQN 10-100 training with different rewards')
plt.ylim([-20, 80])

plt.show()
plt.savefig("dqn_10-100_difference_rewards.png")


df2 = pd.read_csv("dqn_10-100_target_rewards.csv")

episode2 = df2['Episode']
reward2 = df2['Reward']

plt.plot(episode2, reward2)
plt.xlabel('Episode(s)')
plt.ylabel('Total Return')
plt.grid()
plt.title('DQN 10-100 training with target rewards')
plt.ylim([-20, 80])

plt.show()
plt.savefig("dqn_10-100_target_rewards.png")


df3 = pd.read_csv("dqn_4-20_difference_rewards.csv")

episode3 = df3['Episode']
reward3 = df3['Reward']

plt.plot(episode3, reward3)
plt.xlabel('Episode(s)')
plt.ylabel('Total Return')
plt.grid()
plt.title('DQN 4-20 training with different rewards')
plt.ylim([-20, 80])

plt.show()
plt.savefig("dqn_4-20_difference_rewards.png")


df4 = pd.read_csv("dqn_4-20_target_rewards.csv")

episode4 = df4['Episode']
reward4 = df4['Reward']

plt.plot(episode4, reward4)
plt.xlabel('Episode(s)')
plt.ylabel('Total Return')
plt.grid()
plt.title('DQN 4-20 training with target rewards')
plt.ylim([-20, 80])

plt.show()
plt.savefig("dqn_4-20_target_rewards.png")
