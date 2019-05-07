import json
import matplotlib.pyplot as plt

with open('openaigym.episode_batch.0.3399.stats.json', 'r') as fp:
    data = json.load(fp)


timesteps = data["timestamps"]
episode_length = data["episode_lengths"]
episode_types  = data["episode_types"]
episode_reward = data["episode_rewards"]


plt.figure()          
plt.plot(timesteps,episode_length,'r',label='episode_lengths')
plt.plot(timesteps,episode_reward,'b',label='episode_reward')
plt.legend()
plt.savefig('Training_performance.png')