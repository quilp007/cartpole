#import gym
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

# 환경 설정
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 하이퍼파라미터
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.001
batch_size = 64
replay_buffer_size = 10000

# DQN 모델
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

# 리플레이 버퍼
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        #self.model = DQN(state_dim, action_dim)
        self.model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def select_action(self, state):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.replay_buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 학습 루프
agent = DQNAgent(state_dim, action_dim)
#num_episodes = 500
num_episodes = 100

for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        #next_state, reward, done, truncated, info = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train()

        total_reward += reward
        state = next_state

        if epsilon > 0.01:
            epsilon -= epsilon_decay

    #print(f"Episode: {episode+1}, Total Reward: {total_reward}")
    print(f"Episode: {episode + 1}, Score: {total_reward}, Epsilon: {epsilon}")

torch.save(agent.model.state_dict(), 'cartpole_model.pth') # 모델 저장
print("Model saved!")

env.close()