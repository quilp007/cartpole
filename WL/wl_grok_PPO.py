import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random

# 하이퍼파라미터
NUM_CELLS = 20
STATE_SIZE = NUM_CELLS + 1  # 방문 여부 (NUM_CELLS) + 현재 위치 (1)
ACTION_SIZE = NUM_CELLS     # 다음 방문할 셀 선택
GAMMA = 0.99
LEARNING_RATE = 0.0003
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 10
EPISODES = 500
MAX_STEPS = NUM_CELLS - 1

# 환경 정의
class WireLength1DEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.visited = [False] * NUM_CELLS
        self.current_pos = 0
        self.visited[0] = True
        self.path = [0]
        self.total_length = 0.0
        self.state = self._get_state()
        return self.state
    
    def _get_state(self):
        state = np.array(self.visited + [self.current_pos], dtype=np.float32)
        return state
    
    def step(self, action):
        if self.visited[action]:
            reward = -1.0
            done = False
            return self.state, reward, done
        
        dist = abs(self.current_pos - action)
        self.total_length += dist
        self.current_pos = action
        self.visited[action] = True
        self.path.append(action)
        self.state = self._get_state()
        
        reward = -dist
        done = sum(self.visited) == NUM_CELLS
        return self.state, reward, done

# PPO 네트워크
class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPONetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        policy = self.actor(state)
        mask = state[:, :-1]  # 방문 여부 (batch_size, NUM_CELLS)
        policy = policy - 1e10 * mask  # 방문한 셀의 확률을 매우 낮게
        policy = torch.softmax(policy, dim=-1)
        value = self.critic(state)
        return policy, value

# PPO 에이전트
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.model = PPONetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = []
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.model(state)
        dist = Categorical(policy)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def store(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
        returns = [g + v for g, v in zip(advantages, values)]
        return advantages, returns
    
    def learn(self):
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        with torch.no_grad():
            _, next_value = self.model(next_states[-1].unsqueeze(0))
            values = self.model(states)[1].squeeze()
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        for _ in range(PPO_EPOCHS):
            policy, values = self.model(states)
            dist = Categorical(policy)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory.clear()

# 학습 실행
env = WireLength1DEnv()
agent = PPOAgent(STATE_SIZE, ACTION_SIZE)

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    
    for step in range(MAX_STEPS):
        action, log_prob = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.store(state, action, reward, next_state, done, log_prob)
        state = next_state
        total_reward += reward
        if done:
            break
    
    agent.learn()
    if (episode + 1) % 50 == 0:
        print(f"Episode: {episode + 1}, Total Wire Length: {-total_reward:.2f}")

# 최종 결과 출력
print("\n최종 방문 순서:", env.path)
print(f"최종 Total Wire Length: {env.total_length:.2f}")