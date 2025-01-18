import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

# 하이퍼파라미터 설정
LEARNING_RATE = 0.01
GAMMA = 0.99
EPISODE_MAX_LENGTH = 500

# 환경 설정
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 정책 신경망 (Policy Network)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1) # Softmax를 사용하여 확률 분포 출력
        )

    def forward(self, x):
        return self.layers(x)

# REINFORCE 에이전트
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.gamma = GAMMA

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = self.policy_net(state)
        action = torch.multinomial(probs, num_samples=1).item() # 확률에 따라 행동 선택
        return action, probs

    def train_step(self, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards): # 보상을 뒤에서부터 계산
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8) # 정규화

        log_probs = torch.stack(log_probs).to(device)
        policy_loss = -(log_probs * discounted_rewards).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

# 학습 루프
agent = REINFORCEAgent(state_dim, action_dim)
#num_episodes = 1000
num_episodes = 100
scores = []

for episode in range(num_episodes):
    state = env.reset()[0]
    episode_rewards = []
    episode_log_probs = []
    score = 0
    for t in range(EPISODE_MAX_LENGTH):
        action, probs = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        episode_rewards.append(reward)
        episode_log_probs.append(torch.log(probs[0][action])) # 선택한 행동의 로그 확률 저장

        state = next_state
        score += reward

        if done:
            break

    agent.train_step(episode_rewards, episode_log_probs) # 에피소드 종료 후 학습
    scores.append(score)
    print(f"Episode: {episode + 1}, Score: {score}")

env.close()

torch.save(agent.policy_net.state_dict(), 'cartpole_pg_model.pth') # 모델 저장
print("Model saved!")

# 결과 그래프 출력 (matplotlib 필요)
import matplotlib.pyplot as plt
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()