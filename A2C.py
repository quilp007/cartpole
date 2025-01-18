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
LEARNING_RATE = 0.001
GAMMA = 0.99
EPISODE_MAX_LENGTH = 500

# 환경 설정
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Actor-Critic 네트워크
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# A2C 에이전트
class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.gamma = GAMMA

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, _ = self.model(state)
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        probs, values = self.model(states)
        _, next_values = self.model(next_states)

        advantages = rewards + (1 - dones) * self.gamma * next_values - values
        actor_loss = -(torch.log(probs.gather(1, actions)) * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 학습 루프
agent = A2CAgent(state_dim, action_dim)
num_episodes = 2000
scores = []

for episode in range(num_episodes):
    state = env.reset()[0]
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_next_states = []
    episode_dones = []
    score = 0

    for t in range(EPISODE_MAX_LENGTH):
        action, probs = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_next_states.append(next_state)
        episode_dones.append(done)

        state = next_state
        score += reward

        if done:
            break

    agent.train_step(episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones)
    scores.append(score)
    print(f"Episode: {episode + 1}, Score: {score}")

env.close()

torch.save(agent.model.state_dict(), 'cartpole_a2c_model.pth') # 모델 저장
print("Model saved!")

# 결과 그래프 출력 (matplotlib 필요)
import matplotlib.pyplot as plt
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()