import gym
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
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
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

# Actor-Critic 에이전트
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer_actor = optim.Adam(self.model.actor.parameters(), lr=LEARNING_RATE_ACTOR)
        self.optimizer_critic = optim.Adam(self.model.critic.parameters(), lr=LEARNING_RATE_CRITIC)
        self.gamma = GAMMA

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, _ = self.model(state)
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        reward = torch.tensor([reward], dtype=torch.float32).to(device)
        done = torch.tensor([done], dtype=torch.float32).to(device)

        probs, value = self.model(state)
        _, next_value = self.model(next_state)

        td_target = reward + (1 - done) * self.gamma * next_value
        td_error = td_target - value

        # Critic 업데이트
        critic_loss = td_error.pow(2)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor 업데이트
        actor_loss = -torch.log(probs[0][action]) * td_error.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

# 학습 루프
agent = ActorCriticAgent(state_dim, action_dim)
num_episodes = 1000
#num_episodes = 500
scores = []

for episode in range(num_episodes):
    state = env.reset()[0]
    score = 0
    for t in range(EPISODE_MAX_LENGTH):
        action, probs = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.train_step(state, action, reward, next_state, done) # 각 스텝마다 학습

        state = next_state
        score += reward

        if done:
            break
    scores.append(score)
    print(f"Episode: {episode + 1}, Score: {score}")

env.close()

torch.save(agent.model.state_dict(), 'cartpole_a2c_model.pth') # 모델 저장
print("Model saved!")

# 결과 그래프 출력
import matplotlib.pyplot as plt
plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.show()