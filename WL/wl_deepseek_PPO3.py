import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 하이퍼파라미터
num_cells = 20  # 셀의 개수
num_episodes = 1000  # 에피소드 수
learning_rate = 0.0005  # 학습률
gamma = 0.99  # 할인율
eps_clip = 0.2  # PPO 클리핑 파라미터
K_epochs = 4  # PPO 업데이트 횟수
epsilon_start = 1.0  # 초기 탐험률
epsilon_end = 0.01  # 최소 탐험률
epsilon_decay = 0.995  # 탐험률 감소율

# 그래프 정보 (랜덤으로 생성)
graph = np.random.randint(2, size=(num_cells, num_cells))
np.fill_diagonal(graph, 0)  # 자기 자신과의 연결은 없음

# 와이어 길이 계산 함수
def calculate_wire_length(cells, graph):
    total_length = 0
    for i in range(num_cells):
        for j in range(num_cells):
            if graph[i][j]:
                total_length += abs(np.where(cells == i)[0][0] - np.where(cells == j)[0][0])
    return total_length

# Actor-Critic 신경망
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_cells, 128)  # 입력: 셀 순서
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, num_actions)  # 정책 네트워크
        self.critic = nn.Linear(128, 1)  # 가치 네트워크

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# PPO 에이전트
class PPOAgent:
    def __init__(self, learning_rate=0.0005, gamma=0.99, clip_param=0.2):
        self.gamma = gamma
        self.clip_param = clip_param
        self.num_actions = num_cells * num_cells  # 가능한 행동의 개수
        self.actor_critic = ActorCritic(self.num_actions)  # ActorCritic 초기화
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.memory = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs, state_value = self.actor_critic(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_prob = dist.log_prob(action)
        return action.item(), action_prob, state_value

    def get_value(self, state):
        state_tensor = torch.FloatTensor(state)
        _, state_value = self.actor_critic(state_tensor)
        return state_value.item()

    def save_transition(self, action_prob, state, action, reward, value):
        self.memory.append((action_prob, state, action, reward, value))

    def update_policy(self, optimizer, ppo_epochs, mini_batch_size):
        if self.memory[0]:
            print("probs empty!!")
            return

        states = torch.FloatTensor(np.array([m[1] for m in self.memory]))
        actions = torch.LongTensor(np.array([m[2] for m in self.memory]))
        old_action_probs = torch.FloatTensor(np.array([m[0] for m in self.memory]))
        rewards = torch.FloatTensor(np.array([m[3] for m in self.memory]))
        old_values = torch.FloatTensor(np.array([m[4] for m in self.memory]))

        # Advantage 계산
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        for _ in range(ppo_epochs):
            for index in range(0, len(self.memory), mini_batch_size):
                batch_states = states[index:index + mini_batch_size]
                batch_actions = actions[index:index + mini_batch_size]
                batch_old_probs = old_action_probs[index:index + mini_batch_size]
                batch_advantages = advantages[index:index + mini_batch_size]

                # 새로운 정책과 가치 계산
                new_action_probs, new_values = self.actor_critic(batch_states)
                dist = Categorical(new_action_probs)
                new_probs = dist.log_prob(batch_actions)

                # 정책 손실 계산 (PPO 클리핑)
                ratios = torch.exp(new_probs - batch_old_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실 계산
                value_loss = nn.MSELoss()(new_values.squeeze(), rewards[index:index + mini_batch_size])

                # 전체 손실
                loss = policy_loss + 0.5 * value_loss

                # 역전파 및 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.memory = []  # 메모리 초기화

# 메인 학습 루프
agent = PPOAgent()

# 탐험률 초기화
epsilon = epsilon_start

for episode in range(num_episodes):
    # 초기 상태
    current_cells = np.arange(num_cells)
    done = False
    total_reward = 0

    #while not done:
    for step in range(100):  # 각 에피소드에서 최대 100 스텝
        # 현재 상태에서의 와이어 길이
        current_length = calculate_wire_length(current_cells, graph)

        # 액션 선택 (epsilon-greedy)
        if np.random.random() < epsilon:
            # 무작위 탐험: 랜덤으로 두 셀을 선택하여 교환
            cell1, cell2 = np.random.choice(num_cells, 2, replace=False)
            action = np.ravel_multi_index((cell1, cell2), (num_cells, num_cells))
            log_prob = torch.tensor(0.0)  # 무작위 액션은 로그 확률이 없음
        else:
            # 정책에 따른 액션 선택
            action, log_prob, value = agent.choose_action(current_cells)

        # 셀 교환
        cell1, cell2 = np.unravel_index(action, (num_cells, num_cells))
        current_cells[cell1], current_cells[cell2] = current_cells[cell2], current_cells[cell1]

        # 새로운 상태에서의 와이어 길이
        new_length = calculate_wire_length(current_cells, graph)

        # 보상 계산 (와이어 길이 감소량)
        reward = current_length - new_length

        # 메모리에 저장
        #agent.save_transition(log_prob, current_cells.copy(), action, reward, value)
        agent.save_transition(log_prob, current_cells.copy(), action, reward, new_length)

        total_reward += reward

        # 종료 조건 (와이어 길이가 충분히 작아지면)
        if new_length < 10:
            done = True

    # PPO 업데이트
    agent.update_policy(agent.optimizer, K_epochs, mini_batch_size=32)

    # 탐험률 감소
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Wire Length: {new_length}, Epsilon: {epsilon}")

# 최종 셀 순서 출력
print("Final cell order:")
print(current_cells)