import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 하이퍼파라미터
num_cells = 20  # 셀의 개수
num_episodes = 1000  # 에피소드 수
learning_rate = 0.001  # 학습률
gamma = 0.99  # 할인율
eps_clip = 0.2  # PPO 클리핑 파라미터
K_epochs = 4  # PPO 업데이트 횟수

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

# PPO를 위한 신경망 정의
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_cells, 128)  # 입력: 셀 순서
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_cells * num_cells)  # 출력: 모든 가능한 교환 액션에 대한 로짓
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# PPO 알고리즘
def ppo_update(policy, memory, optimizer):
    states = torch.FloatTensor(np.array(memory['states']))
    actions = torch.LongTensor(np.array(memory['actions']))
    rewards = torch.FloatTensor(np.array(memory['rewards']))
    old_log_probs = torch.FloatTensor(np.array(memory['log_probs']))

    # 이전 정책의 로짓 계산
    old_logits = policy(states)
    old_logits = old_logits.view(-1, num_cells * num_cells)  # Flatten
    old_probs = torch.softmax(old_logits, dim=-1)
    old_probs = old_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # 올바른 차원으로 gather

    # PPO 업데이트
    for _ in range(K_epochs):
        logits = policy(states)
        logits = logits.view(-1, num_cells * num_cells)  # Flatten
        probs = torch.softmax(logits, dim=-1)
        new_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # 올바른 차원으로 gather

        # Advantage 계산
        advantages = rewards - rewards.mean()

        # 정책 손실 계산 (PPO 클리핑)
        ratios = new_probs / old_probs.detach()  # old_probs를 detach하여 그래프 분리
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 역전파 및 업데이트
        optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)  # retain_graph=True 추가
        optimizer.step()

# 메인 학습 루프
policy = Policy()
optimizer = policy.optimizer

# Anomaly detection 활성화 (디버깅용)
torch.autograd.set_detect_anomaly(True)

for episode in range(num_episodes):
    # 초기 상태
    current_cells = np.arange(num_cells)
    memory = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}

    for step in range(100):  # 각 에피소드에서 최대 100 스텝
        # 현재 상태에서의 와이어 길이
        current_length = calculate_wire_length(current_cells, graph)

        # 액션 선택
        state = torch.FloatTensor(current_cells)
        logits = policy(state)
        logits = logits.view(num_cells * num_cells)  # Flatten
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()  # 액션 선택 (교환할 두 셀의 인덱스)
        log_prob = dist.log_prob(action)

        # 셀 교환
        cell1, cell2 = np.unravel_index(action.item(), (num_cells, num_cells))
        current_cells[cell1], current_cells[cell2] = current_cells[cell2], current_cells[cell1]

        # 새로운 상태에서의 와이어 길이
        new_length = calculate_wire_length(current_cells, graph)

        # 보상 계산 (와이어 길이 감소량)
        reward = current_length - new_length

        # 메모리에 저장
        memory['states'].append(current_cells.copy())
        memory['actions'].append(action.item())
        memory['rewards'].append(reward)
        memory['log_probs'].append(log_prob.item())

        # 종료 조건 (와이어 길이가 충분히 작아지면)
        if new_length < 10:
            break

    # PPO 업데이트
    ppo_update(policy, memory, optimizer)

    if episode % 10 == 0:
        print(f"Episode {episode}, Wire Length: {calculate_wire_length(current_cells, graph)}, reward: {reward}")

# 최종 셀 순서 출력
print("Final cell order:")
print(current_cells)