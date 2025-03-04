import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# GPU 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

#if torch.backends.mps.is_available():
#    DEVICE = torch.device("mps")
#elif torch.cuda.is_available():
#    DEVICE = torch.device("cuda")
#else:
#    DEVICE = torch.device("cpu")

class WireLengthEnv:
    def __init__(self, graph):
        """
        와이어 길이 환경 초기화

        Args:
            graph (dict): 그래프 정보 (인접 리스트 형태)
        """
        self.graph = graph
        self.num_cells = 20 # 셀 개수
        self.num_positions = 20 # 위치 개수
        self.state = list(range(self.num_cells)) # 초기 상태: 셀 i는 위치 i에 배치
        self.initial_wire_length = self.calculate_wire_length(self.state) # 초기 와이어 길이

    def calculate_wire_length(self, current_state):
        """
        현재 상태의 와이어 길이 계산

        Args:
            current_state (list): 현재 셀 위치 배치 상태

        Returns:
            float: 와이어 길이
        """
        wire_length = 0
        for cell in range(self.num_cells):
            if cell in self.graph:
                for neighbor in self.graph[cell]:
                    cell_pos = current_state.index(cell) # 셀의 현재 위치
                    neighbor_pos = current_state.index(neighbor) # 이웃 셀의 현재 위치
                    wire_length += abs(cell_pos - neighbor_pos) # 위치 차이 (거리) 누적
        return wire_length / 2 # 중복 계산 방지 (양방향 연결 고려)

    def step(self, action):
        """
        환경에 행동을 적용하고 다음 상태, 보상, 종료 여부 반환

        Args:
            action (tuple): 행동 (바꿀 셀 인덱스 2개)

        Returns:
            tuple: 다음 상태, 보상, 종료 여부 (항상 False)
        """
        cell1_idx, cell2_idx = action
        next_state = self.state[:] # 현재 상태 복사
        # 두 셀 위치 교환
        pos1 = next_state.index(cell1_idx)
        pos2 = next_state.index(cell2_idx)
        next_state[pos1], next_state[pos2] = next_state[pos2], next_state[pos1]

        current_wire_length = self.calculate_wire_length(self.state)
        next_wire_length = self.calculate_wire_length(next_state)
        reward = current_wire_length - next_wire_length # 보상: 와이어 길이 감소량
        self.state = next_state # 상태 업데이트

        return self.state, reward, False # 종료 조건 없음 (계속 학습)

    def reset(self):
        """
        환경 초기화 (상태 초기화)

        Returns:
            list: 초기 상태
        """
        self.state = list(range(self.num_cells)) # 초기 상태: 셀 i는 위치 i에 배치
        return self.state

    def get_possible_actions(self):
        """
        가능한 모든 행동 반환 (두 셀을 선택하여 위치를 바꾸는 모든 조합)

        Returns:
            list: 가능한 행동 리스트 (각 행동은 (cell1_idx, cell2_idx) 튜플)
        """
        actions = []
        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells): # 중복 방지 및 순서 무관
                actions.append((i, j))
        return actions

# Actor-Critic 네트워크 (PPO 에이전트에서 사용)
class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()

        # Actor 네트워크 (정책 신경망)
        self.actor = nn.Sequential(
            nn.Linear(20, 128),  # 입력: 상태 (20개 셀 위치 정보), 출력: 정책
            nn.ReLU(),
            nn.Linear(128, num_actions) # 출력: 각 행동에 대한 확률 (총 가능한 행동 개수)
        )

        # Critic 네트워크 (가치 신경망)
        self.critic = nn.Sequential(
            nn.Linear(20, 128),  # 입력: 상태, 출력: 가치 함수
            nn.ReLU(),
            nn.Linear(128, 1)     # 출력: 상태 가치 V(s)
        )

    def forward(self, state):
        """
        순전파 연산

        Args:
            state (torch.Tensor): 상태 (원-핫 인코딩 또는 직접적인 위치 정보)

        Returns:
            tuple: 행동에 대한 확률 분포, 상태 가치
        """
        action_probs = self.actor(state)
        action_probs = torch.softmax(action_probs, dim=-1) # Softmax 를 통해 확률 분포로 변환
        state_value = self.critic(state)

        return action_probs, state_value

# PPO 에이전트
class PPOAgent:
    def __init__(self, env, learning_rate=0.0005, gamma=0.99, clip_param=0.2):
        self.env = env
        self.gamma = gamma
        self.clip_param = clip_param
        self.num_actions = len(env.get_possible_actions()) # 가능한 행동의 개수
        self.policy = ActorCritic(self.num_actions).to(DEVICE) # Actor-Critic 네트워크 생성 및 GPU로 이동
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate) # Adam 옵티마이저 사용

        # trajectory 저장을 위한 buffer
        self.buffer_actions = []
        self.buffer_states = []
        self.buffer_rewards = []
        self.buffer_probs = []
        self.buffer_values = []

    def choose_action(self, state):
        """
        정책 네트워크를 사용하여 행동 선택

        Args:
            state (list): 현재 상태

        Returns:
            tuple: 선택된 행동 (바꿀 셀 인덱스 2개), 행동 확률
        """
        state_tensor = torch.FloatTensor(self.state_to_tensor(state)).unsqueeze(0).to(DEVICE) # 상태를 Tensor로 변환 및 GPU로 이동
        action_probs, state_value = self.policy(state_tensor) # 정책 네트워크 예측
        action_distribution = Categorical(action_probs) # 확률 분포 생성
        action_index = action_distribution.sample() # 행동 샘플링
        action_log_prob = action_distribution.log_prob(action_index) # 선택된 행동의 로그 확률

        possible_actions = self.env.get_possible_actions()
        action = possible_actions[action_index.item()] # 행동 인덱스를 실제 행동으로 변환

        return action, action_log_prob, state_value

    def state_to_tensor(self, state):
        """
        상태 리스트를 신경망 입력 텐서로 변환 (여기서는 간단하게 원-핫 인코딩 대신 직접적인 위치 정보 사용)

        Args:
            state (list): 상태 리스트

        Returns:
            torch.Tensor: 신경망 입력 텐서
        """
        return np.array(state) # numpy array 로 변환 후 tensor 로 변환 시 torch.FloatTensor 사용

    def get_value(self, state):
        """
        가치 네트워크를 사용하여 상태 가치 예측

        Args:
            state (list): 상태

        Returns:
            torch.Tensor: 상태 가치 V(s)
        """
        state_tensor = torch.FloatTensor(self.state_to_tensor(state)).unsqueeze(0).to(DEVICE) # 상태를 Tensor로 변환 및 GPU로 이동
        _, state_value = self.policy(state_tensor) # 정책 네트워크 (critic 네트워크 포함) 예측
        return state_value

    def save_transition(self, action_prob, state, action, reward, value):
        """
        trajectory buffer 에 transition 저장

        Args:
            action_prob (torch.Tensor): 행동 확률
            state (list): 상태
            action (tuple): 행동
            reward (float): 보상
            value (torch.Tensor): 상태 가치
        """
        self.buffer_probs.append(action_prob)
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_values.append(value)

    def update_policy(self, optimizer, ppo_epochs, mini_batch_size):
        """
        PPO 알고리즘으로 정책 업데이트

        Args:
            optimizer (torch.optim.Optimizer): Optimizer
            ppo_epochs (int): PPO epoch 횟수
            mini_batch_size (int): 미니 배치 크기
        """
        # numpy array 로 변환
        np_values = np.array([v.cpu().detach().numpy() for v in self.buffer_values])
        rewards_np = np.array(self.buffer_rewards)

        # Advantage 계산: GAE (Generalized Advantage Estimation) 사용 가능, 여기서는 간단하게 TD(lambda) 와 유사하게 advantage 계산
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards_np): # rewards 를 뒤집어서 순회
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward) # 앞에 계속 삽입

        discounted_rewards = np.array(discounted_rewards)
        advantages = discounted_rewards - np_values.flatten() # Advantage 계산

        # Tensor 로 변환 및 GPU 로 이동
        tensor_probs = torch.stack(self.buffer_probs).to(DEVICE).detach() # detach: gradient 계산에서 제외
        tensor_states = torch.FloatTensor(np.array(self.buffer_states)).to(DEVICE) # 상태 tensor 변환 및 GPU 이동
        tensor_actions_idx = torch.LongTensor(np.array([self.env.get_possible_actions().index(act) for act in self.buffer_actions])).to(DEVICE) # 행동 index tensor 변환 및 GPU 이동
        tensor_advantages = torch.FloatTensor(advantages).to(DEVICE)
        tensor_discounted_rewards = torch.FloatTensor(discounted_rewards).to(DEVICE)

        # PPO Epoch 반복
        for _ in range(ppo_epochs):
            # Mini-batch index 생성
            mini_batch_indices = np.random.permutation(len(self.buffer_rewards))
            for batch_start in range(0, len(self.buffer_rewards), mini_batch_size):
                batch_indices = mini_batch_indices[batch_start:batch_start + mini_batch_size]

                # Mini-batch 데이터 추출
                batch_states = tensor_states[batch_indices]
                batch_actions_idx = tensor_actions_idx[batch_indices]
                batch_advantages = tensor_advantages[batch_indices]
                batch_old_probs = tensor_probs[batch_indices]
                batch_discounted_rewards = tensor_discounted_rewards[batch_indices]

                # Policy 업데이트
                optimizer.zero_grad() # Optimizer gradient 초기화
                batch_action_probs, batch_state_values = self.policy(batch_states) # 현재 정책으로 action probability, state value 계산
                batch_action_probs_selected = batch_action_probs.gather(1, batch_actions_idx.unsqueeze(1)).squeeze(1) # 선택된 행동에 대한 확률 추출

                ratio = batch_action_probs_selected / batch_old_probs # 확률 비율 계산 (r_t)
                surr1 = ratio * batch_advantages # surrogate objective 1
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages # clipped surrogate objective 2
                actor_loss = -torch.min(surr1, surr2).mean() # Actor loss: surrogate objective 1과 2 중 작은 값 선택 (최소화)

                # Value 업데이트
                critic_loss = nn.MSELoss()(batch_state_values.squeeze(1), batch_discounted_rewards) # Critic loss: MSE loss

                # 총 Loss 계산 및 Backpropagation
                loss = actor_loss + critic_loss # Total loss = Actor loss + Critic loss
                loss.backward() # Backpropagation
                optimizer.step() # Optimizer step (parameter 업데이트)

        # buffer 비우기
        self.buffer_actions.clear()
        self.buffer_states.clear()
        self.buffer_probs.clear()
        self.buffer_rewards.clear()
        self.buffer_values.clear()


# 예시 그래프 (임의로 생성 - 실제 그래프 정보로 대체해야 함)
example_graph = {
    0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1, 6], 4: [1, 7],
    5: [2, 8], 6: [3, 9], 7: [4, 10], 8: [5, 11], 9: [6, 12],
    10: [7, 13], 11: [8, 14], 12: [9, 15], 13: [10, 16], 14: [11, 17],
    15: [12, 18], 16: [13, 19], 17: [14], 18: [15], 19: [16]
}

# 환경 및 PPO 에이전트 생성
env = WireLengthEnv(example_graph)
agent = PPOAgent(env)

# 학습 하이퍼파라미터
#num_episodes = 10000
#print_interval = 100

num_episodes = 1000
print_interval = 10

ppo_epochs = 3 # PPO epoch 횟수
mini_batch_size = 64 # 미니 배치 크기
optimizer = agent.optimizer

best_wire_length = env.initial_wire_length
best_state = env.state[:] # 초기 상태를 최적 상태로 가정

print(f"초기 와이어 길이: {best_wire_length}")

# 학습 루프
for episode in range(num_episodes):
    state = env.reset() # 환경 초기화
    episode_reward = 0

    for t in range(200): # time step 제한 (optional, 에피소드가 너무 길어지는 것 방지)
        action, action_prob, value = agent.choose_action(state) # 행동 선택 (PPO Agent)
        next_state, reward, done = env.step(action) # 환경과 상호작용

        agent.save_transition(action_prob, state, action, reward, value) # trajectory buffer 에 transition 저장

        state = next_state # 상태 업데이트
        episode_reward += reward # 에피소드 보상 누적

        if t % 4 == 0: # trajectory 가 어느정도 쌓이면 policy update (hyperparameter tuning 필요)
            agent.update_policy(optimizer, ppo_epochs, mini_batch_size) # PPO 정책 업데이트

        if done: # 종료 조건 (현재 환경에는 종료 조건 없음)
            break

    current_wire_length = env.calculate_wire_length(state) # 현재 상태의 와이어 길이 계산

    if current_wire_length < best_wire_length: # 현재 와이어 길이가 최적 와이어 길이보다 작으면
        best_wire_length = current_wire_length # 최적 와이어 길이 업데이트
        best_state = state[:] # 최적 상태 업데이트


    if episode % print_interval == 0 and episode != 0:
        print(f"Episode: {episode}, Avg Reward: {episode_reward/print_interval:.2f}, Best Wire Length: {best_wire_length:.2f}")
        print(f"셀 배치 상태: {best_state}")


print(f"\n최적 와이어 길이: {best_wire_length}")
print(f"최적 셀 배치 상태: {best_state}")