import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# GPU 사용 가능 여부 확인 및 디바이스 설정
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

NUM_CELLS = 20

class WireLengthEnv:
    def __init__(self, graph):
        self.graph = graph
        self.num_cells = 20
        self.num_positions = 20
        self.state = list(range(self.num_cells))
        self.initial_wire_length = self.calculate_wire_length(self.state)

    def calculate_wire_length(self, current_state):
        wire_length = 0
        for cell in range(self.num_cells):
            if cell in self.graph:
                for neighbor in self.graph[cell]:
                    cell_pos = current_state.index(cell)
                    neighbor_pos = current_state.index(neighbor)
                    wire_length += abs(cell_pos - neighbor_pos)
        return wire_length / 2

    def step(self, action):
        cell1_idx, cell2_idx = action
        next_state = self.state[:]
        pos1 = next_state.index(cell1_idx)
        pos2 = next_state.index(cell2_idx)
        next_state[pos1], next_state[pos2] = next_state[pos2], next_state[pos1]

        current_wire_length = self.calculate_wire_length(self.state)
        next_wire_length = self.calculate_wire_length(next_state)
        reward = current_wire_length - next_wire_length
        self.state = next_state
        return self.state, reward, False

    def reset(self):
        self.state = list(range(self.num_cells))
        return self.state

    def get_possible_actions(self):
        actions = []
        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells):
                actions.append((i, j))
        return actions


class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def forward(self, state):
        action_probs = self.actor(state)
        action_probs = torch.softmax(action_probs, dim=-1)
        state_value = self.critic(state)
        return action_probs, state_value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class PPOAgent:
    def __init__(self, env, learning_rate=1e-5, gamma=0.99, clip_param=0.2,
                 initial_exploration_rate=0.3, exploration_decay_rate=0.00001): # Reduced initial exploration, increased decay
        self.env = env
        self.gamma = gamma
        self.clip_param = clip_param
        self.num_actions = len(env.get_possible_actions())
        self.policy = ActorCritic(self.num_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.buffer_actions = []
        self.buffer_states = []
        self.buffer_rewards = []
        self.buffer_probs = []
        self.buffer_values = []

        self.exploration_rate = initial_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.min_exploration_rate = 0.01

    def choose_action(self, state):
        """
        정책 네트워크 또는 랜덤 액션 선택 (exploration 적용)
        """
        if random.random() < self.exploration_rate:
            possible_actions = self.env.get_possible_actions()
            action = random.choice(possible_actions)
            action_prob = None
            state_value = None
            return action, action_prob, state_value

        else:
            state_tensor = torch.FloatTensor(self.state_to_tensor(state)).unsqueeze(0).to(DEVICE)
            action_probs, state_value = self.policy(state_tensor)

            action_distribution = Categorical(action_probs)
            action_index = action_distribution.sample()
            action_log_prob = action_distribution.log_prob(action_index)

            possible_actions = self.env.get_possible_actions()
            action = possible_actions[action_index.item()]
            return action, action_log_prob, state_value

    def state_to_tensor(self, state):
        return np.array(state)

    def get_value(self, state):
        state_tensor = torch.FloatTensor(self.state_to_tensor(state)).unsqueeze(0).to(DEVICE)
        _, state_value = self.policy(state_tensor)
        return state_value

    def save_transition(self, action_prob, state, action, reward, value):
        """
        trajectory buffer 에 transition 저장 (POLICY-BASED ACTIONS ONLY)
        """
        if action_prob is not None and value is not None: # Only save policy-based actions
            # print("Saving Policy-Based Transition - action_prob:", action_prob) # Debug print - uncomment if needed
            self.buffer_probs.append(action_prob)
            self.buffer_states.append(state)
            self.buffer_actions.append(action)
            self.buffer_rewards.append(reward)
            self.buffer_values.append(value)
        # else:
        #     print("NOT Saving Random Action Transition") # Debug print - uncomment if needed


    def update_policy(self, optimizer, ppo_epochs, mini_batch_size):
        """
        PPO 알고리즘으로 정책 업데이트
        """
        np_values = np.array([v.cpu().detach().numpy() for v in self.buffer_values if v is not None])
        rewards_np = np.array(self.buffer_rewards)

        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards_np):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = np.array(discounted_rewards)


        # Debugging prints (Uncomment if needed)
        # print("Shape of discounted_rewards (before calc):", discounted_rewards.shape)
        # print("Shape of np_values.flatten() (before calc):", np_values.flatten().shape if np_values.size > 0 else (0,))
        # print("Length of buffer_rewards:", len(self.buffer_rewards))
        # print("Length of buffer_values (original):", len(self.buffer_values))
        # print("Length of np_values (filtered):", len(np_values))
        # print("Length of self.buffer_probs BEFORE stack:", len(self.buffer_probs))


        advantages = discounted_rewards - np_values.flatten() if np_values.size > 0 else np.array([])


        tensor_probs = torch.empty((0, self.num_actions)).to(DEVICE).detach() # Handle empty buffer_probs - create empty tensor
        if self.buffer_probs: # Stack only if buffer_probs is not empty
            tensor_probs = torch.stack([prob for prob in self.buffer_probs if prob is not None]).to(DEVICE).detach()
        else:
            print("Warning: buffer_probs is empty, creating empty tensor_probs to continue (no update for this batch).")


        tensor_states = torch.FloatTensor(np.array(self.buffer_states)).to(DEVICE)
        tensor_actions_idx = torch.LongTensor(np.array([self.env.get_possible_actions().index(act) for act in self.buffer_actions])).to(DEVICE)
        tensor_advantages = torch.FloatTensor(advantages).to(DEVICE) if advantages.size > 0 else torch.tensor([]).to(DEVICE)
        tensor_discounted_rewards = torch.FloatTensor(discounted_rewards).to(DEVICE)

        # Skip update if advantages is empty - already handled by tensor_probs being empty (no gradients computed if no probs)
        if tensor_probs.numel() == 0: # Check tensor_probs instead of tensor_advantages
            print("Warning: tensor_probs is empty, skipping mini-batch update.")
            self.buffer_actions.clear() # Clear buffers even if skipping update to start fresh next time
            self.buffer_states.clear()
            self.buffer_probs.clear()
            self.buffer_rewards.clear()
            self.buffer_values.clear()
            self.decay_exploration_rate()
            return # Early return to skip update


        for _ in range(ppo_epochs):
            mini_batch_indices = np.random.permutation(len(self.buffer_rewards))
            for batch_start in range(0, len(self.buffer_rewards), mini_batch_size):
                batch_indices = mini_batch_indices[batch_start:batch_start + mini_batch_size]

                batch_states = tensor_states[batch_indices]
                batch_actions_idx = tensor_actions_idx[batch_indices]
                batch_advantages = tensor_advantages[batch_indices]
                batch_old_probs = tensor_probs[batch_indices]
                batch_discounted_rewards = tensor_discounted_rewards[batch_indices]

                optimizer.zero_grad()
                batch_action_probs, batch_state_values = self.policy(batch_states)
                batch_action_probs_selected = batch_action_probs.gather(1, batch_actions_idx.unsqueeze(1)).squeeze(1)

                ratio = batch_action_probs_selected / batch_old_probs
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(batch_state_values.squeeze(1), batch_discounted_rewards)

                loss = actor_loss + critic_loss
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0) # Gradient Clip
                loss.backward()
                optimizer.step()

        self.buffer_actions.clear()
        self.buffer_states.clear()
        self.buffer_probs.clear()
        self.buffer_rewards.clear()
        self.buffer_values.clear()

        self.decay_exploration_rate()

    def decay_exploration_rate(self):
        self.exploration_rate -= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)


# 예시 그래프
example_graph = {
    0: [1, 2], 1: [0, 3, 4], 2: [0, 5], 3: [1, 6], 4: [1, 7],
    5: [2, 8], 6: [3, 9], 7: [4, 10], 8: [5, 11], 9: [6, 12],
    10: [7, 13], 11: [8, 14], 12: [9, 15], 13: [10, 16], 14: [11, 17],
    15: [12, 18], 16: [13, 19], 17: [14], 18: [15], 19: [16]
}

graph = np.random.randint(2, size=(NUM_CELLS, NUM_CELLS))
np.fill_diagonal(graph, 0)  # 자기 자신과의 연결은 없음

# 환경 및 PPO 에이전트 생성
#env = WireLengthEnv(example_graph)
env = WireLengthEnv(graph)
agent = PPOAgent(env)

# 학습 하이퍼파라미터
num_episodes = 1000
print_interval = 10
ppo_epochs = 3
mini_batch_size = 64
optimizer = agent.optimizer

best_wire_length = env.initial_wire_length
best_state = env.state[:]

print(f"초기 Exploration Rate: {agent.exploration_rate:.2f}")
print(f"초기 와이어 길이: {best_wire_length}")


# 학습 루프
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(200):
        action, action_prob, value = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.save_transition(action_prob, state, action, reward, value) # Save transitions (policy-based only now)

        state = next_state
        episode_reward += reward

        if t % 4 == 0:
            agent.update_policy(optimizer, ppo_epochs, mini_batch_size)

        if done:
            break

    current_wire_length = env.calculate_wire_length(state)

    if current_wire_length < best_wire_length:
        best_wire_length = current_wire_length
        best_state = state[:]

    if episode % print_interval == 0 and episode != 0:
        print(f"Episode: {episode}, Avg Reward: {episode_reward / print_interval:.2f}, Best Wire Length: {best_wire_length:.2f}, Exploration Rate: {agent.exploration_rate:.4f}")


print(f"\n최적 와이어 길이: {best_wire_length}")
print(f"최적 셀 배치 상태: {best_state}")