import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import hypernetx as hnx
import matplotlib.pyplot as plt

# 하이퍼파라미터
NUM_CELLS = 20
STATE_SIZE = NUM_CELLS
ACTION_SIZE = NUM_CELLS * (NUM_CELLS - 1) // 2
GAMMA = 0.99
LEARNING_RATE = 0.0003
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 10
EPISODES = 500
MAX_STEPS = 50
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# 하이퍼그래프 생성 및 wire length 계산 함수
def create_initial_hypergraph():
    edges = {}
    edge_id = 0
    covered_nodes = set()
    all_nodes = set(range(NUM_CELLS))
    
    # 모든 노드가 최소 하나의 하이퍼간선에 포함되도록
    while covered_nodes != all_nodes:
        size = random.randint(2, 5)
        available_nodes = list(all_nodes - covered_nodes)
        if not available_nodes:
            available_nodes = list(all_nodes)
        
        nodes = random.sample(available_nodes, min(size, len(available_nodes)))
        if len(nodes) < 2 and covered_nodes:
            extra_node = random.choice(list(covered_nodes))
            nodes.append(extra_node)
        
        edges[f"e{edge_id}"] = nodes
        covered_nodes.update(nodes)
        edge_id += 1
    
    # 추가 연결 생성
    for _ in range(3):
        size = random.randint(2, 5)
        nodes = random.sample(range(NUM_CELLS), size)
        edges[f"e{edge_id}"] = nodes
        edge_id += 1
    
    return hnx.Hypergraph(edges)

def calculate_hypergraph_wire_length(cells, hypergraph):
    total_length = 0
    for edge in hypergraph.edges():
        nodes = hypergraph.edges[edge]
        # 하이퍼간선 내 모든 노드 쌍의 거리 합산
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                total_length += abs(cells[nodes[i]] - cells[nodes[j]])
    return total_length

def plot_hypergraph(cells, hypergraph, title):
    node_labels = {i: str(cells[i]) for i in range(NUM_CELLS)}
    plt.figure(figsize=(12, 8))
    hnx.draw(hypergraph, with_node_labels=True, node_labels=node_labels)
    plt.title(title)
    plt.show()

# 환경 정의
class WireLength1DEnv:
    def __init__(self):
        self.cells = list(range(NUM_CELLS))
        self.initial_cells = None
        self.initial_hypergraph = None
        self.reset()
    
    def reset(self):
        random.shuffle(self.cells)
        self.state = np.array(self.cells, dtype=np.float32)
        if self.initial_cells is None:
            self.initial_cells = self.cells.copy()
            self.initial_hypergraph = create_initial_hypergraph()  # 고정된 하이퍼그래프
        self.total_length = self._calculate_wire_length()
        return self.state
    
    def _calculate_wire_length(self):
        return calculate_hypergraph_wire_length(self.cells, self.initial_hypergraph)
    
    def step(self, action):
        cell_pairs = [(i, j) for i in range(NUM_CELLS) for j in range(i + 1, NUM_CELLS)]
        idx1, idx2 = cell_pairs[action]
        
        self.cells[idx1], self.cells[idx2] = self.cells[idx2], self.cells[idx1]
        new_length = self._calculate_wire_length()
        reward = self.total_length - new_length  # 길이 감소량을 보상으로
        self.total_length = new_length
        self.state = np.array(self.cells, dtype=np.float32)
        
        done = False
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
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
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
        value = self.critic(state)
        return policy, value

# PPO 에이전트
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.model = PPONetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = []
        self.epsilon = EPSILON_START
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1), 0.0
        else:
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
    
    def update_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

# 학습 실행 및 시각화
env = WireLength1DEnv()
agent = PPOAgent(STATE_SIZE, ACTION_SIZE)

# 초기 상태 출력 및 하이퍼그래프
print("초기 셀 배열:", env.initial_cells)
initial_length = calculate_hypergraph_wire_length(env.initial_cells, env.initial_hypergraph)
print(f"초기 Total Wire Length (Hypergraph): {initial_length:.2f}")
plot_hypergraph(env.initial_cells, env.initial_hypergraph, "Initial Hypergraph")

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
    agent.update_epsilon()
    if (episode + 1) % 50 == 0:
        print(f"Episode: {episode + 1}, Total Wire Length: {env.total_length:.2f}, Epsilon: {agent.epsilon:.4f}")

# 최종 결과 출력 및 하이퍼그래프
print("\n최종 셀 배열:", env.cells)
final_length = calculate_hypergraph_wire_length(env.cells, env.initial_hypergraph)
print(f"최종 Total Wire Length (Hypergraph): {final_length:.2f}")
plot_hypergraph(env.cells, env.initial_hypergraph, "Final Hypergraph")