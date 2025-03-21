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
ACTION_SIZE = NUM_CELLS * (NUM_CELLS - 1) // 2
GAMMA = 0.99
LEARNING_RATE = 0.001
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 20
EPISODES = 1000
MAX_STEPS = 50
NUM_EIGENVALUES = 3

# 하이퍼그래프 생성 및 wire length 계산 함수
def create_initial_hypergraph():
    edges = {}
    edge_id = 0
    covered_nodes = set()
    all_nodes = set(range(NUM_CELLS))
    node_edge_count = [0] * NUM_CELLS
    
    while covered_nodes != all_nodes:
        size = random.randint(2, 5)
        available_nodes = [i for i in all_nodes if i not in covered_nodes or node_edge_count[i] < 2]
        if not available_nodes:
            break
        
        nodes = random.sample(available_nodes, min(size, len(available_nodes)))
        edges[f"e{edge_id}"] = nodes
        covered_nodes.update(nodes)
        for n in nodes:
            node_edge_count[n] += 1
        edge_id += 1
    
    return hnx.Hypergraph(edges)

def calculate_hypergraph_wire_length(cells, hypergraph):
    total_length = 0
    for edge in hypergraph.edges():
        nodes = hypergraph.edges[edge]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                total_length += abs(cells[nodes[i]] - cells[nodes[j]])
    return total_length

def get_hypergraph_laplacian(hypergraph):
    adj_matrix = np.zeros((NUM_CELLS, NUM_CELLS))
    degree_matrix = np.zeros((NUM_CELLS, NUM_CELLS))
    
    for edge in hypergraph.edges():
        nodes = hypergraph.edges[edge]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                adj_matrix[nodes[i], nodes[j]] = 1
                adj_matrix[nodes[j], nodes[i]] = 1
    
    degrees = np.sum(adj_matrix, axis=1)
    np.fill_diagonal(degree_matrix, degrees)
    return degree_matrix - adj_matrix

def get_state_features(cells, hypergraph):
    num_hyperedges = len(list(hypergraph.edges()))
    
    # 1. 셀 번호 배열 (정규화)
    cell_positions = np.array(cells, dtype=np.float32) / (NUM_CELLS - 1)
    
    # 2. 하이퍼간선별 평균 거리 (정규화)
    edge_distances = np.zeros(num_hyperedges, dtype=np.float32)
    edge_max_distances = np.zeros(num_hyperedges, dtype=np.float32)  # 추가: 최대 거리
    edge_variances = np.zeros(num_hyperedges, dtype=np.float32)      # 추가: 분산
    for idx, edge in enumerate(hypergraph.edges()):
        nodes = hypergraph.edges[edge]
        distances = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = abs(cells[nodes[i]] - cells[nodes[j]])
                distances.append(dist)
        if distances:
            edge_distances[idx] = np.mean(distances) / (NUM_CELLS - 1)
            edge_max_distances[idx] = np.max(distances) / (NUM_CELLS - 1)
            edge_variances[idx] = np.var(distances) / ((NUM_CELLS - 1) ** 2)  # 분산 정규화
    
    # 3. 노드별 연결된 셀의 평균 위치
    node_connected_positions = np.zeros(NUM_CELLS, dtype=np.float32)
    node_position_deviations = np.zeros(NUM_CELLS, dtype=np.float32)  # 추가: 위치 편차
    for i in range(NUM_CELLS):
        connected_positions = []
        for edge in hypergraph.edges():
            if i in hypergraph.edges[edge]:
                connected_positions.extend([cells[n] for n in hypergraph.edges[edge] if n != i])
        if connected_positions:
            mean_pos = np.mean(connected_positions)
            node_connected_positions[i] = mean_pos / (NUM_CELLS - 1)
            node_position_deviations[i] = abs(cells[i] - mean_pos) / (NUM_CELLS - 1)
    
    # 4. 라플라시안 고유값
    laplacian = get_hypergraph_laplacian(hypergraph)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    top_eigenvalues = eigenvalues[-NUM_EIGENVALUES:] / np.max(eigenvalues)
    
    # 상태 벡터 결합
    state = np.concatenate([
        cell_positions, 
        edge_distances, 
        edge_max_distances, 
        edge_variances, 
        node_connected_positions, 
        node_position_deviations, 
        top_eigenvalues
    ])
    return state, num_hyperedges

def plot_hypergraph(cells, hypergraph, title):
    node_labels = {i: str(cells[i]) for i in range(NUM_CELLS)}
    plt.figure(figsize=(12, 8))
    hnx.draw(hypergraph, with_node_labels=True, node_labels=node_labels)
    plt.title(title)
    plt.show()

# PPO 네트워크 정의
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

# 환경 정의
class WireLength1DEnv:
    def __init__(self, hypergraph):
        self.cells = list(range(NUM_CELLS))
        self.initial_cells = None
        self.hypergraph = hypergraph
        self.state_size = None
        self.reset()
    
    def reset(self):
        random.shuffle(self.cells)
        if self.initial_cells is None:
            self.initial_cells = self.cells.copy()
        self.state, num_hyperedges = get_state_features(self.cells, self.hypergraph)
        if self.state_size is None:
            self.state_size = (NUM_CELLS + num_hyperedges * 3 + NUM_CELLS * 2 + NUM_EIGENVALUES)  # 추가된 특징 반영
        self.total_length = self._calculate_wire_length()
        return self.state
    
    def _calculate_wire_length(self):
        return calculate_hypergraph_wire_length(self.cells, self.hypergraph)
    
    def step(self, action):
        cell_pairs = [(i, j) for i in range(NUM_CELLS) for j in range(i + 1, NUM_CELLS)]
        idx1, idx2 = cell_pairs[action]
        
        self.cells[idx1], self.cells[idx2] = self.cells[idx2], self.cells[idx1]
        new_length = self._calculate_wire_length()
        reward = (self.total_length - new_length) * 10
        if reward < 0:
            reward *= 2
        self.total_length = new_length
        self.state, _ = get_state_features(self.cells, self.hypergraph)
        
        done = False
        return self.state, reward, done

    def inference_step(self, action):
        cell_pairs = [(i, j) for i in range(NUM_CELLS) for j in range(i + 1, NUM_CELLS)]
        idx1, idx2 = cell_pairs[action]
        
        self.cells[idx1], self.cells[idx2] = self.cells[idx2], self.cells[idx1]
        new_length = self._calculate_wire_length()
        self.total_length = new_length
        self.state, _ = get_state_features(self.cells, self.hypergraph)
        
        return self.state, new_length

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
    
    def infer(self, env, max_steps=MAX_STEPS):
        state = env.reset()
        initial_length = env.total_length
        print("\n=== 추론 시작 ===")
        print("추론 초기 셀 배열:", env.initial_cells)
        print(f"추론 초기 Total Wire Length (Hypergraph): {initial_length:.2f}")
        plot_hypergraph(env.initial_cells, env.hypergraph, "Inference Initial Hypergraph")

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                policy, _ = self.model(state_tensor)
            dist = Categorical(policy)
            action = dist.sample().item()
            
            next_state, new_length = env.inference_step(action)
            state = next_state
            
            if step > 10 and abs(new_length - env.total_length) < 1e-5:
                break
        
        final_length = env.total_length
        print("\n추론 최종 셀 배열:", env.cells)
        print(f"추론 최종 Total Wire Length (Hypergraph): {final_length:.2f}")
        plot_hypergraph(env.cells, env.hypergraph, "Inference Final Hypergraph")
        return env.cells, final_length

# 학습 및 추론 통합 함수
def train_and_infer(env, agent, episodes=EPISODES, max_steps=MAX_STEPS):
    print("=== 학습 시작 ===")
    print("초기 셀 배열:", env.initial_cells)
    initial_length = calculate_hypergraph_wire_length(env.initial_cells, env.hypergraph)
    print(f"초기 Total Wire Length (Hypergraph): {initial_length:.2f}")
    print(f"초기 State 크기: {env.state_size}")
    plot_hypergraph(env.initial_cells, env.hypergraph, "Initial Hypergraph")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action, log_prob = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done, log_prob)
            state = next_state
            total_reward += reward
            if done:
                break
        
        agent.learn()
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1}, Total Wire Length: {env.total_length:.2f}")
    
    final_length = env.total_length
    print("\n학습 최종 셀 배열:", env.cells)
    print(f"학습 최종 Total Wire Length (Hypergraph): {final_length:.2f}")
    plot_hypergraph(env.cells, env.hypergraph, "Training Final Hypergraph")
    
    torch.save(agent.model.state_dict(), "ppo_model.pth")
    print("모델이 'ppo_model.pth'로 저장되었습니다.")

    agent.model.eval()
    final_cells, final_length = agent.infer(env)
    return final_cells, final_length

# 메인 실행
if __name__ == "__main__":
    hypergraph = create_initial_hypergraph()
    env = WireLength1DEnv(hypergraph)
    agent = PPOAgent(env.state_size, ACTION_SIZE)
    final_cells, final_length = train_and_infer(env, agent)