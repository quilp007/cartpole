import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

# Actor-Critic 네트워크
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# 환경 및 모델 설정
env = gym.make('CartPole-v1', render_mode="rgb_array") # render_mode 추가
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 학습된 모델 불러오기
model = ActorCritic(state_dim, action_dim)
model.load_state_dict(torch.load('cartpole_ppo_model.pth')) # 저장된 모델 파일명
model.eval()
model.to(device)

# CartPole 제어 함수
def run_cartpole(model, env, num_episodes=10):
    for episode in range(num_episodes):
        state = env.reset()[0]
        score = 0
        done = False

        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state)
            action = q_values.argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            score += reward
            env.render() # 화면에 렌더링

        print(f"Episode: {episode + 1}, Score: {score}")
    env.close()

# CartPole 실행 및 애니메이션 생성 함수
def run_cartpole_with_animation(model, env, num_episodes=1):
    for episode in range(num_episodes):
        state = env.reset()[0]
        frames = [] # 프레임 저장 리스트

        done = False
        while not done:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
               probs, _ = model(state)  # 수정: Actor 출력(probs)만 가져옴
            action = probs.argmax().item()  # 수정: probs에 argmax() 적용

            next_state, reward, done, _, _ = env.step(action)
            frames.append(env.render()) # 프레임 저장
            state = next_state

        # 애니메이션 생성
        fig = plt.figure()
        plt.axis('off')
        im = plt.imshow(frames[0])

        def animate(i):
            im.set_data(frames[i])
            return [im]

        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=20, blit=True, repeat=False)

        plt.show() # 애니메이션 출력
        # anim.save('cartpole_animation.gif', writer='imagemagick', fps=60) # GIF로 저장 (필요시)
    env.close()

# CartPole 실행
#run_cartpole(model, env)
run_cartpole_with_animation(model, env)
