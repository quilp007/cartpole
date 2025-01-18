# cartpole
- RL with cartpole

# DQN 1
- 1 model
```python
q_values = self.model(states).gather(1, actions)
next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
target_q_values = rewards + (1 - dones) * gamma * next_q_values

loss = nn.MSELoss()(q_values, target_q_values)
```
# DQN 2
- 2 model
```python
q_values = self.q_net(states).gather(1, actions)
next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

loss = nn.MSELoss()(q_values, target_q_values)
```
# PG
- Policy Gradient
```python
for r in reversed(rewards): # 보상을 뒤에서부터 계산
    cumulative_reward = r + self.gamma * cumulative_reward
    discounted_rewards.insert(0, cumulative_reward)
discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8) # 정규화

log_probs = torch.stack(log_probs).to(device)
policy_loss = -(log_probs * discounted_rewards).mean()
```
# AC
- Actor-Critic
```python
probs, value = self.model(state)
_, next_value = self.model(next_state)

td_target = reward + (1 - done) * self.gamma * next_value
td_error = td_target - value

critic_loss = td_error.pow(2)
actor_loss = -torch.log(probs[0][action]) * td_error.detach()
```
# A2C
- Advanced Actor-Critic
```python
probs, values = self.model(states)
_, next_values = self.model(next_states)

advantages = rewards + (1 - dones) * self.gamma * next_values - values
actor_loss = -(torch.log(probs.gather(1, actions)) * advantages.detach()).mean()
critic_loss = advantages.pow(2).mean()

loss = actor_loss + critic_loss
```
# PPO
- Proximal Policy Optimization
```python
probs, values = self.model(states)
_, next_values = self.model(next_states)

advantages = rewards + (1 - dones) * GAMMA * next_values - values
ratios = torch.exp(torch.log(probs.gather(1, actions)) - torch.log(old_probs.gather(1, actions)))
surr1 = ratios * advantages.detach()
surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages.detach()
actor_loss = -torch.min(surr1, surr2).mean()
critic_loss = advantages.pow(2).mean()

loss = actor_loss + critic_loss
```
