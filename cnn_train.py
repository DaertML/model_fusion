import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPISODES = 1000 
MODEL_PATH = "blackjack_cnn.pth" # Local filename

class BlackjackCNN(nn.Module):
    def __init__(self):
        super(BlackjackCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800, 64), 
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def preprocess_state(state):
    grid = np.zeros((3, 3))
    grid[0, 0] = state[0] / 31.0
    grid[1, 1] = state[1] / 10.0
    grid[2, 2] = 1.0 if state[2] else 0.0
    return torch.FloatTensor(grid).view(1, 1, 3, 3)

# --- Training Loop ---
env = gym.make('Blackjack-v1')
policy_net = BlackjackCNN()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START

print(f"Starting training for {EPISODES} episodes...")

for episode in range(EPISODES):
    obs, info = env.reset()
    state_img = preprocess_state(obs)
    done = False
    
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state_img).argmax().item()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state_img = preprocess_state(next_obs)
        
        memory.append((state_img, action, reward, next_state_img, done))
        state_img = next_state_img
        
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.cat(states)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards).float()
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones).float()
            
            current_q = policy_net(states).gather(1, actions)
            next_q = policy_net(next_states).max(1)[0].detach()
            target_q = rewards + (GAMMA * next_q * (1 - dones))
            
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1} | Epsilon: {epsilon:.2f}")

# --- Save the Model ---
torch.save(policy_net.state_dict(), MODEL_PATH)
print(f"\nModel saved locally to {MODEL_PATH}")

# --- Quick Test ---
print("\nTesting saved model for 5 rounds:")
policy_net.eval() # Set to evaluation mode
for i in range(5):
    obs, _ = env.reset()
    state_img = preprocess_state(obs)
    with torch.no_grad():
        action = policy_net(state_img).argmax().item()
    action_name = "HIT" if action == 1 else "STICK"
    print(f"Round {i+1}: Hand={obs[0]}, Dealer={obs[1]}, Action={action_name}")
