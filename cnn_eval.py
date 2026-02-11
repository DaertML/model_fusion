import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

# --- 1. Re-defining the exact architecture from your training script ---
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
    """
    State: (Player Sum, Dealer Card, Useable Ace)
    Normalization: Player/31, Dealer/10, Ace(0 or 1)
    """
    grid = np.zeros((3, 3))
    grid[0, 0] = state[0] / 31.0
    grid[1, 1] = state[1] / 10.0
    grid[2, 2] = 1.0 if state[2] else 0.0
    return torch.FloatTensor(grid).view(1, 1, 3, 3)

def test_cnn(path="blackjack_cnn.pth", num_rounds=1000):
    env = gym.make('Blackjack-v1')
    model = BlackjackCNN()
    
    # Load the weights
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Successfully loaded: {path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    wins = 0
    draws = 0
    losses = 0
    
    print(f"\nEvaluating CNN for {num_rounds} rounds...")
    
    for i in range(num_rounds):
        obs, _ = env.reset()
        done = False
        
        # Log the first 5 rounds to see what's happening
        if i < 5:
            print(f"\nRound {i+1} Start: Player={obs[0]}, Dealer={obs[1]}, Ace={obs[2]}")
            
        while not done:
            state_img = preprocess_state(obs)
            with torch.no_grad():
                q_values = model(state_img)
                action = q_values.argmax().item()
            
            action_name = "HIT" if action == 1 else "STICK"
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if i < 5:
                print(f"  -> Action: {action_name} | Next State: {obs[0]} | Reward: {reward}")

        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1

    print("-" * 30)
    print(f"RESULTS FOR CNN ALONE:")
    print(f"Wins:   {wins} ({wins/num_rounds:.1%})")
    print(f"Draws:  {draws} ({draws/num_rounds:.1%})")
    print(f"Losses: {losses} ({losses/num_rounds:.1%})")
    print("-" * 30)

if __name__ == "__main__":
    test_cnn()
