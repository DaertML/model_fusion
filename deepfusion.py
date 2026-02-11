import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque
from transformers import GPT2Model

# --- 1. Model Architecture ---

class BlackjackCNN(nn.Module):
    """ Matches your training script's architecture """
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

class GluedSandwichVLA(nn.Module):
    def __init__(self, cnn_path="blackjack_cnn.pth"):
        super().__init__()
        
        # A. LOAD & FREEZE CNN
        self.cnn_base = BlackjackCNN()
        try:
            self.cnn_base.load_state_dict(torch.load(cnn_path))
            print(f"Successfully loaded weights from {cnn_path}")
        except FileNotFoundError:
            print(f"Warning: {cnn_path} not found. Using random weights for demo.")
            
        # We strip the last layer to get 64-dim features
        self.cnn_features = nn.Sequential(*list(self.cnn_base.conv.children()), 
                                         nn.Flatten(), 
                                         self.cnn_base.fc[1]) 
        for param in self.cnn_features.parameters():
            param.requires_grad = False
            
        # B. LOAD & FREEZE LLM (GPT-2)
        full_gpt = GPT2Model.from_pretrained('gpt2')
        self.embed = full_gpt.wte
        self.llm_part1 = full_gpt.h[:6]  # Layers 0-5
        self.llm_part2 = full_gpt.h[6:]  # Layers 6-11
        self.ln_f = full_gpt.ln_f
        
        for param in full_gpt.parameters():
            param.requires_grad = False

        # C. TRAINABLE PROJECTIONS (The Glue)
        self.proj_in = nn.Linear(64, 768)   # CNN feature -> LLM residual stream
        self.proj_out = nn.Linear(768, 2)  # LLM state -> Blackjack Action
        
    def forward(self, x):
        # 1. Extract visual features
        features = self.cnn_features(x).unsqueeze(1) # [B, 1, 64]
        
        # 2. Project to LLM space
        v_token = self.proj_in(features) # [B, 1, 768]
        
        # 3. First half of LLM
        hidden = self.embed(torch.zeros((x.size(0), 1), dtype=torch.long)) 
        for block in self.llm_part1:
            hidden = block(hidden)[0]
            
        # 4. Sandwich: Inject the CNN token into the LLM flow
        fused = torch.cat([hidden, v_token], dim=1)
        
        # 5. Second half of LLM
        for block in self.llm_part2:
            fused = block(fused)[0]
            
        # 6. Final Decision
        fused = self.ln_f(fused)
        return self.proj_out(fused[:, -1, :])

# --- 2. Training Logic for Projections ---

def preprocess_state(state):
    grid = np.zeros((3, 3))
    grid[0, 0] = state[0] / 31.0
    grid[1, 1] = state[1] / 10.0
    grid[2, 2] = 1.0 if state[2] else 0.0
    return torch.FloatTensor(grid).view(1, 1, 3, 3)

def run_experiment():
    env = gym.make('Blackjack-v1')
    vla = GluedSandwichVLA("blackjack_cnn.pth")
    
    # Only optimize the projection layers
    optimizer = optim.Adam(list(vla.proj_in.parameters()) + 
                           list(vla.proj_out.parameters()), lr=1e-3)
    
    print("\nTraining projection layers for 500 episodes...")
    for ep in range(500):
        obs, _ = env.reset()
        done = False
        while not done:
            state_img = preprocess_state(obs)
            logits = vla(state_img)
            action = logits.argmax().item()
            
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # Simple policy gradient / Q-style loss for the glue
            with torch.no_grad():
                next_q = vla(preprocess_state(next_obs)).max()
                target = reward + (0.95 * next_q * (1 - int(done)))
            
            loss = nn.MSELoss()(logits[0, action], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            obs = next_obs
            
    # Evaluation
    print("\nEvaluating the Glued Model (100 rounds):")
    wins = 0
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = vla(preprocess_state(obs)).argmax().item()
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            if reward > 0: wins += 1
    print(f"Final Win Rate: {wins}%")

if __name__ == "__main__":
    run_experiment()
