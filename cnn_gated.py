import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from transformers import GPT2Model

# --- 1. The CNN (Frozen) ---
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

# --- 2. Gated Fusion Module ---
class GatedFusion(nn.Module):
    def __init__(self, embed_dim=768, feat_dim=64):
        super().__init__()
        self.proj_v = nn.Linear(feat_dim, embed_dim)
        # The Gate: Takes both LLM and CNN state to decide the mix
        self.gate_layer = nn.Sequential(
            nn.Linear(embed_dim + feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_llm, x_cnn):
        # x_llm: [B, 768], x_cnn: [B, 64]
        v_projected = self.proj_v(x_cnn)
        
        # Calculate gate value g based on the context of both models
        combined = torch.cat([x_llm, x_cnn], dim=-1)
        g = self.gate_layer(combined) # Value between 0 and 1
        
        # Weighted blend
        return (1 - g) * x_llm + g * v_projected

# --- 3. The Gated VLA Model ---
class GatedBlackjackVLA(nn.Module):
    def __init__(self, cnn_path="blackjack_cnn.pth"):
        super().__init__()
        
        # A. Load Frozen CNN
        self.cnn_base = BlackjackCNN()
        try:
            self.cnn_base.load_state_dict(torch.load(cnn_path))
        except: pass
        
        # Strip final layer to get 64-dim features
        self.cnn_feat = nn.Sequential(
            self.cnn_base.conv, nn.Flatten(), 
            self.cnn_base.fc[0], self.cnn_base.fc[1]
        )
        for param in self.cnn_feat.parameters(): param.requires_grad = False

        # B. Load Frozen LLM
        self.llm = GPT2Model.from_pretrained('gpt2')
        for param in self.llm.parameters(): param.requires_grad = False

        # C. Trainable Gated Bridge
        self.fusion = GatedFusion(768, 64)
        self.action_head = nn.Linear(768, 2)

    def forward(self, x_img):
        # 1. Get CNN features: [B, 64]
        v_feat = self.cnn_feat(x_img) 
        
        # 2. Get LLM hidden state (standard forward)
        dummy_input = torch.zeros((x_img.size(0), 1), dtype=torch.long, device=x_img.device)
        llm_out = self.llm(dummy_input).last_hidden_state[:, -1, :] # [B, 768]
        
        # 3. Apply Gated Fusion
        fused = self.fusion(llm_out, v_feat)
        
        # 4. Action decision
        return self.action_head(fused)

# --- 4. Training Loop ---
def preprocess_state(state):
    grid = np.zeros((3, 3))
    grid[0, 0], grid[1, 1], grid[2, 2] = state[0]/31.0, state[1]/10.0, float(state[2])
    return torch.FloatTensor(grid).view(1, 1, 3, 3)

def train_gated_fusion():
    env = gym.make('Blackjack-v1')
    model = GatedBlackjackVLA()
    
    # Only train the fusion gate and the action head
    optimizer = optim.Adam(list(model.fusion.parameters()) + 
                           list(model.action_head.parameters()), lr=1e-3)
    
    print("Training Gated Fusion...")
    for ep in range(600):
        obs, _ = env.reset()
        done = False
        while not done:
            state_t = preprocess_state(obs)
            logits = model(state_t)
            action = logits.argmax().item()
            
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            with torch.no_grad():
                next_q = model(preprocess_state(next_obs)).max()
                target = reward + 0.95 * next_q * (1 - int(done))
            
            loss = nn.MSELoss()(logits[0, action], target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            obs = next_obs

    # Evaluation
    wins = 0
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = model(preprocess_state(obs)).argmax().item()
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            if reward > 0: wins += 1
    print(f"Gated Fusion Win Rate: {wins}%")

if __name__ == "__main__":
    train_gated_fusion()
