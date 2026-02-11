import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from transformers import GPT2Model

# --- 1. The Architecture ---

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

class CrossAttentionBridge(nn.Module):
    """
    Standard Multi-Head Attention where:
    Query = LLM Hidden States
    Key/Value = CNN Features
    """
    def __init__(self, embed_dim=768, feat_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(feat_dim, embed_dim)
        self.v_proj = nn.Linear(feat_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_llm, x_cnn):
        # x_llm: [Batch, Seq, 768]
        # x_cnn: [Batch, 1, 64]
        q = self.q_proj(x_llm)
        k = self.k_proj(x_cnn)
        v = self.v_proj(x_cnn)
        
        attn_out, _ = self.attn(q, k, v)
        return self.norm(x_llm + attn_out) # Residual connection

class CrossAttentionVLA(nn.Module):
    def __init__(self, cnn_path="blackjack_cnn.pth"):
        super().__init__()
        # Load and Freeze CNN
        self.cnn_base = BlackjackCNN()
        try:
            self.cnn_base.load_state_dict(torch.load(cnn_path))
        except: pass 
        
        # Strip final layer to get features
        self.cnn_features = nn.Sequential(*list(self.cnn_base.conv.children()), 
                                         nn.Flatten(), self.cnn_base.fc[1])
        for param in self.cnn_features.parameters(): param.requires_grad = False

        # Load and Freeze LLM
        full_gpt = GPT2Model.from_pretrained('gpt2')
        self.embed = full_gpt.wte
        self.llm_part1 = full_gpt.h[:6]
        self.llm_part2 = full_gpt.h[6:]
        self.ln_f = full_gpt.ln_f
        for param in full_gpt.parameters(): param.requires_grad = False

        # THE GLUE: Cross-Attention + Action Head
        self.bridge = CrossAttentionBridge(768, 64)
        self.action_head = nn.Linear(768, 2)

    def forward(self, x_img):
        # 1. Extract CNN features
        v_feat = self.cnn_features(x_img).unsqueeze(1) # [B, 1, 64]
        
        # 2. LLM Part 1 (Internal Reasoning)
        hidden = self.embed(torch.zeros((x_img.size(0), 1), dtype=torch.long))
        for block in self.llm_part1:
            hidden = block(hidden)[0]
            
        # 3. CROSS ATTENTION: LLM "looks" at the CNN features
        fused = self.bridge(hidden, v_feat)
        
        # 4. LLM Part 2 (Post-Visual Reasoning)
        for block in self.llm_part2:
            fused = block(fused)[0]
            
        return self.action_head(self.ln_f(fused)[:, -1, :])

# --- 2. Training Logic ---

def preprocess_state(state):
    grid = np.zeros((3, 3))
    grid[0, 0], grid[1, 1], grid[2, 2] = state[0]/31.0, state[1]/10.0, float(state[2])
    return torch.FloatTensor(grid).view(1, 1, 3, 3)

def train_and_eval():
    env = gym.make('Blackjack-v1')
    model = CrossAttentionVLA()
    # Only train the Bridge and the Head
    optimizer = optim.Adam(list(model.bridge.parameters()) + 
                           list(model.action_head.parameters()), lr=5e-4)
    
    print("Training Cross-Attention Glue...")
    for ep in range(1000): # Increased episodes for attention convergence
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

    # Final Eval
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
    print(f"Final Win Rate with Cross-Attention: {wins}%")

if __name__ == "__main__":
    train_and_eval()
