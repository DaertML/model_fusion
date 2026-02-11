import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from transformers import GPT2Model

# --- 1. Re-defining CNN so no imports are needed ---
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
        return self.fc(self.conv(x))

# --- 2. Corrected LoRA Layer for GPT-2 ---
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # GPT-2 layers use Conv1D which is essentially a transposed Linear layer
        # Weights are usually [in_features, out_features]
        in_dim = original_layer.weight.shape[0] 
        out_dim = original_layer.weight.shape[1]
        
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
    def forward(self, x):
        # Original frozen output
        original_out = self.original_layer(x)
        # LoRA path: (x @ A @ B) * scaling
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out

# --- 3. The LoRA-Fused VLA Model ---
class LoRABlackjackVLA(nn.Module):
    def __init__(self, cnn_path="blackjack_cnn.pth"):
        super().__init__()
        
        # A. Load Frozen CNN
        self.cnn_base = BlackjackCNN()
        try:
            self.cnn_base.load_state_dict(torch.load(cnn_path))
            print(f"Loaded CNN weights from {cnn_path}")
        except:
            print("Starting with uninitialized CNN weights.")

        # Strip the last layer (64 -> 2) to get features
        self.cnn_feat = nn.Sequential(
            self.cnn_base.conv,
            nn.Flatten(),
            self.cnn_base.fc[0], # Linear(800, 64)
            self.cnn_base.fc[1]  # ReLU
        )
        for param in self.cnn_feat.parameters():
            param.requires_grad = False

        # B. Load LLM and Inject LoRA
        self.llm = GPT2Model.from_pretrained('gpt2')
        for param in self.llm.parameters():
            param.requires_grad = False

        # Inject LoRA into the Attention layers of the first few blocks
        for i in range(len(self.llm.h)):
            target_layer = self.llm.h[i].attn.c_attn
            self.llm.h[i].attn.c_attn = LoRALayer(target_layer, rank=16)

        # C. Fusion Bridge (Projection)
        self.proj_in = nn.Linear(64, 768)
        self.action_head = nn.Linear(768, 2)

    def forward(self, x_img):
        v_feat = self.cnn_feat(x_img).unsqueeze(1) # [B, 1, 64]
        v_token = self.proj_in(v_feat)             # [B, 1, 768]
        
        # GPT-2 forward
        # Using a dummy token and concatenating visual info
        dummy_input = torch.zeros((x_img.size(0), 1), dtype=torch.long, device=x_img.device)
        hidden = self.llm.wte(dummy_input)
        
        combined = torch.cat([v_token, hidden], dim=1)
        output = self.llm(inputs_embeds=combined).last_hidden_state
        
        return self.action_head(output[:, -1, :])

# --- 4. Helpers & Training ---
def preprocess_state(state):
    grid = np.zeros((3, 3))
    grid[0, 0], grid[1, 1], grid[2, 2] = state[0]/31.0, state[1]/10.0, float(state[2])
    return torch.FloatTensor(grid).view(1, 1, 3, 3)

def train_lora_fusion():
    env = gym.make('Blackjack-v1')
    model = LoRABlackjackVLA()
    
    # Trainable parameters are just the Projections + Action Head + LoRA params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=5e-4)
    
    print("Training LoRA-Fused VLA...")
    for ep in range(500):
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
            
    # Quick Evaluation
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
    print(f"LoRA Fusion Win Rate: {wins}%")

if __name__ == "__main__":
    train_lora_fusion()
