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

# --- 2. The Soft Prompt VLA ---
class SoftPromptBlackjackVLA(nn.Module):
    def __init__(self, cnn_path="blackjack_cnn.pth", prompt_len=5):
        super().__init__()
        self.prompt_len = prompt_len
        
        # A. Load Frozen CNN
        self.cnn_base = BlackjackCNN()
        try:
            self.cnn_base.load_state_dict(torch.load(cnn_path))
        except: pass
        
        self.cnn_feat = nn.Sequential(
            self.cnn_base.conv, nn.Flatten(), 
            self.cnn_base.fc[0], self.cnn_base.fc[1]
        )
        for param in self.cnn_feat.parameters(): param.requires_grad = False

        # B. Load Frozen LLM
        self.llm = GPT2Model.from_pretrained('gpt2')
        for param in self.llm.parameters(): param.requires_grad = False

        # C. Prompt Encoder: Maps 1 CNN vector to N LLM tokens
        # We use a bottleneck structure to make the mapping more expressive
        self.prompt_encoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, prompt_len * 768)
        )
        
        self.action_head = nn.Linear(768, 2)

    def forward(self, x_img):
        batch_size = x_img.size(0)
        
        # 1. Get CNN features
        v_feat = self.cnn_feat(x_img) # [B, 64]
        
        # 2. Project to Soft Prompt Tokens
        # Shape: [B, prompt_len * 768] -> [B, prompt_len, 768]
        soft_prompts = self.prompt_encoder(v_feat).view(batch_size, self.prompt_len, 768)
        
        # 3. Get LLM standard embeddings for a "Start" token
        dummy_input = torch.zeros((batch_size, 1), dtype=torch.long, device=x_img.device)
        start_embedding = self.llm.wte(dummy_input) # [B, 1, 768]
        
        # 4. Concatenate: [Soft Prompts... Start Token]
        # Total sequence length: prompt_len + 1
        full_embeddings = torch.cat([soft_prompts, start_embedding], dim=1)
        
        # 5. LLM Forward
        output = self.llm(inputs_embeds=full_embeddings).last_hidden_state
        
        # 6. Action decision from the last token (the Start token's position)
        return self.action_head(output[:, -1, :])

# --- 3. Training logic ---
def preprocess_state(state):
    grid = np.zeros((3, 3))
    grid[0, 0], grid[1, 1], grid[2, 2] = state[0]/31.0, state[1]/10.0, float(state[2])
    return torch.FloatTensor(grid).view(1, 1, 3, 3)

def train_soft_prompt():
    env = gym.make('Blackjack-v1')
    model = SoftPromptBlackjackVLA(prompt_len=4) # Using 4 virtual tokens
    
    # Train only the Prompt Encoder and Action Head
    optimizer = optim.Adam(list(model.prompt_encoder.parameters()) + 
                           list(model.action_head.parameters()), lr=1e-3)
    
    print("Training Soft Prompt Injection...")
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
    print(f"Soft Prompt Win Rate: {wins}%")

if __name__ == "__main__":
    train_soft_prompt()
