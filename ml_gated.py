import torch
import torch.nn as nn
import gymnasium as gym
from transformers import GPT2Model

class GLAFFusionAgent(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=768):
        super().__init__()
        self.llm = GPT2Model.from_pretrained("gpt2")
        self.projector = nn.Linear(state_dim, hidden_dim)
        
        # The Gate: learns a value between 0 and 1
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        self.action_head = nn.Linear(hidden_dim, 2)

    def forward(self, state):
        # 1. Get LLM internal "logic" (dummy input to get baseline hidden state)
        dummy_input = torch.tensor([[50256]]) # <|endoftext|>
        llm_h = self.llm(dummy_input).last_hidden_state[:, 0, :]
        
        # 2. Project Blackjack State
        state_tensor = torch.tensor(state, dtype=torch.float32).view(-1, 3)
        state_h = self.projector(state_tensor)
        
        # 3. GLAF Gating
        combined = torch.cat([llm_h, state_h], dim=-1)
        g = self.gate(combined)
        
        # Formula: (1-g)*LLM + g*State
        fused = (1 - g) * llm_h + g * state_h
        return self.action_head(fused)

# --- Evaluation Script ---
env = gym.make("Blackjack-v1")
model = GLAFFusionAgent()
wins, trials = 0, 1000

for _ in range(trials):
    obs, _ = env.reset()
    done = False
    while not done:
        logits = model(obs)
        action = torch.argmax(logits).item()
        obs, reward, _, _, _ = env.step(action)
        done = _ or reward != 0 # Basic termination
    if reward > 0: wins += 1

print(f"GLAF Fusion Win Rate: {(wins/trials)*100:.2f}%")
