import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from transformers import GPT2Model, GPT2Config
from peft import LoraConfig, get_peft_model

class LoRAFusionAgent(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=768):
        super().__init__()
        # 1. Load Base LLM
        self.config = GPT2Config.from_pretrained("gpt2")
        self.llm = GPT2Model.from_pretrained("gpt2")
        
        # 2. Add LoRA Adapters to Attention
        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["c_attn"], 
            lora_dropout=0.1, fan_in_fan_out=True
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        # 3. State Projector
        self.projector = nn.Linear(state_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 2) # Output: Stay or Hit

    def forward(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).view(-1, 3)
        state_embed = self.projector(state_tensor).unsqueeze(1) # [B, 1, 768]
        
        # Pass merged knowledge through Adapted LLM
        outputs = self.llm(inputs_embeds=state_embed)
        last_hidden = outputs.last_hidden_state[:, 0, :]
        return self.action_head(last_hidden)

# --- Evaluation Script ---
env = gym.make("Blackjack-v1")
model = LoRAFusionAgent()
model.eval()

wins, trials = 0, 1000
for _ in range(trials):
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            logits = model(obs)
            action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    if reward > 0: wins += 1

print(f"LoRA Fusion Win Rate: {(wins/trials)*100:.2f}%")
