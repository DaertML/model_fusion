import torch
import torch.nn as nn
import gymnasium as gym
from transformers import GPT2Model

class SoftPromptAgent(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=768, n_tokens=5):
        super().__init__()
        self.llm = GPT2Model.from_pretrained("gpt2")
        self.n_tokens = n_tokens
        
        # Maps 1 state to 5 virtual tokens
        self.prompt_gen = nn.Linear(state_dim, n_tokens * hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 2)

    def forward(self, state):
        batch_size = 1
        state_tensor = torch.tensor(state, dtype=torch.float32).view(batch_size, -1)
        
        # Generate Soft Prompts [B, 5, 768]
        prompts = self.prompt_gen(state_tensor).view(batch_size, self.n_tokens, 768)
        
        # Pass Prompts as the only input to the LLM
        outputs = self.llm(inputs_embeds=prompts)
        
        # Use the mean of the soft prompt hidden states for action
        final_h = outputs.last_hidden_state.mean(dim=1)
        return self.action_head(final_h)

# --- Evaluation Script ---
env = gym.make("Blackjack-v1")
model = SoftPromptAgent()
wins, trials = 0, 1000

for _ in range(trials):
    obs, _ = env.reset()
    done = False
    while not done:
        logits = model(obs)
        action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    if reward > 0: wins += 1

print(f"Soft Prompt Win Rate: {(wins/trials)*100:.2f}%")
