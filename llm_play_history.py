import ollama
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
import re
from collections import deque

# Initialize environment
env = gym.make('Blackjack-v1', render_mode="rgb_array", natural=True)

# --- Stats Tracking ---
stats = {"Wins": 0, "Losses": 0, "Draws": 0, "Total Games": 0}

def extract_action_from_latex(model_output):
    match = re.search(r"\\boxed{(\d+)}", model_output)
    if match:
        return int(match.group(1))
    return None

# Setup Plotting
plt.ion()
fig, ax = plt.subplots(figsize=(6, 5))
observation, info = env.reset()
img_plot = ax.imshow(env.render())
plt.axis('off')
plt.show()

try:
    while True:  # Continuous play loop
        stats["Total Games"] += 1
        observation, info = env.reset()
        terminated = False
        truncated = False
        history_buffer = []
        step = 0
        
        print(f"\n===== Starting Game #{stats['Total Games']} =====")

        while not terminated and not truncated:
            step += 1
            
            # Update UI
            img_plot.set_data(env.render())
            ax.set_title(f"Game {stats['Total Games']} | Player: {observation[0]} | Dealer: {observation[1]}")
            plt.draw()
            plt.pause(0.1)

            # Prepare Prompt
            prompt_content = f"""
Current Blackjack State:
- Player Sum: {observation[0]}
- Dealer Showing: {observation[1]}
- Usable Ace: {bool(observation[2])}

History of this hand: {history_buffer}

Goal: Beat the dealer by getting closer to 21 without busting.
Actions:
0: Stick (Stop)
1: Hit (Take card)

Think step-by-step, then provide the action in boxed LaTeX: $\\boxed{{0}}$ or $\\boxed{{1}}$.
"""

            # Get LLM Action
            response = ollama.chat(
                model='qwen3:8b',
                messages=[{'role': 'user', 'content': prompt_content}],
            )
            action_text = response['message']['content'].strip()
            action = extract_action_from_latex(action_text)
            
            if action not in [0, 1]:
                action = 0  # Default to Stick

            history_buffer.append(f"Step {step}: {action}")
            
            # Take Step
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                # Update Stats
                if reward > 0:
                    stats["Wins"] += 1
                    result_text = "WIN"
                elif reward < 0:
                    stats["Losses"] += 1
                    result_text = "LOSE"
                else:
                    stats["Draws"] += 1
                    result_text = "DRAW"

                print(f"Result: {result_text} | Reward: {reward}")
                print(f"Current Record: {stats['Wins']}W - {stats['Losses']}L - {stats['Draws']}D")
                
                # Visual pause at the end of a hand
                img_plot.set_data(env.render())
                plt.draw()
                plt.pause(1.5) 

except KeyboardInterrupt:
    print("\nStopping session...")
finally:
    env.close()
    plt.ioff()
    print(f"Final Stats: {stats}")