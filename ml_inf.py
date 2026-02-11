import gymnasium as gym
import pickle
import numpy as np

# 1. Load the trained models
with open('blackjack_models.pkl', 'rb') as f:
    models = pickle.load(f)

env = gym.make("Blackjack-v1")
num_games = 10000

# Counters
results = {"win": 0, "loss": 0, "draw": 0}

print(f"Running {num_games} games of Blackjack...")

for _ in range(num_games):
    obs, _ = env.reset()
    done = False
    
    while not done:
        # State: (player_sum, dealer_card, usable_ace)
        features = np.array([[obs[0], obs[1], int(obs[2])]])
        
        # Action: Predict expected reward for Stay (0) and Hit (1)
        # We pick the action with the higher predicted value
        q_values = [m.predict(features)[0] for m in models]
        action = np.argmax(q_values)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Gymnasium reward: +1 = Win, -1 = Loss, 0 = Draw
    if reward > 0:
        results["win"] += 1
    elif reward < 0:
        results["loss"] += 1
    else:
        results["draw"] += 1

# 2. Calculate Statistics
win_pct = (results["win"] / num_games) * 100
loss_pct = (results["loss"] / num_games) * 100
draw_pct = (results["draw"] / num_games) * 100

print("-" * 30)
print(f"RESULTS OVER {num_games} GAMES")
print("-" * 30)
print(f"Wins:   {results['win']} ({win_pct:.2f}%)")
print(f"Losses: {results['loss']} ({loss_pct:.2f}%)")
print(f"Draws:  {results['draw']} ({draw_pct:.2f}%)")
print("-" * 30)

env.close()
