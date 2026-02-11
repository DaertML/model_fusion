import gymnasium as gym
import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor

# 1. Setup Environment and Models
env = gym.make("Blackjack-v1")

# We use two regressors to approximate the Q-function for each action
# warm_start=True allows us to update the model incrementally
models = [
    SGDRegressor(learning_rate='constant', eta0=0.01, warm_start=True),
    SGDRegressor(learning_rate='constant', eta0=0.01, warm_start=True)
]

# Initialize models with a dummy step so they can predict
for m in models:
    m.partial_fit([[0, 0, 0]], [0])

def get_features(obs):
    # Blackjack obs is (player_sum, dealer_card, usable_ace)
    # We convert the boolean usable_ace to 1 or 0
    return np.array([[obs[0], obs[1], int(obs[2])]])

# 2. Training Loop (Online Learning)
episodes = 5000
epsilon = 0.1  # Exploration rate

for i in range(episodes):
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Epsilon-Greedy Action Selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # Predict values for both actions and pick the best
            q_values = [m.predict(get_features(obs))[0] for m in models]
            action = np.argmax(q_values)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update the model for the action we just took (Standard Q-Learning Update)
        # Target = immediate reward + discounted future value
        if not done:
            future_q = max([m.predict(get_features(next_obs))[0] for m in models])
            target = reward + 0.95 * future_q
        else:
            target = reward
            
        models[action].partial_fit(get_features(obs), [target])
        obs = next_obs

# 3. Save the dual-model system
with open('blackjack_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("Trained on 5000 hands and saved blackjack_models.pkl")
