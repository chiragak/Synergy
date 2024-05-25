import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Environment Simulation
def get_environment_state():
    longitude = np.random.uniform(-180, 180)
    latitude = np.random.uniform(-90, 90)
    altitude = np.random.uniform(0, 40000)  # in feet
    weather_conditions = np.random.choice([0, 1, 2])  # 0: clear, 1: rainy, 2: stormy
    windspeed = np.random.uniform(0, 100)  # km/h
    visibility = np.random.uniform(0, 10)  # km
    dest_longitude = np.random.uniform(-180, 180)
    dest_latitude = np.random.uniform(-90, 90)
    current_direction = np.random.uniform(0, 360)
    return np.array([longitude, latitude, altitude, weather_conditions, windspeed, visibility, dest_longitude, dest_latitude, current_direction])

def calculate_reward(state, next_state):
    distance_to_destination = np.linalg.norm(next_state[:2])  # Example: Distance to destination
    weather_penalty = state[3]  # Higher penalty for worse weather conditions
    wind_penalty = state[4] / 10  # Normalized windspeed penalty
    visibility_penalty = (10 - state[5]) / 10  # Normalized visibility penalty
    reward = - (distance_to_destination + weather_penalty + wind_penalty + visibility_penalty)
    return reward

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[action] = target
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Training the agent
state_size = 9  # Updated state size
action_size = 4  # Example action size: change in direction (0: turn left, 1: turn right, 2: ascend, 3: descend)
agent = DQNAgent(state_size, action_size)
episodes = 1000
batch_size = 32

for e in range(episodes):
    state = get_environment_state()  # Initialize with random state
    for time in range(500):
        action = agent.act(state)
        next_state = get_environment_state()  # Simulate next state
        reward = calculate_reward(state, next_state)  # Calculate reward
        done = time == 499  # Simulate end of episode
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Save the model at regular intervals
    if e % 100 == 0:
        agent.save_model(f"dqn_model_{e}.pth")

# Save the final model
agent.save_model("dqn_model_final.pth")
