import numpy as np

# Define the gridworld environment
class Gridworld:
    def __init__(self, size=4, start=(0, 0), goal=(3, 3)):
        self.size = size
        self.start = start
        self.goal = goal
        self.reset()

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        x, y = self.position
        if action == 0: # up
            x = max(x - 1, 0)
        elif action == 1: # right
            y = min(y + 1, self.size - 1)
        elif action == 2: # down
            x = min(x + 1, self.size - 1)
        elif action == 3: # left
            y = max(y - 1, 0)
        self.position = (x, y)
        reward = 0 if self.position != self.goal else 1
        done = self.position == self.goal
        return self.position, reward, done

# Define the TD learning agent
class TDAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.V = np.zeros((env.size, env.size))

    def choose_action(self):
        return np.random.choice(4)

    def update(self, state, reward, next_state):
        x, y = state
        nx, ny = next_state
        td_target = reward + self.gamma * self.V[nx][ny]
        td_error = td_target - self.V[x][y]
        self.V[x][y] += self.alpha * td_error

# Run the TD learning agent on the gridworld environment
env = Gridworld()
agent = TDAgent(env)

for episode in range(10):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action()
        next_state, reward, done = env.step(action)
        agent.update(state, reward, next_state)
        state = next_state

print(agent.V)
