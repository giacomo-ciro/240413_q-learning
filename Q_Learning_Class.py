import numpy as np
import matplotlib.pyplot as plt

class Q_Learning:
    
    def __init__(self, n, perc_bombs, starting_location = None):
        self.n = n
        self.num_bombs = int(perc_bombs*n*n)
        self.r = np.zeros((n,n)) - 1
        self.actions = [(0, 1), (0, -1), (-1, 0), (1, 0), (1,-1), (-1,1), (1,1), (-1,-1)]
        self.q_matrix = np.random.normal(size=(n,n, len(self.actions)))
        self.paths = []
        self.init_environment()

        if starting_location is None:
            starting_location = self.get_starting_location()
        self.starting_location = starting_location
        
        return 

    
    def init_environment(self):
        n, num_bombs = self.n, self.num_bombs
        
        self.r = np.zeros((n,n)) - 1
        
        self.r[0, :] = -100
        self.r[-1, :] = -100
        self.r[:, 0] = -100
        self.r[:, -1] = -100
        
        x_bombs = np.random.choice(np.arange(1,n-1), size=num_bombs, replace=True)
        y_bombs = np.random.choice(np.arange(1, n-1), size=num_bombs, replace=True)
        self.r[x_bombs, y_bombs] = -100
        self.x_bombs, self.y_bombs = x_bombs, y_bombs

        x_target, y_target = (np.random.randint(1, n-1), np.random.randint(1, n-1))
        self.r[x_target, y_target] = 100
        self.x_target, self.y_target = x_target, y_target

        return
    
    def __repr__(self):
        return 'Q_Learning'
    
    def is_terminal_state(self, x, y):
        if self.r[x,y] == -1:
            return False
        else:
            return True

    def get_starting_location(self):
        x,y = np.random.randint(self.n, size=2)
        
        while self.is_terminal_state(x,y):
            x,y = np.random.randint(self.n, size=2)
        
        return x,y 

    def get_next_action(self, x, y, epsilon=0.9):
        if np.random.random() < epsilon:
            action = np.argmax(self.q_matrix[x, y])
        else:
            action = np.random.randint(len(self.actions))
        return action
    
    def get_next_state(self, x, y, action):
        return x + self.actions[action][0], y + self.actions[action][1]
    
    def train(self, num_episodes, lr=0.9, gamma=0.95, random_starting_location = False):
            
        for _ in range(num_episodes):
            
            if random_starting_location:
                x, y = self.get_starting_location()
            else:
                x, y = self.starting_location
            
            x_path = [x]
            y_path = [y]

            while not self.is_terminal_state(x, y):

                action = self.get_next_action(x, y)

                old_x, old_y = x, y
                x, y = self.get_next_state(x, y, action)

                x_path.append(x)
                y_path.append(y)

                reward = self.r[x, y]
                old_q_value = self.q_matrix[old_x, old_y, action]
                temporal_difference = reward + (gamma * np.max(self.q_matrix[x, y])) - old_q_value

                new_q_value = old_q_value + (lr * temporal_difference)
                self.q_matrix[old_x, old_y, action] = new_q_value
            
            self.paths.append([x_path, y_path])

        return
    
    def get_optimal_path(self, starting_location=None, epsilon = 0.9):

        if starting_location is None:
            x, y = self.starting_location
        else:
            x, y = starting_location
        
        x_path, y_path = [x], [y]

        while not self.is_terminal_state(x, y):

                action = self.get_next_action(x, y, epsilon=epsilon)

                old_x, old_y = x, y
                x, y = self.get_next_state(x, y, action)

                x_path.append(x)
                y_path.append(y)

        return [x_path, y_path]

    
    def plot(self, path = None):

        x, y = self.starting_location
        
        n = self.n

        fig, ax = plt.subplots()

        ax.axis([0, n-1, 0, n-1])
        ax.scatter(self.x_bombs, self.y_bombs, c='k', label='Bombs')
        ax.scatter(self.x_target, self.y_target, c='red', label='Target')
        ax.scatter(x, y, c='blue', label='Start')

        if path is not None:
            x_path, y_path = path
            ax.plot(x_path, y_path)
        
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.grid()
        ax.legend()

        return fig


                    
