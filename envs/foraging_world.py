import numpy as np
import gym
from gym import spaces, utils
import cv2
import pdb

# Action keywords
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Change this if number of food items/resources changes!
FOOD_TYPES = [(1, 0), (0, 1), (1, 1)]

MAX_STEPS = 300


class ForagingWorldEnv(gym.Env, utils.EzPickle):
    def __init__(self,
                 grid_length=12,
                 m=2,
                 n=3,
                 decrease=[0.05, 0.05],
                 scenario=1,
                 initial_food_items=[2, 2, 2]):
        """
            grid_length: Length of the square grid
            m: Number of resources/nutrients
            n: Number of food items
            decrease: Decrease in proportion of resources at each time step
            scenario: Desirability function for resources (details in paper)
            initial_food_items: Number of food items in the initial state
            spawn_prob: Probability of spawning a new food item of each type
                        at every time step
        """
        self.grid = np.zeros((grid_length, grid_length, n))
        self.resources = np.zeros(m)
        self.cumulants = np.zeros(m)
        self.m = m
        self.n = n
        self.decrease = np.array(decrease)

        obs_len = grid_length * grid_length * n + m
        self.observation_space = spaces.Box(low=-np.inf, high=1,
                                            shape=(obs_len,))
        self.action_space = spaces.Discrete(4)

        self.n_steps = 0
        self.learning_options = False
        self.scenario = scenario
        self.initial_food_items = initial_food_items

    def reset(self):
        grid_length, _, _ = self.grid.shape
        self.grid = np.zeros((grid_length, grid_length, self.n))

        # Populate the grid.
        for index, n_items in enumerate(self.initial_food_items):
            for _ in range(n_items):
                while True:
                    x, y = np.random.randint(grid_length, size=(2))
                    if (self.grid[x, y].sum() == 0 and
                            (x != int(grid_length / 2) or
                             y != int(grid_length / 2))):
                        break
                self.grid[x, y, index] = 1

        self.resources = np.zeros(self.m)
        self.cumulants = np.zeros(self.m)
        self.n_steps = 0

        return self.get_observation()

    def step(self, a):
        self.n_steps += 1
        self.update_grid(a)

        grid_length = self.grid.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))
        food_type = (0, 0)

        if self.grid[centre].sum() != 0:
            food_type = FOOD_TYPES[np.flatnonzero(self.grid[centre])[0]]
            self.resources += food_type
            self.spawn_new_item(np.flatnonzero(self.grid[centre])[0])
            self.grid[centre] = np.zeros(self.n)

        reward = np.sum(food_type * self.desirability())
        done = False if self.n_steps < MAX_STEPS else True
        info = {'grid': self.grid, 'resources': self.resources,
                'food type': food_type,
                'rewards': food_type * self.desirability()}
        self.resources -= self.decrease

        return self.get_observation(), reward, done, info

    def get_observation(self):
        return np.concatenate((self.grid, self.resources), axis=None)

    def render(self):
        grid_length = self.grid.shape[0]
        centre = (int(grid_length / 2), int(grid_length / 2))
        self.grid[centre] = np.ones(self.n)
        scale = max(int(400 / self.grid.shape[1]), 1)
        modified_size = self.grid.shape[1] * scale
        img = cv2.resize(self.grid, (modified_size, modified_size),
                         interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                 value=[0.375, 0.375, 0.375])
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
        cv2.imshow('ForagingWorld', img)
        cv2.waitKey(100)
        self.grid[centre] = np.zeros(self.n)

    def close(self):
        cv2.destroyAllWindows()

    def desirability(self):
        if self.learning_options:
            # Scenario 0
            d1 = self.w[0]
            d2 = self.w[1]
        elif self.scenario == 1:
            d1 = 1 if self.resources[0] <= 10 else -1
            d2 = -1 if self.resources[1] <= 5 else (-1
                                                    if self.resources[1] >= 25
                                                    else 5)
        elif self.scenario == 2:
            d1 = 1 if self.resources[0] <= 10 else -1
            d2 = -1 if self.resources[1] <= 5 else (-1
                                                    if self.resources[1] >= 15
                                                    else 5)
        return np.array([d1, d2])

    def update_grid(self, a):
        """
            The grid follows toroidal dynamics i.e. it wraps around, connecting
            cells on opposite edges.
        """
        if a == UP:
            self.grid = np.concatenate((self.grid[-1:, :, :],
                                        self.grid[:-1, :, :]),
                                       axis=0)

        elif a == DOWN:
            self.grid = np.concatenate((self.grid[1:, :, :],
                                        self.grid[:1, :, :]),
                                       axis=0)

        elif a == RIGHT:
            self.grid = np.concatenate((self.grid[:, 1:, :],
                                        self.grid[:, :1, :]),
                                       axis=1)

        elif a == LEFT:
            self.grid = np.concatenate((self.grid[:, -1:, :],
                                        self.grid[:, :-1, :]),
                                       axis=1)

    def num_resources(self):
        return self.m

    def set_learning_options(self, w=[1, 1], flag=True):
        self.w = w
        self.learning_options = flag

    def spawn_new_item(self, food_type_index):
        """
            Each time a food item is consumed, a replacement item is spawned
            at a random position in the grid.
        """
        grid_length = self.grid.shape[0]
        while True:
            x, y = np.random.randint(grid_length, size=(2))
            if (self.grid[x, y].sum() == 0 and (x != int(grid_length / 2) or
                                                y != int(grid_length / 2))):
                break
        self.grid[x, y, food_type_index] = 1
