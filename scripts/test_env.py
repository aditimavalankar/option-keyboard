import gym
import envs
import time
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

KEY_MAP = ['UP', 'RIGHT', 'DOWN', 'LEFT']


def main():
    env = gym.make('ForagingWorld-v0', scenario=1)
    env.set_learning_options()
    s = env.reset()
    env.render()
    ep_reward = 0
    rewards = []

    for _ in range(300000):
        a = np.random.randint(0, 4)
        s, r, done, _ = env.step(a)
        ep_reward += r

        if done:
            print('Episode terminated!')
            print('Total reward:', ep_reward)
            rewards.append(ep_reward)

            s = env.reset()
            ep_reward = 0
            time.sleep(1)

        env.render()
        time.sleep(0.1)
    print(np.array(rewards).mean(), np.array(rewards).std())
    return


if __name__ == '__main__':
    main()
