from gym.envs.registration import register

register(
    id='ForagingWorld-v0',
    entry_point='envs.foraging_world:ForagingWorldEnv',
)
