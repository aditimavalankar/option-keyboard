from gym.envs.registration import register

register(
    id='ForagingWorld-v0',
    entry_point='option_keyboard.envs.foraging_world:ForagingWorldEnv',
)
