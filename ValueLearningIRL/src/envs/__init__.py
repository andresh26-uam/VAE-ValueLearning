from gymnasium.envs.registration import register

register(
     id="FireFighters-v0",
     entry_point="src.envs.firefighters_env:FireFightersEnv",
     max_episode_steps=1000,
)

register(
     id="FixedDestRoadWorld-v0",
     entry_point="src.envs.roadworld_env:FixedDestRoadWorldGymPOMDP",
     max_episode_steps=1000,
)
