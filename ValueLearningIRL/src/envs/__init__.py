from gymnasium.envs.registration import register

register(
     id="FireFighters-v0",
     entry_point="src.envs.firefighters_env:FireFightersEnv",
     max_episode_steps=1000,
)
