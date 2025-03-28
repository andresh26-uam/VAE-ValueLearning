from gymnasium.envs.registration import register

register(
     id="FireFighters-v0",
     entry_point="envs.firefighters_env:FireFightersEnv",
     max_episode_steps=1000,
)
register(
     id="FireFightersEnvWithObservation-v0",
     entry_point="envs.firefighters_env:FireFightersEnvWithObservation",
     max_episode_steps=1000,
)

register(
     id="FixedDestRoadWorld-v0",
     entry_point="envs.roadworld_env:FixedDestRoadWorldGymPOMDP",
     max_episode_steps=1000,
)

register(
     id="VariableDestRoadWorld-v0",
     entry_point="envs.roadworld_env:VariableDestRoadWorldGymPOMDP",
     max_episode_steps=1000,
)

register(
     id="RouteChoiceEnvironmentApollo-v0",
     entry_point="envs.routechoiceApollo:RouteChoiceEnvironmentApollo",
     max_episode_steps=1,
)