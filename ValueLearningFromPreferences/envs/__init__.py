from gymnasium.envs.registration import register

register(
     id="FireFighters-v0",
     entry_point="envs.firefighters_env:FireFightersEnv",
     max_episode_steps=1000,
)
register(
     id="FireFightersEnvWithObservation-v0",
     entry_point="envs.firefighters_env:FireFightersEnvWithObservation",
)

register(
     id="FixedDestRoadWorld-v0",
     entry_point="envs.roadworld_env:FixedDestRoadWorldGymPOMDP",
)

register(
     id="VariableDestRoadWorld-v0",
     entry_point="envs.roadworld_env:VariableDestRoadWorldGymPOMDP",
)

register(
     id="RouteChoiceEnvironmentApollo-v0",
     entry_point="envs.routechoiceApollo:RouteChoiceEnvironmentApollo",
     max_episode_steps=1,
)

register(
     id="RouteChoiceEnvironmentApolloComfort-v0",
     entry_point="envs.routechoiceApollo:RouteChoiceEnvironmentApolloComfort",
     max_episode_steps=1,
)

register(
     id="MultiValuedCarEnv-v0",
     entry_point="envs.multivalued_car_env:MultiValuedCarEnv",
)

register(
    id='FireFightersMO-v0',
    entry_point='envs.firefighters_env_mo:FireFightersEnvMO',
)