import numpy as np
from ADS_Environment import Environment
from VI import value_iteration

SPEED = 0
INTERNAL_SAFETY = 1
EXTERNAL_SAFETY = 2
lex_orders = [[SPEED, INTERNAL_SAFETY, EXTERNAL_SAFETY],
              [SPEED, EXTERNAL_SAFETY, INTERNAL_SAFETY],
              [INTERNAL_SAFETY, SPEED, EXTERNAL_SAFETY],
              [INTERNAL_SAFETY, EXTERNAL_SAFETY, SPEED],
              [EXTERNAL_SAFETY, INTERNAL_SAFETY, SPEED],
              [EXTERNAL_SAFETY, SPEED, INTERNAL_SAFETY]
              ]
for n_obstacles in range(20):
    env = Environment(obstacles=n_obstacles)
    weights = [0.0, 0.0, 0.0]

    print("=================", n_obstacles, "=================")
    for lex_ord in lex_orders:
        policy, v, q = value_iteration(env, weights=weights, lex=lex_ord, discount_factor=1.0, model_used=None)
        np.save("Policies_Celeste/Unpolicy_" + str(lex_ord) + "_v" + str(n_obstacles) + ".npy", policy)
