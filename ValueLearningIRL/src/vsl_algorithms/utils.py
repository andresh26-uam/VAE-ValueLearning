
import numpy as np
from scipy.stats import entropy
def JSD(P, Q):
    n = len(P)
    avg_jsd = 0.0
    for p,q in zip(P,Q):
        dist_p = np.array([p,1.0-p])
        dist_q = np.array([q,1.0-q])
        M = 0.5 * (dist_p + dist_q)
        avg_jsd += ((0.5 * (entropy(dist_p, M) + entropy(dist_q,M)))/n)
    return avg_jsd
