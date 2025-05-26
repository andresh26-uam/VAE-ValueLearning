from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'arial',
        'size'   : 28}

matplotlib.rc('font', **font)


points = np.load('graphics.npy')
x_0 = points[:,0]
x_E = points[:,1]

mean_scores0 = list()
scores0 = deque(maxlen=100)
for point in x_0:
    scores0.append(point)
    mean_score = [np.mean(scores0)]
    mean_scores0.append(mean_score)

mean_scoresE = list()
scoresE = deque(maxlen=100)
for point in x_E:
    scoresE.append(point)
    mean_score = [np.mean(scoresE)]
    mean_scoresE.append(mean_score)


fig, ax = plt.subplots()
plt.axhline(y=0.59)
plt.axhline(y=0.24)
plt.plot(range(5000), mean_scores0[:5000], markersize=10, marker='s', markevery=250, label="Individual objective $V_0$", color='red') # plotting t, a separately
plt.plot(range(5000), mean_scoresE[:5000], markersize=10, marker='^', markevery=250, label="Ethical objective $V_N + V_E$", color='green')


ax.set(xlabel='Episode', ylabel='Discounted sum of rewards')
plt.legend(loc='best')
plt.show()