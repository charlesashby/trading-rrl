import pickle
import matplotlib.pyplot as plt
import numpy as np

data = []
for i in range(9):
    with open('data/data_stop_loss_{}.pickle'.format(i), 'rb') as f:
        data.append(pickle.load(f))


""" Plotting stop losses hyper-parameter search. 
    Data stored is all_W, all_prices, all_F, w
"""

fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
ax[0].plot(data[0][1])
ax[0].set_xlabel("time")
ax[0].set_ylabel("USDJPY")
ax[0].grid(True)

ax[1].plot(data[0][2])
ax[1].set_xlabel("time")
ax[1].set_ylabel("F")
ax[1].grid(True)

for i, d in enumerate(data):
    # Plot positions


    # Plot cumulative reward
    ax[2].plot(d[0], label='Stop Loss: {}'.format(str(0.1 * (i + 1))[:3]))
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("Sum of reward[yen]")
    ax[2].grid(True)

plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("img/stop-loss-hp-search.png", dpi=300)
plt.close()


# Average reward and variance of total cumulative reward
averages = [np.mean(d[0]) for d in data]
variances = [np.std(d[0]) for d in data]
plt.plot([(averages[i], variances[i]) for i in range(len(averages))])

data = []
for i in range(6):
    with open('data/data_learning_rate_{}.pickle'.format(i), 'rb') as f:
        data.append(pickle.load(f))


learning_rates = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
ax[0].plot(data[0][1])
ax[0].set_xlabel("time")
ax[0].set_ylabel("USDJPY")
ax[0].grid(True)

for i, d in enumerate(data):
    # Plot positions
    ax[1].plot(d[2], label='Learning Rate: {}'.format(learning_rates[i]))
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("F")
    ax[1].grid(True)

    # Plot cumulative reward
    ax[2].plot(d[0], label='Learning Rate: {}'.format(learning_rates[i]))
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("Sum of reward[yen]")
    ax[2].grid(True)



# All ticks graphs
with open('data/data_all_ticks.pickle', 'rb') as f:
    data_all_ticks = pickle.load(f)

with open('data/data_all_ticks_no_train.pickle', 'rb') as f:
    data_all_ticks_no_train = pickle.load(f)

with open('data/data_all_ticks_long.pickle', 'rb') as f:
    data_all_ticks_long = pickle.load(f)

fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
ax[0].plot(data_all_ticks[1])
ax[0].set_xlabel("time")
ax[0].set_ylabel("USD/JPY")
ax[0].grid(True)

ax[1].plot(data_all_ticks_no_train[2], color='orange', label="With random weights")
ax[1].plot(data_all_ticks[2], color='blue', label="With optimized weights")
ax[1].plot(data_all_ticks_long[2], color='green', label="Long position")

ax[1].set_xlabel("time")
ax[1].set_ylabel("F")
ax[1].legend(loc="lower right")
ax[1].grid(True)

ax[2].plot(data_all_ticks[0], label="With optimized weights")
ax[2].plot(data_all_ticks_no_train[0], label="With random weights")
ax[2].plot(data_all_ticks_long[0], label="Long position")

ax[2].set_xlabel("time")
ax[2].set_ylabel("Sum of reward (Yen)")
ax[2].legend(loc="upper left")
ax[2].grid(True)
plt.tight_layout()

plt.savefig("img/comparison_all_ticks.png", dpi=300)
plt.close()
