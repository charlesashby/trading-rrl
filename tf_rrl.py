from utils.dataloader import DataLoader
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def tanh(input_, out_dim, scope=None, reuse=False):

    with tf.variable_scope(scope or 'tanh', reuse=reuse):
        # W = tf.get_variable('W', [input_.get_shape()[1], out_dim], constraint=lambda x: tf.clip_by_value(x, -5, 5))
        # b = tf.get_variable('b', [out_dim], constraint=lambda x: tf.clip_by_value(x, -5, 5))
        W = tf.get_variable('W', [input_.get_shape()[1], out_dim])
        b = tf.get_variable('b', [out_dim])
    return tf.nn.tanh(tf.matmul(input_, W) + b)

def relu(input_, out_dim, scope=None, reuse=False):

    with tf.variable_scope(scope or 'relu', reuse=reuse):
        # W = tf.get_variable('W', [input_.get_shape()[1], out_dim], constraint=lambda x: tf.clip_by_value(x, -5, 5))
        # b = tf.get_variable('b', [out_dim], constraint=lambda x: tf.clip_by_value(x, -5, 5))
        W = tf.get_variable('W', [input_.get_shape()[1], out_dim])
        b = tf.get_variable('b', [out_dim])
    return tf.nn.relu(tf.matmul(input_, W) + b)


class RRL(object):
    def __init__(self, is_deep, is_lstm, is_adam, m, T, training_c,
                 testing_c, learning_rate, n_epoch):

        self.rnn_size = 1
        self.is_adam = is_adam
        self.is_deep = is_deep
        self.is_lstm = is_lstm
        self.m = m
        self.T = T
        self.training_c = training_c
        self.testing_c = testing_c
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.batch_size = 1
        self.save_path = 'tflstm_c{}_lr{}'.format(str(testing_c).replace('.', ''), str(learning_rate).replace('.', ''))
        self.build()

    def build(self):
        tf.random.set_random_seed(1)
        self.prices = tf.placeholder('float32', shape=[self.T + self.m])
        self.prices_unpacked = tf.unstack(self.prices)

        self.c = tf.placeholder(tf.float32)

        self.x = tf.placeholder('float32', shape=[1, self.T, self.m])
        self.x_returns = tf.placeholder('float32', shape=[self.T + self.m])
        self.x_unpacked = tf.unstack(self.x_returns)
        # self.N = tf.placeholder('float32')  # Current time step, used for computing moments

        if self.is_deep:
            input = relu(tf.reshape(self.x, shape=[1 * self.T, self.m]), out_dim=64)
            input = tf.reshape(input, shape=[1, self.T, 64])

        if self.is_lstm:
            self.cell_state = tf.placeholder(tf.float32, [self.batch_size, self.rnn_size])
            self.hidden_state = tf.placeholder(tf.float32, [self.batch_size, self.rnn_size])
            self.init_state = tf.nn.rnn_cell.LSTMStateTuple(self.cell_state, self.hidden_state)
            self.rnn = rnn.BasicLSTMCell(self.rnn_size, state_is_tuple=True)
            if self.is_deep:
                self.outputs, self.state = tf.nn.dynamic_rnn(self.rnn, input, initial_state=self.init_state,
                                                             dtype=tf.float32)
            else:
                self.outputs, self.state = tf.nn.dynamic_rnn(self.rnn, self.x, initial_state=self.init_state,
                                                             dtype=tf.float32)
        else:
            self.hidden_state = tf.placeholder(tf.float32, [self.batch_size, self.rnn_size])
            self.rnn = rnn.BasicRNNCell(self.rnn_size)
            if self.is_deep:
                self.outputs, self.state = tf.nn.dynamic_rnn(self.rnn, input, initial_state=self.hidden_state,
                                                         dtype=tf.float32)
            else:
                self.outputs, self.state = tf.nn.dynamic_rnn(self.rnn, self.x, initial_state=self.hidden_state,
                                                         dtype=tf.float32)

        self.outputs_unpacked = tf.unstack(tf.reshape(self.outputs, [self.T, self.rnn_size]))

        self.trading_signals = []
        self.returns = []
        for l, o in enumerate(self.outputs_unpacked):
            if l == 0:
                self.trading_signals.append(o)
                self.returns.append(self.outputs_unpacked[l] * self.x_unpacked[l + self.m])

            else:
                self.trading_signals.append(o)
                self.returns.append(self.outputs_unpacked[l] * self.x_unpacked[l + self.m] - self.c *
                               tf.abs(self.outputs_unpacked[l] - self.outputs_unpacked[l - 1]) *
                                    self.prices_unpacked[l + self.m]
                               )

        self.trading_signals = tf.concat(self.trading_signals, axis=0)
        # trading_signals = outputs
        self.returns = tf.concat(self.returns, axis=0)
        # expected_return = tf.reduce_sum(returns) / N
        # squared_expected_return = tf.reduce_sum(returns ** 2) / N
        self.moments = tf.nn.moments(self.returns, axes=0)
        # sharpe_ratio = expected_return / tf.sqrt(squared_expected_return - expected_return ** 2)
        self.sharpe_ratio = self.moments[0] / tf.sqrt(self.moments[1])

        if self.is_adam:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(- self.sharpe_ratio)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(- self.sharpe_ratio)

    def train(self):

        with open('data_30_usdjpy.pickle', 'rb') as f:
            data = pickle.load(f)
            data = data[1:]
            data_dates = data[:, 0]
            data_returns = data[:, 1]
            data_prices = data[:, 2]

        all_prices = []
        all_signals = []
        all_returns = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # print('Starting training')
            # n_batch = (len(data) - self.m) // self.T
            n_batch = 5
            for i in range(n_batch):
                i += 20
                batch_returns = data_returns[i * self.T: i * self.T + self.T + self.m]  # shape [T + m]
                batch = np.array([batch_returns[k: k + self.m] for k in range(self.T)]).reshape([1, self.T, self.m])  # shape [1, T, m]
                batch_prices = data_prices[i * self.T: i * self.T + self.T + self.m]  # shape [T + m]
                tqdm_range = tqdm(range(self.n_epoch), desc='Training')
                _current_cell_state = np.zeros((self.batch_size, self.rnn_size))
                _current_hidden_state = np.zeros((self.batch_size, self.rnn_size))
                for epoch in tqdm_range:
                    # [DEPRECATED]
                    # Train on a subset of 700 time steps and validate on the following 300
                    # times steps, we want to train on 700 t.s., therefore, batch_returns_padded
                    # and batch_prices_padded goes from 0 to n_steps + m because we feed our input
                    # vectors the last m returns, n_steps time
                    # n_steps = 1000
                    # batch_returns_padded = np.zeros_like(batch_returns)
                    # batch_returns_padded[:n_steps + m] = batch_returns[:n_steps + m]
                    # batch_prices_padded = np.zeros_like(batch_prices)
                    # batch_prices_padded[:n_steps + m] = batch_prices[:n_steps + m]
                    # batch_padded = np.zeros_like(batch)
                    # batch_padded[:n_steps] = batch[:n_steps]
                    # _, r, s, ts, op = sess.run([optimizer, returns, sharpe_ratio, trading_signals, outputs],
                    #                            feed_dict={
                    #                                 x: batch_padded,
                    #                                 x_returns: batch_returns_padded,
                    #                                 prices: batch_prices_padded,
                    #                                 N: n_steps
                    #                            })
                    # [/DEPRECATED]

                    if self.is_lstm:
                        feed_dict = {
                                     self.x: batch,
                                     self.x_returns: batch_returns,
                                     self.prices: batch_prices,
                                     # self.N: n_steps,
                                     self.cell_state: _current_cell_state,
                                     self.hidden_state: _current_hidden_state,
                                     self.c: self.training_c
                                 }
                    else:
                        feed_dict = {
                                     self.x: batch,
                                     self.x_returns: batch_returns,
                                     self.prices: batch_prices,
                                     # self.N: n_steps,
                                     self.hidden_state: _current_hidden_state,
                                     self.c: self.training_c
                                 }
                    _, r, s, ts, op, _current_state = \
                        sess.run([self.optimizer, self.returns, self.sharpe_ratio, self.trading_signals,
                                  self.outputs, self.state],
                                 feed_dict=feed_dict)

                    if self.is_lstm:
                        _current_cell_state, _current_hidden_state = _current_state
                    else:
                        _current_hidden_state = _current_state

                    tqdm_range.set_description('Training (Sharpe: {}, Cum. Return: {})'.format(
                        s, np.cumsum(r)[-1]))

                # Inference
                batch_returns_test = data_returns[i * self.T + self.T: i * self.T + 2 * self.T + self.m]
                batch_test = np.array([batch_returns_test[k: k + self.m] for k in range(self.T)]).reshape([1, self.T, self.m])
                batch_prices_test = data_prices[i * self.T + self.T: i * self.T + 2 * self.T + self.m]

                if self.is_lstm:
                    feed_dict_test = {
                        self.x: batch_test,
                        self.x_returns: batch_returns_test,
                        self.prices: batch_prices_test,
                        # self.N: 1000,
                        self.cell_state: _current_cell_state,
                        self.hidden_state: _current_hidden_state,
                        self.c: self.testing_c
                        }
                else:
                    feed_dict_test = {
                        self.x: batch_test,
                        self.x_returns: batch_returns_test,
                        self.prices: batch_prices_test,
                        # self.N: 1000,
                        self.hidden_state: _current_hidden_state,
                        self.c: self.testing_c
                        }
                r_test, s_test, t_test = sess.run([self.returns, self.sharpe_ratio, self.trading_signals],
                                                  feed_dict=feed_dict_test)

                for v in range(len(r_test)):
                    all_prices.append(batch_prices_test[v + self.m])
                    all_returns.append(r_test[v])
                    all_signals.append(t_test[v])

                print('[INFERENCE] Sharpe ratio: {} -- Cumulative return: {} [{}/{}]'.format(
                    s_test, np.cumsum(r_test)[-1], i, n_batch
                ))
        return all_returns

"""
with open('data_{}.pickle'.format(self.save_path), 'wb') as f:
    pickle.dump([all_prices, all_returns, all_signals], f)

fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
ax[0].plot(all_prices)
ax[0].set_xlabel("time")
ax[0].set_ylabel("USDJPY")
ax[0].grid(True)

ax[1].plot(all_signals, color="blue", label="With optimized weights")
ax[1].set_xlabel("time")
ax[1].set_ylabel("F")
ax[1].legend(loc="lower right")
ax[1].grid(True)

ax[2].plot(np.cumsum(all_returns), color="blue", label="With optimized weights")
ax[2].set_xlabel("time")
ax[2].set_ylabel("Sum of reward[yen]")
ax[2].legend(loc="lower right")
ax[2].grid(True)
plt.tight_layout()
plt.savefig("img/tf_imgs/rrl_prediction_{}_{}_{}.png".format(
    i * self.T + self.T, i * self.T + 2 * self.T + self.m, self.save_path), dpi=300)
plt.close()

"""


if __name__ == '__main__':
    _is_deep = True
    _is_lstm = True
    _is_adam = True
    _c = 0.0005
    _learning_rate = 0.001
    _n_epoch = 1000
    path = 'lstm_deep'
    rrl = RRL(is_deep=_is_deep, is_lstm=_is_lstm, is_adam=_is_adam, m=100, T=1000, training_c=_c,
              testing_c=0.0001, learning_rate=_learning_rate, n_epoch=_n_epoch)
    data = []
    _c_data_return = []
    for _c in [0.001, 0.0005, 0.0001, 0.00005]:
        del rrl
        tf.reset_default_graph()
        rrl = RRL(is_deep=_is_deep, is_lstm=_is_lstm, is_adam=_is_adam, m=100, T=1000, training_c=_c,
              testing_c=0.0001, learning_rate=_learning_rate, n_epoch=_n_epoch)
        returns = rrl.train()
        average = np.mean(returns)
        std = np.std(returns)
        print('is_deep: {} - is_lstm: {} - is_adam: {} - training_c: {} - learning_rate: {} - n_epoch: {}'.format(
            _is_deep, _is_lstm, _is_adam, _c, _learning_rate, _n_epoch
        ))
        data.append([_is_deep, _is_lstm, _is_adam, _c, _learning_rate, _n_epoch, average, std])
        _c_data_return.append([average, _c])
        with open('img/tf_imgs/data_{}.pickle'.format(path), 'wb') as f:
            pickle.dump(data, f)

    _c = _c_data_return[_c_data_return.index(max(_c_data_return, key=lambda x: x[0]))][1]

    _learning_rate_data_return = []
    for _learning_rate in [0.001, 0.005, 0.01, 0.008]:
        del rrl
        tf.reset_default_graph()

        rrl = RRL(is_deep=_is_deep, is_lstm=_is_lstm, is_adam=_is_adam, m=100, T=1000, training_c=_c,
                  testing_c=0.0001, learning_rate=_learning_rate, n_epoch=_n_epoch)
        returns = rrl.train()
        average = np.mean(returns)
        std = np.std(returns)
        print('is_deep: {} - is_lstm: {} - is_adam: {} - training_c: {} - learning_rate: {} - n_epoch: {}'.format(
            _is_deep, _is_lstm, _is_adam, _c, _learning_rate, _n_epoch
        ))
        data.append([_is_deep, _is_lstm, _is_adam, _c, _learning_rate, _n_epoch, average, std])
        _learning_rate_data_return.append([average, _learning_rate])
        with open('img/tf_imgs/data_{}.pickle'.format(path), 'wb') as f:
            pickle.dump(data, f)
    _learning_rate = _learning_rate_data_return[_learning_rate_data_return.index(max(_learning_rate_data_return))][1]

    _n_epoch_data_return = []
    for _n_epoch in [1, 50, 100, 500, 1000]:
        del rrl
        tf.reset_default_graph()

        rrl = RRL(is_deep=_is_deep, is_lstm=_is_lstm, is_adam=_is_adam, m=100, T=1000, training_c=_c,
                  testing_c=0.0001, learning_rate=_learning_rate, n_epoch=_n_epoch)
        returns = rrl.train()
        average = np.mean(returns)
        std = np.std(returns)
        print('is_deep: {} - is_lstm: {} - is_adam: {} - training_c: {} - learning_rate: {} - n_epoch: {}'.format(
            _is_deep, _is_lstm, _is_adam, _c, _learning_rate, _n_epoch
        ))
        data.append([_is_deep, _is_lstm, _is_adam, _c, _learning_rate, _n_epoch, average, std])
        _n_epoch_data_return.append([average, _n_epoch])
        with open('img/tf_imgs/data_{}.pickle'.format(path), 'wb') as f:
            pickle.dump(data, f)
    _n_epoch = _n_epoch_data_return[_n_epoch_data_return.index(max(_n_epoch_data_return))][1]


    """
    
    with open('img/tf_imgs/data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('img/tf_imgs/data_rnn.pickle', 'rb') as f:
        data_rnn = pickle.load(f)
    with open('img/tf_imgs/data_rnn_sgd.pickle', 'rb') as f:
        data_rnn_sgd = pickle.load(f)
    with open('img/tf_imgs/data_lstm_sgd.pickle', 'rb') as f:
        data_lstm_sgd = pickle.load(f)
    # Scatter plot
    fig, ax = plt.subplots(2, 3)

    # Varying the transaction cost
    ax[0][0].scatter([d[-1] for d in data[:4]], [d[-2] for d in data[:4]], c='blue')
    ax[0][0].scatter([d[-1] for d in data_rnn[:4]], [d[-2] for d in data_rnn[:4]], c='green')
    ax[0][0].set_xlabel("Risk")
    ax[0][0].set_ylabel("Return")
    ax[0][0].set_title('Varying transaction cost')
    ax[0][0].grid(True)

    # Varying the learning rate
    ax[0][1].scatter([d[-1] for d in data[4:8]], [d[-2] for d in data[4:8]], c='blue')
    ax[0][1].scatter([d[-1] for d in data_rnn[4:8]], [d[-2] for d in data_rnn[4:8]], c='green')
    ax[0][1].set_xlabel("Risk")
    ax[0][1].set_ylabel("Return")
    ax[0][1].set_title('Varying learning rate')
    ax[0][1].grid(True)

    # Varying number epochs
    ax[0][2].scatter([d[-1] for d in data[8:]], [d[-2] for d in data[8:]], c='blue')
    ax[0][2].scatter([d[-1] for d in data_rnn[8:]], [d[-2] for d in data_rnn[8:]], c='green')
    ax[0][2].set_xlabel("Risk")
    ax[0][2].set_ylabel("Return")
    ax[0][2].set_title('Varying # epochs')
    ax[0][2].grid(True)


    # GD vs Adam
    ax[1][0].scatter([d[-1] for d in data_lstm_sgd[:4]], [d[-2] for d in data_lstm_sgd[:4]], c='blue')
    ax[1][0].scatter([d[-1] for d in data_rnn_sgd[:4]], [d[-2] for d in data_rnn_sgd[:4]], c='green')
    ax[1][0].set_xlabel("Risk")
    ax[1][0].set_ylabel("Return")
    ax[1][0].set_title('Varying transaction cost')
    ax[1][0].grid(True)

    # Varying the learning rate
    ax[1][1].scatter([d[-1] for d in data_lstm_sgd[4:8]], [d[-2] for d in data_lstm_sgd[4:8]], c='blue')
    ax[1][1].scatter([d[-1] for d in data_rnn_sgd[4:8]], [d[-2] for d in data_rnn_sgd[4:8]], c='green')
    ax[1][1].set_xlabel("Risk")
    ax[1][1].set_ylabel("Return")
    ax[1][1].set_title('Varying learning rate')
    ax[1][1].grid(True)

    # Varying number epochs
    ax[1][2].scatter([d[-1] for d in data_lstm_sgd[9:]], [d[-2] for d in data_lstm_sgd[9:]], c='blue')
    ax[1][2].scatter([d[-1] for d in data_rnn_sgd[9:]], [d[-2] for d in data_rnn_sgd[9:]], c='green')
    ax[1][2].set_xlabel("Risk")
    ax[1][2].set_ylabel("Return")
    ax[1][2].set_title('Varying # epochs')
    ax[1][2].grid(True)
    plt.tight_layout()

    """

    # Deep network vs no deep

"""
import pickle
with open('img/tf_imgs/data_rnn_sgd.pickle', 'rb') as f:
    tt = pickle.load(f)
print(len(tt))
for t in tt:
    print(t)
"""

"""
# rnn_size = 128
rnn_size = 1
batch_size = 1
m = 100         # number of past returns to consider in the NN
T = 1000        # Number of time steps for training
# c = 5e-4         # Transaction cost
training_c = 0.0005
testing_c = 0.0001
learning_rate = 0.1       # Learning rate
n_epoch = 500
save_path = 'tflstm_c{}_lr{}'.format(str(testing_c).replace('.', ''), str(learning_rate).replace('.', ''))

prices = tf.placeholder('float32', shape=[T + m])
prices_unpacked = tf.unstack(prices)

c = tf.placeholder(tf.float32)

# Eventually, replace this with tf.contrib.data.sliding_window_batch
x = tf.placeholder('float32', shape=[1, T, m])
x_returns = tf.placeholder('float32', shape=[T + m])
x_unpacked = tf.unstack(x_returns)
N = tf.placeholder('float32')     # Current time step, used for computing moments

cell_state = tf.placeholder(tf.float32, [batch_size, rnn_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, rnn_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
# init_state = tf.nn.rnn_cell.BasicRNNCell()
# lstm = rnn.BasicRNNCell(rnn_size)
lstm = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
# initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
# outputs, state = tf.nn.dynamic_rnn(lstm, x, initial_state=init_state, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(lstm, x, initial_state=init_state, dtype=tf.float32)

outputs_unpacked = tf.unstack(tf.reshape(outputs, [T, rnn_size]))

trading_signals = []
returns = []
for l, o in enumerate(outputs_unpacked):
    if l == 0:
        # trading_signals.append(tf.reshape(
        #     tanh(tf.reshape(o, [1, rnn_size]), out_dim=1, scope='tanh', reuse=False), [1]))
        trading_signals.append(o)
        # returns.append(trading_signals[-1] * x_unpacked[l + m])
        returns.append(outputs_unpacked[l] * x_unpacked[l + m])
        # returns.append(tf.sign(trading_signals[-1]) * x_unpacked[l + m])
    else:
        trading_signals.append(o)
        # trading_signals.append(tf.reshape(
        #     tanh(tf.reshape(o, [1, rnn_size]), out_dim=1, scope='tanh', reuse=True), [1]))
        # returns.append(tf.sign(trading_signals[-1]) * x_unpacked[l + m] - c * (
        #                 tf.sign(trading_signals[-1]) - tf.sign(trading_signals[-2])
        #         ))
        # returns.append(trading_signals[-1] * x_unpacked[l + m] - c *
        #                tf.abs(trading_signals[-1] - trading_signals[-2]) * prices_unpacked[l + m]
        #                )
        returns.append(outputs_unpacked[l] * x_unpacked[l + m] - c *
                       tf.abs(outputs_unpacked[l] - outputs_unpacked[l - 1]) * prices_unpacked[l + m]
                       )

trading_signals = tf.concat(trading_signals, axis=0)
# trading_signals = outputs
returns = tf.concat(returns, axis=0)
# expected_return = tf.reduce_sum(returns) / N
# squared_expected_return = tf.reduce_sum(returns ** 2) / N
moments = tf.nn.moments(returns, axes=0)
# sharpe_ratio = expected_return / tf.sqrt(squared_expected_return - expected_return ** 2)
sharpe_ratio = moments[0] / tf.sqrt(moments[1])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(- sharpe_ratio)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(- sharpe_ratio)




all_prices = []
all_signals = []
all_returns = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print('Starting training')
    n_batch = (len(data) - m) // T
    for i in range(n_batch):
        i += 20
        batch_returns = data_returns[i * T: i * T + T + m]      # shape [T + m]
        batch = np.array([batch_returns[k: k + m] for k in range(T)]).reshape([1, T, m])        # shape [1, T, m]
        batch_prices = data_prices[i * T: i * T + T + m]        # shape [T + m]
        tqdm_range = tqdm(range(n_epoch), desc='Training')
        _current_cell_state = np.zeros((batch_size, rnn_size))
        _current_hidden_state = np.zeros((batch_size, rnn_size))
        for epoch in tqdm_range:
            # Train on a subset of 700 time steps and validate on the following 300
            # times steps, we want to train on 700 t.s., therefore, batch_returns_padded
            # and batch_prices_padded goes from 0 to n_steps + m because we feed our input
            # vectors the last m returns, n_steps time
            n_steps = 1000

            # batch_returns_padded = np.zeros_like(batch_returns)
            # batch_returns_padded[:n_steps + m] = batch_returns[:n_steps + m]
            # batch_prices_padded = np.zeros_like(batch_prices)
            # batch_prices_padded[:n_steps + m] = batch_prices[:n_steps + m]
            # batch_padded = np.zeros_like(batch)
            # batch_padded[:n_steps] = batch[:n_steps]
            # _, r, s, ts, op = sess.run([optimizer, returns, sharpe_ratio, trading_signals, outputs],
            #                            feed_dict={
            #                                 x: batch_padded,
            #                                 x_returns: batch_returns_padded,
            #                                 prices: batch_prices_padded,
            #                                 N: n_steps
            #                            })
            _, r, s, ts, op, _current_state = \
                sess.run([optimizer, returns, sharpe_ratio, trading_signals,
                          outputs, state],
                         feed_dict={
                                x: batch,
                                x_returns: batch_returns,
                                prices: batch_prices,
                                N: n_steps,
                                cell_state: _current_cell_state,
                                hidden_state: _current_hidden_state,
                                c: training_c
                         })
            _current_cell_state, _current_hidden_state = _current_state
            # _current_hidden_state = _current_state
            tqdm_range.set_description('Training (Sharpe: {}, Cum. Return: {})'.format(
                s, np.cumsum(r)[-1]))
            # print(op)
            # if epoch == 999:
            #     print('[TRAINING] Cumulative return: {} -- Sharpe ratio: {}'.format(np.cumsum(r)[-1], s))
        # Inference
        batch_returns_test = data_returns[i * T + T: i * T + 2 * T + m]
        batch_test = np.array([batch_returns_test[k: k + m] for k in range(T)]).reshape([1, T, m])
        batch_prices_test = data_prices[i * T + T: i * T + 2 * T + m]

        r_test, s_test, t_test = sess.run([returns, sharpe_ratio, trading_signals],
                                          feed_dict={x: batch_test,
                                                     x_returns: batch_returns_test,
                                                     prices: batch_prices_test,
                                                     N: 1000,
                                                     cell_state: _current_cell_state,
                                                     hidden_state: _current_hidden_state,
                                                     c: testing_c
                                                     })

        for v in range(len(r_test)):
            all_prices.append(batch_prices_test[v + m])
            all_returns.append(r_test[v])
            all_signals.append(t_test[v])

        print('[INFERENCE] Sharpe ratio: {} -- Cumulative return: {} [{}/{}]'.format(
            s_test, np.cumsum(r_test)[-1], i, n_batch
        ))

        # Risk management

        with open('data_{}.pickle'.format(save_path), 'wb') as f:
            pickle.dump([all_prices, all_returns, all_signals], f)

        fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
        ax[0].plot(all_prices)
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("USDJPY")
        ax[0].grid(True)

        ax[1].plot(all_signals, color="blue", label="With optimized weights")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("F")
        ax[1].legend(loc="lower right")
        ax[1].grid(True)

        ax[2].plot(np.cumsum(all_returns), color="blue", label="With optimized weights")
        ax[2].set_xlabel("time")
        ax[2].set_ylabel("Sum of reward[yen]")
        ax[2].legend(loc="lower right")
        ax[2].grid(True)
        plt.tight_layout()
        plt.savefig("img/tf_imgs/rrl_prediction_{}_{}_{}.png".format(
            i * T + T,  i * T + 2 * T + m, save_path), dpi=300)
        plt.close()
"""