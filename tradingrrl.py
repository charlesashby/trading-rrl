import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm


class TradingRRL(object):
    """
    credit for this class: https://github.com/darden1/tradingrrl
    """
    def __init__(self, T=1000, M=200, init_t=10000, mu=1, sigma=0.1, rho=1.0, n_epoch=10000):
        self.T = T
        self.M = M                      # History size fed to the network
        self.init_t = init_t
        self.mu = mu
        self.sigma = sigma
        self.rho = rho                  # Learning rate
        self.all_t = None               # All time steps reversed (2017-2010)
        self.all_p = None               # All prices reversed (2017-2010)
        self.t = None                   # Time
        self.p = None                   # Price
        self.r = None                   # Reward
        self.x = np.zeros([T, M+2])     # Features fed to the network
        self.F = np.zeros(T+1)          # Position to take F_t = tanh(w^t x_t) \in [-1, 1]
        self.R = np.zeros(T)            # Rewards in yen $ at each time step
        self.w = np.ones(M+2)
        self.w_opt = np.ones(M+2)
        self.epoch_S = np.empty(0)
        self.n_epoch = n_epoch
        self.progress_period = 100
        self.q_threshold = 0.7          # Threshold for actually buying/shorting (output of tanh) -- not used

    def change_T(self, t_size):
        self.T = t_size
        self.x = np.zeros(shape=[t_size, self.M + 2])
        self.F = np.zeros(shape=[t_size + 1])
        self.R = np.zeros(shape=[t_size])

    def load_csv(self, fname):
        tmp = pd.read_csv(fname, header=None)
        tmp_tstr = tmp[0] + " " + tmp[1]
        tmp_t = [dt.strptime(tmp_tstr[i], '%Y.%m.%d %H:%M') for i in range(len(tmp_tstr))]
        tmp_p = list(tmp[5])
        self.all_t = np.array(tmp_t[::-1])
        self.all_p = np.array(tmp_p[::-1])

    def set_t_p_r(self):
        self.t = self.all_t[self.init_t:self.init_t+self.T+self.M+1]
        self.p = self.all_p[self.init_t:self.init_t+self.T+self.M+1]
        self.r = -np.diff(self.p)

    def set_x_F(self):
        """ 1. This method sets, for each time step (T), the position
            to take by considering an history of size M

            2. Trader receives as features the last trade F_{t-1},
            the last M returns r_t, ..., r_{t-M} and we add a
            bias term by appending 1 to the feature vector. Thus,
            w \in R^{M + 2}
        """
        for i in range(self.T-1, -1, -1):
            self.x[i] = np.zeros(self.M+2)
            self.x[i][0] = 1.0
            self.x[i][self.M+2-1] = self.F[i+1]
            for j in range(1, self.M+2-1, 1):
                self.x[i][j] = self.r[i+j-1]
            self.F[i] = np.tanh(np.dot(self.w, self.x[i]))

    def calc_R(self):
        self.R = self.mu * (self.F[1:] * self.r[:self.T] - self.sigma * np.abs(-np.diff(self.F)))

    def calc_sumR(self):
        self.sumR  = np.cumsum(self.R[::-1])[::-1]
        self.sumR2  = np.cumsum((self.R**2)[::-1])[::-1]

    def calc_dSdw(self):
        self.set_x_F()
        self.calc_R()
        self.calc_sumR()
        self.A      =  self.sumR[0] / self.T
        self.B      =  self.sumR2[0] / self.T
        self.S      =  self.A / np.sqrt(self.B - self.A**2)
        self.dSdA   =  self.S * (1 + self.S**2) / self.A
        self.dSdB   = -self.S**3 / 2 / self.A**2
        self.dAdR   =  1.0 / self.T
        self.dBdR   =  2.0 / self.T * self.R
        self.dRdF   = -self.mu * self.sigma * np.sign(-np.diff(self.F))
        self.dRdFp  =  self.mu * self.r[:self.T] + self.mu * self.sigma * np.sign(-np.diff(self.F))
        self.dFdw   = np.zeros(self.M+2)
        self.dFpdw  = np.zeros(self.M+2)
        self.dSdw   = np.zeros(self.M+2)
        for i in range(self.T-1, -1, -1):
            if i != self.T-1:
                self.dFpdw = self.dFdw.copy()
            self.dFdw  = (1 - self.F[i]**2) * (self.x[i] + self.w[self.M+2-1] * self.dFpdw)
            self.dSdw += (self.dSdA * self.dAdR + self.dSdB * self.dBdR[i]) * \
                         (self.dRdF[i] * self.dFdw + self.dRdFp[i] * self.dFpdw)

    def update_w(self):
        self.w += self.rho * self.dSdw

    def fit(self):
        
        pre_epoch_times = len(self.epoch_S)

        self.calc_dSdw()
        # print("Epoch loop start. Initial sharp's ratio is " + str(self.S) + ".")
        self.S_opt = self.S
        
        tic = time.clock()
        for e_index in tqdm(range(self.n_epoch), desc='Fitting weights'):
            self.calc_dSdw()
            if self.S > self.S_opt:
                self.S_opt = self.S
                self.w_opt = self.w.copy()
            self.epoch_S = np.append(self.epoch_S, self.S)
            self.update_w()
            # if e_index % self.progress_period == self.progress_period-1:
                # toc = time.clock()
                # import pdb; pdb.set_trace()
                # print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" +
                #       str(self.n_epoch + pre_epoch_times) +". Sharpe's ratio: " +
                #       str(self.S) + ". Elapsed time: " + str(toc-tic) + " sec." +
                #       "-- Reward: " + str(np.sum(self.R)))
        # toc = time.clock()
        # print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" +
        #       str(self.n_epoch + pre_epoch_times) +". Sharpe's ratio: " +
        #       str(self.S) + ". Elapsed time: " + str(toc-tic) + " sec.")
        self.w = self.w_opt.copy()
        self.calc_dSdw()
        # print("Epoch loop end. Optimized sharp's ratio is " + str(self.S_opt) + ".")

    def save_weight(self):
        pd.DataFrame(self.w).to_csv("w.csv", header=False, index=False)
        pd.DataFrame(self.epoch_S).to_csv("epoch_S.csv", header=False, index=False)
        
    def load_weight(self):
        tmp = pd.read_csv("w.csv", header=None)
        self.w = tmp.T.values[0]


class LayeredRRL(TradingRRL):
    def __init__(self, T=1000, M=200, init_t=10000, mu=1, sigma=0.04, rho=1.0, n_epoch=10000, save_path='data.pickle'):
        TradingRRL.__init__(self, T, M, init_t, mu, sigma, rho, n_epoch)
        self.position = 0       # Position at the moment \in {-1, 0, 1}
        self.position_max = - np.inf    # Maximal value of the current position
        self.W_position = 0.        # Cumulative profit for the position
        self.stop_loss = 0.5        # Stop-loss parameter when position goes below stop_loss * position_max
        self.cool_down = 0      # Cool down param, if 1 than wait before trading
        self.W = 0      # Cumulative wealth (profit)
        self.W_max = - np.inf       # Maximal cumulative profit
        self.z = 0.3        # Shut down parameter, when cumulative profit goes below z * w_max
        self.y = 0.7        # Threshold for validating trading signal
        self.load_csv(fname='USDJPY30.csv')
        self.trading_start_t = self.all_t[self.init_t]      # Time at which trading starts
        self.trading_end_t = None       # Time at which trading ends
        self.c = 0.005       # Transaction costs
        self.all_W = []      # Cumulative profits for all testing time steps
        self.all_F = []      # All signals from the model
        self.all_prices = []        # All prices on which we traded at test time
        self.nu = 0.5       # Trader's risk aversion
        self.save_path = save_path

    def risk_management(self):
        """ To implement the risk management layer we need to:
                - Store the maximal price of a position since we took it to
                  implement the trailing stop-loss
                - Store the stop-loss parameter x. When the maximal price - the
                  actual price of a certain position is below x, we exit the position
                  and assume the market is behaving irrationally
                - Store a cool-down parameter for stopping trading when we reach
                  the stop loss before an exit signal is given by layer 1. This is
                  a constant number of time steps (e.g. 1)
                - Store cumulative profits and a performance management parameter z
                  to shut down the system when a draw-down from the maximum in
                  cumulative profits is larger than this parameter
                - Threshold parameter y for validating a trading signal (only trade
                  when the signal is greater than y)

        :return:
            Updates cumulative profit w, w_max, position, position_max and cool_down
            using F and p
        """

        if self.all_W:
            last_W = self.all_W[-1]
        else:
            last_W = 0.

        for i, r in enumerate(self.r[::-1][:self.T]):
            # self.r.shape = self.T + M and self.F.shape = self.T + 1
            # Update cumulative profit and check trailing stop-loss
            # print('i: {} -- W_position: {} -- position_max: {} -- W: {} -- pos * sl: {}'.format(
            #     i, self.W_position, self.position_max, self.W, self.position_max * self.stop_loss
            # ))
            exit = False
            # signal = self.F[::-1][i + 1]
            signal = 1.
            self.W_position += self.position * r
            self.W += self.position * r
            # print('Prices: {}/{} -- return: {} -- W: {} -- position: {}'.format(
            #     self.p[::-1][i], self.p[::-1][i - 1], r, self.W, self.position
            # ))
            self.W_max = np.maximum(self.W, self.W_max)
            self.position_max = np.maximum(self.W_position, self.position_max)

            # Update cool-down parameter
            self.cool_down = np.maximum(self.cool_down - 1, 0.)

            if 2. < self.W_position < self.position_max * self.stop_loss and self.cool_down == 0.:
                # Cool-down trading for 100 time steps; the market is behaving irrationally.
                # Since we exit the position at this point we need to pay transaction costs
                print('[SYSTEM COOL DOWN] Total profit (Yen): {} -- Time: {} -- Observed a drop of {} Yens'.format(
                    self.W, self.t[::-1][i], self.position_max - self.W_position
                ))
                self.W -= self.c * 2 * self.p[::-1][i]
                self.position = 0
                self.cool_down = np.minimum(self.T - i, 100)
                self.W_position = 0.
                self.position_max = - np.inf

            elif np.abs(last_W - self.W) > np.abs(last_W) * self.z \
                    and self.cool_down == 0. and self.W < - 5.:
                # Cool-down trading until the end of the testing session
                print('[SYSTEM SHUT DOWN] Total profit (Yen): {} -- Trading started at: {} and ended at: {}'.format(
                    self.W, self.trading_start_t, self.t[::-1][i]
                ))
                self.W -= self.c * 2 * self.p[::-1][i]
                self.position = 0
                self.cool_down = self.T - i
                self.W_position = 0.
                self.position_max = - np.inf

            elif abs(signal) > self.y and np.sign(signal) != self.position and self.cool_down == 0.:
                # Trading signal is strong enough to close the current position
                # and stop-loss have not been reached and we're not in a cool-down phase
                exit = True
                new_position = np.sign(signal)

            # If exit signal, add transaction costs to the cumulative profits and
            # set new position
            if exit:
                if self.position != 0:
                    # If the position is neutral, than we don't pay
                    # the transaction costs yet (note that transaction
                    # cost is doubled when we close a position to account
                    # for the position we're closing and the one we're
                    # opening)
                    self.W -= self.c * 2 * self.p[::-1][i]
                self.position = new_position
                self.W_position = 0.
                self.position_max = - np.inf

            # We store cumulative wealth, signals and prices for plots and
            # the optimization layer
            self.all_W.append(self.W)
            self.all_F.append(signal)
            self.all_prices.append(self.p[::-1][i])
            last_W = self.W

    def train(self, optimization=False, optimization_i=None):
        initial_t = 63000
        self.T = 1000
        T_test = 1000
        self.M = 200
        self.mu = 1
        # self.sigma = 0.1
        # self.rho = 1.
        # self.n_epoch = 1000
        fname = 'USDJPY30.csv'
        self.load_csv(fname)
        self.set_t_p_r()

        if optimization:
            n_batch = optimization_i
        else:
            n_batch = (initial_t - self.T) // T_test

        for i in range(n_batch):
            # Train data goes from init_t - T_test * i to init_t - T_test * i - T
            train_init_t = initial_t - T_test * i
            self.init_t = train_init_t
            self.set_t_p_r()

            print('[{}/{}] Training from {} to {}'.format(i, n_batch, self.t[::-1][0], self.t[::-1][self.T]))
            self.calc_dSdw()
            self.fit()

            # Fit hyper-parameters using the optimization layer
            if i % 10000:
                learning_rate, stop_loss, sigma, n_epoch = optimization_layer(i)
                self.rho = learning_rate
                self.stop_loss = stop_loss
                self.sigma = sigma
                self.n_epoch = n_epoch

            # Once the agent is trained for n_epoch, we can test it on the next
            # T_test time steps. Test data goes from init_t - T_test * i - T to
            # init_t - (T_test + 1) * i - T
            test_init_t = initial_t - T_test * i - self.T
            self.init_t = test_init_t
            self.set_t_p_r()
            print('[{}/{}] Testing from {} to {}'.format(i, n_batch, self.t[::-1][0], self.t[::-1][self.T]))
            self.calc_dSdw()

            # The risk management layer computes the actual profits and check stop-losses
            self.risk_management()

            # Save data
            # with open('data.pickle', 'wb') as f:
            #     pickle.dump([self.all_W, self.all_prices, self.all_F, self.w], f)

            # We print the profit and signals from the network at each time steps
            fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
            ax[0].plot(self.all_prices)
            ax[0].set_xlabel("time")
            ax[0].set_ylabel("USDJPY")
            ax[0].grid(True)

            ax[1].plot(self.all_F, color="blue", label="With optimized weights")
            ax[1].set_xlabel("time")
            ax[1].set_ylabel("F")
            ax[1].legend(loc="lower right")
            ax[1].grid(True)

            ax[2].plot(self.all_W, color="blue", label="With optimized weights")
            ax[2].set_xlabel("time")
            ax[2].set_ylabel("Sum of reward[yen]")
            ax[2].legend(loc="lower right")
            ax[2].grid(True)
            plt.tight_layout()
            plt.savefig("img/rrl_prediction_tc005_{}_{}_{}.png".format(
                test_init_t - T_test, test_init_t, self.save_path), dpi=300)
            plt.close()


def optimization_layer(optimization_i, nu=0.5, alpha=1.):
    """ We want to maximize a risk measure sigma and a utility function U
        defined, respectively by:

        sigma = \sum_{i=0->n} (R_i)^2 I(R_i < 0) / \sum_{i=0->n} (R_i)^2 I(R_i>0)
        U = \alpha * (1 - \nu) * \hat{R} - nu * sigma

        The strategy raw return at time i is R_i = W_i - W_{i-1} and the cumulative
        profit at time is is W_i and \hat{R} = W_N / N is the average profit per
        time interval

        \nu is the trader's personal risk aversion

        The goal is to find max(U) using random search. We try values
        for each params while fixing the others, we then keep the ones that maximize
        utility for each of them.
        ------------------------------------------------------------------------------
    :returns:
        hyper-parameters with maximal utility
    """
    stop_loss = 0.3
    sigma = 0.2
    n_epoch = 1000
    stop_losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 0.3
    learning_rates = [0.1, 0.3, 1., 1.5]
    transaction_costs = [0, 0.01, 0.05, 0.1, 0.2, 0.5] # 0.2
    n_epochs = [0, 100, 500, 1000, 2000, 4000] # 1000
    data = []
    for i, learning_rate in enumerate(learning_rates):
        rrl = LayeredRRL(save_path='learning_rate_{}'.format(i))
        rrl.stop_loss = stop_loss
        rrl.rho = learning_rate
        rrl.sigma = sigma
        rrl.n_epoch = n_epoch
        rrl.train(optimization=True, optimization_i=optimization_i)

        # Compute utility
        rs = np.diff(rrl.all_W)
        sigma = np.sum((rs ** 2) * (rs < 0)) / np.sum((rs ** 2) * (rs >= 0))
        hat_r = np.cumsum(rrl.all_W) / len(rrl.all_W)
        U = alpha * (1 - nu) * hat_r - nu * sigma
        data.append(U)

    learning_rate = learning_rates[data.index(max(data))]

    data = []
    for i, stop_loss in enumerate(stop_losses):
        rrl = LayeredRRL(save_path='learning_rate_{}'.format(i))
        rrl.stop_loss = stop_loss
        rrl.rho = learning_rate
        rrl.sigma = sigma
        rrl.n_epoch = n_epoch
        rrl.train(optimization=True, optimization_i=optimization_i)

        # Compute utility
        rs = np.diff(rrl.all_W)
        sigma = np.sum((rs ** 2) * (rs < 0)) / np.sum((rs ** 2) * (rs >= 0))
        hat_r = np.cumsum(rrl.all_W) / len(rrl.all_W)
        U = alpha * (1 - nu) * hat_r - nu * sigma
        data.append(U)

    stop_loss = stop_losses[data.index(max(data))]

    data = []
    for i, sigma in enumerate(transaction_costs):
        rrl = LayeredRRL(save_path='learning_rate_{}'.format(i))
        rrl.stop_loss = stop_loss
        rrl.rho = learning_rate
        rrl.sigma = sigma
        rrl.n_epoch = n_epoch
        rrl.train(optimization=True, optimization_i=optimization_i)

        # Compute utility
        rs = np.diff(rrl.all_W)
        sigma = np.sum((rs ** 2) * (rs < 0)) / np.sum((rs ** 2) * (rs >= 0))
        hat_r = np.cumsum(rrl.all_W) / len(rrl.all_W)
        U = alpha * (1 - nu) * hat_r - nu * sigma
        data.append(U)

    sigma = transaction_costs[data.index(max(data))]

    data = []
    for i, n_epoch in enumerate(n_epochs):
        rrl = LayeredRRL(save_path='learning_rate_{}'.format(i))
        rrl.stop_loss = stop_loss
        rrl.rho = learning_rate
        rrl.sigma = sigma
        rrl.n_epoch = n_epoch
        rrl.train(optimization=True, optimization_i=optimization_i)

        # Compute utility
        rs = np.diff(rrl.all_W)
        sigma = np.sum((rs ** 2) * (rs < 0)) / np.sum((rs ** 2) * (rs >= 0))
        hat_r = np.cumsum(rrl.all_W) / len(rrl.all_W)
        U = alpha * (1 - nu) * hat_r - nu * sigma
        data.append(U)

    n_epoch = n_epochs[data.index(max(data))]

    return learning_rate, stop_loss, sigma, n_epoch


if __name__ == "__main__":
    init_t = 12000   # Time step at which we start training (2015-12-22)
    T = 1000    # Training interval period
    T_test = 200    # Testing interval period
    M = 200     # History size for updating weights at each time step
    mu = 10000  # Number of shares bought at each time step
    sigma = 0.1    # Transaction cost
    rho = 1.0       # Learning rate for weights updates
    n_epoch = 1000     # Number of epochs to train

    # with open('data.pickle', 'rb') as f:
    #     tt = pickle.load(f)
    # rrl.all_W = tt[0]
    # rrl.all_prices = tt[1]
    # rrl.all_F = tt[2]
    # rrl.w = tt[3]
    rrl = LayeredRRL(save_path='all_ticks_long')
    rrl.stop_loss = 0.3
    rrl.rho = 0.1
    rrl.sigma = 0.2
    rrl.n_epoch = 0
    rrl.train()
    with open('data/data_all_ticks_long.pickle', 'wb') as f:
        pickle.dump([rrl.all_W, rrl.all_prices, rrl.all_F, rrl.w], f)

    """ 
    
    # Varying the stop-loss Training from 15000-25000
    # stop_losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 0.3
    learning_rates = [0.1, 0.3, 1., 1.5]
    # transaction_costs = [0, 0.01, 0.05, 0.1, 0.2, 0.5] 0.2
    # n_epochs = [0, 100, 500, 1000, 2000, 4000] 1000
    for i, learning_rate in enumerate(learning_rates):
        rrl = LayeredRRL(save_path='learning_rate_{}'.format(i))
        rrl.stop_loss = 0.3
        rrl.rho = learning_rate
        rrl.sigma = 0.2
        rrl.n_epoch = 1000
        rrl.train()
        with open('data/data_learning_rate_{}.pickle'.format(i), 'wb') as f:
            pickle.dump([rrl.all_W, rrl.all_prices, rrl.all_F, rrl.w], f)
    
    stop_losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ws = np.zeros(shape=[15, 14000])
    returns = np.diff(rrl.all_prices)
    for j, stop_loss in enumerate(stop_losses):
        print('[FINDING HYPERPARAMETER] Stop loss: {} [{}/{}]'.format(stop_loss, j, len(stop_losses)))
        rrl.stop_loss = stop_loss
        W_position = 0
        position = 0
        W = 0.
        W_max = - np.inf
        position_max = - np.inf
        last_W = 0.
        cool_down = 0.
        for i, r in enumerate(returns):
            # self.r.shape = self.T + M and self.F.shape = self.T + 1
            # Update cumulative profit and check trailing stop-loss
            # print('i: {} -- W_position: {} -- position_max: {} -- W: {} -- pos * sl: {}'.format(
            #     i, self.W_position, self.position_max, self.W, self.position_max * self.stop_loss
            # ))
            exit = False
            signal = rrl.all_F[i + 1]
            W_position += position * r
            W += position * r
            # print('Prices: {}/{} -- return: {} -- W: {} -- position: {}'.format(
            #     self.p[::-1][i], self.p[::-1][i - 1], r, self.W, self.position
            # ))
            W_max = np.maximum(W, W_max)
            position_max = np.maximum(W_position, position_max)

            # Update cool-down parameter
            cool_down = np.maximum(cool_down - 1, 0.)

            if 2. < W_position < position_max * stop_loss and cool_down == 0.:
                # Cool-down trading for 100 time steps; the market is behaving irrationally.
                # Since we exit the position at this point we need to pay transaction costs
                print('[SYSTEM COOL DOWN] Total profit (Yen): {} -- Observed a drop of {} Yens'.format(
                    W, position_max - W_position
                ))
                W -= rrl.c * 2 * rrl.all_prices[i]
                position = 0
                cool_down = 100
                W_position = 0.
                position_max = - np.inf

            elif np.abs(last_W - W) > np.abs(last_W) * rrl.z \
                    and cool_down == 0. and W < - 5.:
                # Cool-down trading until the end of the testing session
                print('[SYSTEM SHUT DOWN] Total profit (Yen): {}'.format(W))
                W -= rrl.c * 2 * rrl.all_prices[::-1][i]
                position = 0
                cool_down = 1000
                W_position = 0.
                position_max = - np.inf

            elif abs(signal) > rrl.y and np.sign(signal) != position and cool_down == 0.:
                # Trading signal is strong enough to close the current position
                # and stop-loss have not been reached and we're not in a cool-down phase
                exit = True
                new_position = np.sign(signal)

            # If exit signal, add transaction costs to the cumulative profits and
            # set new position
            if exit:
                if position != 0:
                    # If the position is neutral, than we don't pay
                    # the transaction costs yet (note that transaction
                    # cost is doubled when we close a position to account
                    # for the position we're closing and the one we're
                    # opening)
                    W -= rrl.c * 2 * rrl.all_prices[::-1][i]
                position = new_position
                W_position = 0.
                position_max = - np.inf
            ws[j, i] = W

    
    
    
    
    
    
    
    We will test the algorithm on one year, roughly 15,000 time steps. Here's
        how the procedure works: We train on T steps for n_epoch and then compute
        the rewards on the following T_test time steps, we move T_test time steps 
        forward and retrain the algorithm. At the end, we plot the cumulative reward
        for each time steps on the whole year
        
        rrl = TradingRRL(T, M, init_t, mu, sigma, rho, n_epoch)
    fname = "USDJPY30.csv"
    rrl.load_csv(fname)
    rrl.set_t_p_r()

    n_batch = (init_t - T) // T_test    # Number of batches of training/testing data

    tt = np.zeros(shape=[n_batch * 200])
    pp = np.zeros_like(tt)
    rrl_init_ = np.zeros_like(tt)
    rrl_ = np.zeros_like(tt)
    rrl_init_F_ = np.zeros_like(tt)
    rrl_F_ = np.zeros_like(tt)

    for i in range(n_batch):
        # Train data goes from init_t - T_test * i to init_t - T_test * i - T
        train_init_t = init_t - T_test * i
        rrl.change_T(T)
        rrl.init_t = train_init_t
        rrl.set_t_p_r()
        rrl.calc_dSdw()

        # Training with optimized weights (when agent has
        # been trained for 1 epoch)
        rrl.fit()

        # Once the agent is trained for n_epoch, we can test it on the next
        # T_test time steps. Test data goes from init_t - T_test * i - T to
        # init_t - (T_test + 1) * i - T
        test_init_t = init_t - T_test * i - T
        rrl.init_t = test_init_t
        rrl.change_T(T_test)
        rrl.set_t_p_r()
        rrl.calc_dSdw()
        ini_rrl_f = TradingRRL(T_test, M, test_init_t, mu, sigma, rho, n_epoch)
        ini_rrl_f.all_t = rrl.all_t
        ini_rrl_f.all_p = rrl.all_p
        ini_rrl_f.set_t_p_r()
        ini_rrl_f.calc_dSdw()

        # rrl_[i * T_test: (i + 1) * T_test] = rrl.R
        # rrl_F_[i * T_test: (i + 1) * T_test] = rrl.F
        # rrl_init_[i * T_test: (i + 1) * T_test] = ini_rrl_f.R
        # rrl_init_F_[i * T_test: (i + 1) * T_test] = ini_rrl_f.F
        # pp[i * T_test: (i + 1) * T_test] = rrl.p

        fig, ax = plt.subplots(nrows=3, figsize=(15, 10))
        t_f = np.linspace(rrl.T+1, rrl.T+ T_test, rrl.T)[::-1]
        ax[0].plot(t_f[:T_test], rrl.p[:T_test])
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("USDJPY")
        ax[0].grid(True)

        ax[1].plot(t_f, rrl.F[1:], color="blue", label="With optimized weights")
        ax[1].plot(t_f, ini_rrl_f.F[1:], color="red", label="With initial weights")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("F")
        ax[1].legend(loc="lower right")
        ax[1].grid(True)

        ax[2].plot(t_f, rrl.sumR, color="blue", label="With optimized weights")
        ax[2].plot(t_f, ini_rrl_f.sumR, color="red", label="With initial weights")
        ax[2].set_xlabel("time")
        ax[2].set_ylabel("Sum of reward[yen]")
        ax[2].legend(loc="lower right")
        ax[2].grid(True)
        plt.tight_layout()
        plt.savefig("img/rrl_prediction1_{}_{}.png".format(test_init_t, test_init_t + T_test), dpi=300)
        plt.close()

    """






