from distutils.command import config
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from Transformer.models.transformer import Transformer_base as PredictionModel

import torch
from collections import OrderedDict

import os
import datetime
import pdb
import pickle as pkl


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        temporal_feature_list,
        additional_list,
        time_window_start, # should be a list
        short_prediction_model_path = None,
        long_prediction_model_path = None,
        step_len=1000,
        temporal_len=60,
        figure_path='results/',
        logs_path='results/',
        csv_path = 'results/',
        mode="train",
        hidden_channel=4,
        make_plots=True,
        print_verbosity=1,
        initial=True,
        model_name="",
        iteration="",
        device='cuda:0',
        print_additional_flag=0,
    ):
        # start time
        self.start_day = time_window_start[0]
        self.day = self.start_day
        self.time_window_start = time_window_start
        self.time_windows_point = 0
        self.step_len = step_len

        # help file
        self.log_name = logs_path+mode+'.txt'
        self.figure_path = figure_path
        self.csv_path = csv_path
        os.makedirs(logs_path, exist_ok=True)
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)
    
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.transaction_cost_pct = transaction_cost_pct

        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_dim = action_space
        self.tech_indicator_list = tech_indicator_list
        self.temporal_feature_list = temporal_feature_list
        self.additional_list = additional_list
        self.temporal_len = temporal_len
        self.hidden_channel = hidden_channel


        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self.state_space+len(self.tech_indicator_list)+2*self.hidden_channel+1)) # cov matrix list + technical list + temporal feature * 60 + prediction labels + holding amount
        self.hidden_state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space, self.hidden_channel+1))

        self.data = self.df.loc[self.day, :]
        self.tic = self.df.tic.unique()
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # load model
        self.device = device
        self.short_prediction_model = self.load_model(short_prediction_model_path).to(self.device)
        self.long_prediction_model = self.load_model(long_prediction_model_path).to(self.device)
        self.short_prediction_model.eval()
        self.long_prediction_model.eval()

        # additional list
        self.print_additional_flag = print_additional_flag
        self.short_hidden_feature = []
        self.long_hidden_feature = []

        # initalize state and info
        self.info = self._initiate_info()
        self.state = self._initial_state()

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.amount_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        # self.reset()
        self._seed()


    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.info[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.info[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.info[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.info[index + 1]
                        * sell_num_shares
                        * (1 - self.transaction_cost_pct)
                    )
                    # update balance
                    self.info[0] += sell_amount

                    self.info[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.info[index + 1] * sell_num_shares * self.transaction_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.info[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.info[0] // self.info[index + 1]
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.info[index + 1] * buy_num_shares * (1 + self.transaction_cost_pct)
                )
                self.info[0] -= buy_amount

                self.info[index + self.stock_dim + 1] += buy_num_shares

                self.cost += self.info[index + 1] * buy_num_shares * self.transaction_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        buy_num_shares = _do_buy()

        return buy_num_shares


    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(self.figure_path+self.mode+"_account_value_trade_{}.png".format(self.episode))
        plt.close()

    def step(self, actions):
        if self.mode == 'train':
            self.terminal = (self.day - self.start_day) >= self.step_len + 1
        else:
            self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            self.end_total_asset = self.info[0] + sum(
                np.array(self.info[1 : (self.stock_dim + 1)])
                * np.array(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.end_total_asset
                - self.initial_amount
            )
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )

            self.reward = self.reward + self.reward_scaling * ((self.end_total_asset - self.initial_amount)/(self.initial_amount * 1.0))

            f1 = open(self.log_name, 'a')
            f1.write(str(self.end_total_asset)+'\t'+str(self.reward)+ '\t' + str(np.sum(self.rewards_memory)) + '\t' + str(sharpe) + '\t' + str((self.end_total_asset-self.initial_amount)/self.initial_amount) + '\n')
            f1.close()

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {self.end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    self.csv_path+"actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                df_stock_amount = self.save_holding_amount()
                df_stock_amount.to_csv(
                    self.csv_path+"amount_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    )
                )
                df_total_value.to_csv(
                    self.csv_path+"account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    self.csv_path+"account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    self.figure_path+"account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.episode
                    ),
                    index=False,
                )
                plt.close()

            return self.state, self.reward, self.terminal, {}

        else:

            # pdb.set_trace()
            actions = actions * self.hmax  # actions initially is scaled between 0?-1 to 1
            actions = actions.astype(
                int
            )
            begin_total_asset = self.info[0] + sum(
                np.array(self.info[1 : (self.stock_dim + 1)])
                * np.array(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.info = self._update_info()
            self.end_total_asset = self.info[0] + sum(
                np.array(self.info[1 : (self.stock_dim + 1)])
                * np.array(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            self.state = self._update_state()

            self.asset_memory.append(self.end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = ((self.end_total_asset - begin_total_asset)/(begin_total_asset*1.0))
            self.rewards_memory.append(self.reward)
            self.amount_memory.append(self.info[-self.stock_dim:])
            # self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):

        if self.mode == 'train':
            self.time_windows_point += 1
            self.start_day = self.time_window_start[self.time_windows_point]
        else:
            self.start_day = self.time_window_start[0]

        self.day = self.start_day
        self.data = self.df.loc[self.day, :]
        # self.covs = self.data['cov_list'].values[0]

        self.short_hidden_feature = []
        self.long_hidden_feature = []
        
        self.info = self._initiate_info()
        self.state = self._initial_state()

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.asset_memory = [self.initial_amount]
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.amount_memory = []#[self.info[-self.stock_dim:]]
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_info(self):
        # if len(self.df.tic.unique()) > 1:
        info = (
                    [self.initial_amount]
                    + self.data.price.values.tolist()
                    + [0] * self.stock_dim
            )
        return info

    def _initial_state(self):
        covs = np.array(self.data['cov_list'].values[0]) # (stock_dim, stock_dim)
        technical_indicators = np.array(self.data[self.tech_indicator_list].values.tolist()) # (stock_dim, len(technical_list))

        temporal_feature_data = self.df.loc[self.day-self.temporal_len+1:self.day, :]
        temporal_feature = np.array(temporal_feature_data[self.temporal_feature_list].values.tolist()).reshape(self.temporal_len, self.stock_dim, -1).transpose(1,0,2) # (num_nodes=bs, days, feature_list_len)
        enc_feature = torch.FloatTensor(temporal_feature).to(self.device)
        dec_feature = torch.FloatTensor(temporal_feature[:,-1:,:]).to(self.device)

        _, hidden_short, _ = self.short_prediction_model(enc_feature, dec_feature)
        _, hidden_long, _ = self.long_prediction_model(enc_feature, dec_feature)


        hidden_np1 = hidden_short.detach().cpu().numpy().reshape(self.stock_dim, -1)
        hidden_np2 = hidden_long.detach().cpu().numpy().reshape(self.stock_dim, -1)

        self.short_hidden_feature.append(hidden_np1)
        self.long_hidden_feature.append(hidden_np2)

        # pdb.set_trace()
        holding_amount = np.zeros((self.stock_dim,1), dtype=int)
        state = np.concatenate((covs, technical_indicators, hidden_np1, hidden_np2, holding_amount), axis=-1)
        # print("Initial: ",state.shape)
        return state


    def _update_info(self):
            # for multiple stock
        info = (
                [self.info[0]]
                + self.data.price.values.tolist()
                + list(self.info[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
        return info

    def _update_state(self):
        covs = np.array(self.data['cov_list'].values[0]) # (stock_dim, stock_dim)
        technical_indicators = np.array(self.data[self.tech_indicator_list].values.tolist()) # (stock_dim, len(technical_list))

        temporal_feature_data = self.df.loc[self.day-self.temporal_len+1:self.day, :]
        temporal_feature = np.array(temporal_feature_data[self.temporal_feature_list].values.tolist()).reshape(self.temporal_len, self.stock_dim, -1).transpose(1,0,2) # (num_nodes, temporal_day, feature_list_len)

        enc_feature = torch.FloatTensor(temporal_feature).to(self.device)
        dec_feature = torch.FloatTensor(temporal_feature[:,-1:,:]).to(self.device)

        _, hidden_short, _ = self.short_prediction_model(enc_feature, dec_feature)
        _, hidden_long, _ = self.long_prediction_model(enc_feature, dec_feature)


        hidden_np1 = hidden_short.detach().cpu().numpy().reshape(self.stock_dim, -1)
        hidden_np2 = hidden_long.detach().cpu().numpy().reshape(self.stock_dim, -1)

        self.short_hidden_feature.append(hidden_np1)
        self.long_hidden_feature.append(hidden_np2)
        
        holding_amount = np.array(self.info[-self.stock_dim : ]) # (stock_dim, 1)
        holding_amount_norm = ((holding_amount * np.array(self.info[1: 1+self.stock_dim]))/self.end_total_asset).reshape(self.stock_dim, 1)

        state = np.concatenate((covs, technical_indicators, hidden_np1, hidden_np2, holding_amount_norm), axis=-1)
        # print("Update: ",state.shape)
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_additional_info(self):
        temp_dict = {"short_hidden_feature":self.short_hidden_feature, "long_hidden_feature": self.long_hidden_feature}
        return temp_dict

    def save_holding_amount(self):
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        amount_list = self.amount_memory
        df_amount = pd.DataFrame(amount_list)
        df_amount.columns = self.data.tic.values
        return df_amount


    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def load_model(self, path, enc_in=10, dec_in=10, c_out=1):
        model = PredictionModel(enc_in=enc_in, dec_in=dec_in, c_out=c_out)

        if path is not None:
            state_dict = torch.load(path, map_location='cuda:0')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items(): 
                name = k[7:] 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print("Successfully load prediction mode...", path)
        
        return model
