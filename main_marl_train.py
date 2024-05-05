import random
import scipy
import scipy.io
import numpy as np
import environment as env
import os
import sys
from collections import deque

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, BatchNormalization, Dense
from keras.optimizers import RMSprop, Adam
from keras.metrics import mse
from keras.callbacks import LearningRateScheduler
import keras.backend as K

# Cấu hình tùy chọn bộ nhớ GPU
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# REPLAY_MEMORY_SIZE = 5_000_000
# MIN_REPLAY_MEMORY_SIZE = 200_000
# MINIBATCH_SIZE = 200_000

REPLAY_MEMORY_SIZE = 40_000
MIN_REPLAY_MEMORY_SIZE = 5000
MINIBATCH_SIZE = 2000
num_choose = np.ones((6,))

class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 0.99
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

# ################## SETTINGS ######################

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'marl_model'

env.new_random_game()  # initialize parameters in env

n_episode = 340
n_step_per_episode = 200 
epsi_final = 0.02
epsi_anneal_length = int(270)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes

action_latest = np.zeros((env.group_e + env.group_u, 1))
reward_latest = 0

######################################################

def get_state(env, idx=0):
    """ Get state from the environment """
    # global prev_actions, prev_rewards

    h = env.h *10
    # print(h)
    # a = prev_actions[idx]
    # b = prev_rewards[idx]
    c = num_choose / 10
    b = reward_latest / 260 
    a = action_latest / 10
    # print(a)
    # print(b)
    # print(c)
    # print(len(np.concatenate((np.array([h])), np.array([action_latest]), np.array([train_reward_latest]))))
    # return np.concatenate((np.array([h]).flatten()))
    return np.concatenate(( np.array([h]).flatten(), np.array([a]).flatten(), np.array([b]), np.array([c]).flatten() ))       

# -----------------------------------------------------------
n_hidden_1 = 512
n_hidden_2 = 256
n_hidden_3 = 128
n_input = len(get_state(env=env))
n_output = 6

# ============== Training network ========================
def create_model(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
    model = Sequential([
        Dense(units= n_hidden_1, input_shape=(n_input, ), activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1), bias_initializer=keras.initializers.TruncatedNormal(stddev=0.1)),
        BatchNormalization(),
        Dense(units= n_hidden_2, activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1), bias_initializer=keras.initializers.TruncatedNormal(stddev=0.1)),
        BatchNormalization(),
        Dense(units= n_hidden_3, activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1), bias_initializer=keras.initializers.TruncatedNormal(stddev=0.1)),
        BatchNormalization(),
        Dense(units= n_output, kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1), bias_initializer=keras.initializers.TruncatedNormal(stddev=0.1))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])  
    return model

def predict(sess, s_t, ep, test_ep = False):

    state = np.array(s_t).reshape(-1, *s_t.shape)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(6)
    else:
        pred_actions = sess.predict(state)[0]
        pred_action = np.argmax(pred_actions)
    return pred_action

def q_learning_mini_batch(current_agent, current_sess, current_sess_p):
    """ Training a sampled mini-batch """
    
    if len(current_agent.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
        return
    
    minibatch = random.sample(current_agent.replay_memory, MINIBATCH_SIZE)
    
    if current_agent.double_q:  # double q-learning

        # current_states = np.array([transition[0] for transition in minibatch])
        current_states = np.array([transition[0] for transition in minibatch])
        current_states = tf.convert_to_tensor(current_states)
        current_qs_list_pre = current_sess.predict(current_states)
        
        # future_states = np.array([transition[1] for transition in minibatch])
        future_states = np.array([transition[1] for transition in minibatch])
        future_qs_list_pre = current_sess.predict(future_states)
        future_qs_list_pre_max_index = np.argmax(future_qs_list_pre, axis=1)

        future_qs_list_tar = current_sess_p.predict(future_states) 

        X = []
        Y = []

        for index, (batch_s_t, batch_s_t_plus_1, batch_action, batch_reward) in enumerate(minibatch):

            new_q = batch_reward + current_agent.discount * future_qs_list_tar[index, future_qs_list_pre_max_index[index]]
            # print(future_qs_list_pre_max_index[index])

            current_qs_pre = current_qs_list_pre[index]
            current_qs_pre[batch_action] = new_q

            X.append(batch_s_t)
            Y.append(current_qs_pre)

    history = current_sess.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=1, shuffle=False)
    loss_values = history.history['loss']
    return loss_values

def update_target_q_network(sess, sess_p):
    """ Update target q network once in a while """
    
    sess_p.set_weights(sess.get_weights())

# --------------------------------------------------------------
agents = []
sesses = []
sesses_p = []

for ind_agent in range(env.group_e + env.group_u):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)
    
    sess = create_model(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    sesses.append(sess)
    sess_p = create_model(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output)
    sesses_p.append(sess_p)

# ------------------------- Training -----------------------------
record_reward = np.zeros([n_episode*n_step_per_episode, 1])
record_loss = []
if IS_TRAIN:
    if os.path.exists("Reward_per_ep.txt"):
        os.remove("Reward_per_ep.txt")
    for i_episode in range(n_episode):
        if i_episode == 80:
            new_learning_rate = 0.0001
            for sess in sesses:
                K.set_value(sess.optimizer.lr, new_learning_rate)
        elif i_episode == 250:
            new_learning_rate = 0.00005
            for sess in sesses:
                K.set_value(sess.optimizer.lr, new_learning_rate)
        num_choose = np.ones((6,))
        Reward_per_ep = 0
        utility_average = 0
        eBMM_utility_average = 0
        URRLC_utility_average = 0
        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < np.ceil(MIN_REPLAY_MEMORY_SIZE / n_step_per_episode):
            epsi = 1
        elif (i_episode - np.ceil(MIN_REPLAY_MEMORY_SIZE / n_step_per_episode)) < epsi_anneal_length:
            epsi = 1 - (i_episode - np.ceil(MIN_REPLAY_MEMORY_SIZE / n_step_per_episode)) * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final
        # print(epsi)

        for i_step in range(n_step_per_episode):
            time_step = i_episode*n_step_per_episode + i_step
            state_old_all = []
            action_all = []
            action_all_training = np.zeros([env.group_e + env.group_u, 1], dtype='int8')
            for i in range(env.group_e + env.group_u):
                state = get_state(env, i)
                state_old_all.append(state)
                action = predict(sesses[i], state, epsi)
                action_all.append(action)

                action_all_training[i, 0] = action  # chosen BS

            # print(state_old_all)
            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward, num_choose, utility, eBMM_utility, URRLC_utility = env.act_for_training(action_temp)
            record_reward[time_step] = train_reward
            action_latest = action_temp
            reward_latest = train_reward
            utility_average += utility
            eBMM_utility_average += eBMM_utility
            URRLC_utility_average += URRLC_utility

            Reward_per_ep += train_reward

            env.update_slow_fading()
            env.calculate_power_eMBB()
            env.calculate_power_URRLC()

            for i in range(env.group_e + env.group_u):
                
                state_old = state_old_all[i]
                action = action_all[i]
                state_new = get_state(env, i)
                agents[i].update_replay_memory((state_old, state_new, action, train_reward))
                # prev_actions[i].append(action)
                # prev_rewards[i].append(train_reward)

                # training this agent
                if time_step % mini_batch_step == mini_batch_step-1:
                    loss_val_batch = q_learning_mini_batch(agents[i], sesses[i], sesses_p[i])
                if time_step % target_update_step == target_update_step-1:
                    update_target_q_network(sesses[i], sesses_p[i])
                    if i == 0:
                        print('Update target Q network...')
                            
        with open("Reward_per_ep.txt","a") as f:
            f.write(str(i_episode) + ', ' + str(utility_average / n_step_per_episode) + ', ' + str(eBMM_utility_average / n_step_per_episode) + ', ' + str(URRLC_utility_average / n_step_per_episode) +'\n' )
            
    print('Training Done. Saving models...')





