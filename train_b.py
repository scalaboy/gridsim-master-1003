#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:06:52 2021

@author: deeplp
"""

# -*- coding: UTF-8 -*-

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings
import numpy as np

import pylab
import time
import gym
import tensorflow as tf

#import tensorflow.compat.v2 as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam
#from keras import optimizer_v1
#from keras import backend as K
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tfc
from tensorflow.compat.v1.keras import backend as fk

#from agent import Agent

scores = []
EPISODES_train = 5

class Agent(object):
    def __init__(self, settings, this_directory_path):
        self.num_gen = settings.num_gen
        self.state_size = 91+settings.num_gen
        self.action_size = settings.num_gen
        # get gym environment name
        #self.env_name = env_name
        # these are hyper parameters for the A3C

        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.discount_factor = .95
        self.hidden1, self.hidden2 = 200, 50

        #self.threads = num_threads # 48 or 16 or 32 - corresponds to parallel agents

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()
        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        
        #self.sess = tfc.InteractiveSession()
        #fk.set_session(self.sess)
        #self.sess.run(tfc.global_variables_initializer())

    # approximate policy and value using Neural Network

    # actor -> state is input and probability of each action is output of network

    # critic -> state is input and value of state is output of network

    # actor and critic network share first hidden layer

    def build_model(self):

        state = Input(batch_shape=(None,  self.state_size))

        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(actor_hidden)
        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)
        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)
        #actor.
        actor.make_predict_function()
        critic.make_predict_function()
        actor.summary()
        critic.summary()
        return actor, critic



    # make loss function for Policy Gradient

    # [log(action probability) * advantages] will be input for the back prop

    # we add entropy of action probability to loss
    """
    def actor_optimizer(self):

        action = K.placeholder(shape=(None, self.action_size))

        advantages = K.placeholder(shape=(None, ))
        policy = self.actor.output
        good_prob = K.sum(action * policy, axis=1)

        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        actor_loss = loss + 0.01*entropy
        #optimizer = Adam(lr=self.actor_lr)
        optimizer = Adam(lr=self.actor_lr)
        #updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        #train = K.function([self.actor.input, action, advantages], [], updates=updates)
       #K.track_tf_optimizer(optimizer)
        updates = optimizer.get_updates( actor_loss,self.actor.trainable_weights)
        train = K.function(self.actor.input, action)
        return train
    """

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)

        actor_loss = loss + 0.01*entropy

        optimizer = Adam(lr=self.actor_lr)
        #updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        updates = optimizer.get_updates(params=self.actor.trainable_weights, loss=actor_loss )
        #train = K.function([self.actor.input, action, advantages],[], updates=updates)
        train = K.function(self.actor.input,[], updates=updates)
        return train


    # make loss function for Value approximation
    """
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))
        value = self.critic.output
        loss = K.mean(K.square(discounted_reward - value))
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(loss,self.critic.trainable_weights)
        #function(inputs, outputs, updates=None, name=None, **kwargs)
        train = K.function(self.critic.input, discounted_reward)
        return train
    """
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        #updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        updates = optimizer.get_updates(params=self.critic.trainable_weights, loss=loss )
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train


    # make agents(local) and start training


    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")
   
    def get_act(self, obs, reward, done=False):
        adjust_gen_p_action_space = obs.action_space['adjust_gen_p']
        adjust_gen_v_action_space = obs.action_space['adjust_gen_v']
        adjust_gen_p = adjust_gen_p_action_space.sample()
        adjust_gen_v = adjust_gen_v_action_space.sample()
        
        action = {'adjust_gen_p': adjust_gen_p, 'adjust_gen_v': adjust_gen_v}
        return action    
    def get_action(self, obs, reward, done=False):
        input_tensor1 = obs.gen_v
        input_tensor2 = obs.load_p
        input_tensor = np.append(input_tensor1,input_tensor2)
        policy_nn = self.actor.predict(np.reshape(input_tensor, [1, self.state_size]))[0]
        print('action is ',policy_nn)
        adjust_gen_v_action_space = obs.action_space['adjust_gen_v']
        adjust_gen_v = adjust_gen_v_action_space.sample()
        action = {'adjust_gen_p': policy_nn, 'adjust_gen_v': adjust_gen_v}
        return action
        '''
        policy_nn_subid_mask = policy_nn * (1 - valid_actions_masking_subid_perm.dot((state[-14:]>0).astype(int))) # this masking prevents any illegal operation
        policy_chosen_list = np.random.choice(self.action_size, 4, replace=True,
                                              p=policy_nn_subid_mask / sum(policy_nn_subid_mask))
        policy_chosen_list = np.hstack((0, policy_chosen_list)) # adding no action option # comment this line as agent learns...
        obs_0, rw_0, done_0, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[0],:])
        obs_1, rw_1, done_1, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[1],:])
        obs_2, rw_2, done_2, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[2],:])
        obs_3, rw_3, done_3, _  = env.simulate(valid_actions_array_uniq[policy_chosen_list[3],:])
        rw_0 = self.est_reward_update(obs_0, rw_0, done_0)
        rw_1 = self.est_reward_update(obs_1, rw_1, done_1)
        rw_2 = self.est_reward_update(obs_2, rw_2, done_2)
        rw_3 = self.est_reward_update(obs_3, rw_3, done_3)
        return policy_chosen_list[np.argmax([rw_0,rw_1,rw_2,rw_3])]
        '''
    def train_episode(self, done):
            discounted_rewards = self.discount_rewards(self.rewards, done)
    
            values = self.critic.predict(np.array(self.states))
            values = np.reshape(values, len(values))
    
            advantages = discounted_rewards - values
    
            self.optimizer[0]([self.states, self.actions, advantages])
            self.optimizer[1]([self.states, discounted_rewards])
            self.states, self.actions, self.rewards = [], [], []    
    
    def train(self):
        try:
            self.load_model('pypow_14_a3c')
            print("Loaded saved NN model parameters \n")
        except:
            print("No existing model is found or saved model sizes do not match - initializing random NN weights \n")
        if (len(scores) < EPISODES_train ):
            print("Begin to train a Model\n")
            time.sleep(2) # main thread saves the model every 200 sec
            if (len(scores)>10):
                self.save_model('pypow_14_a3c')
                print("saved NN model at episode", episode, "\n")
        max_timestep = 9  # 最大时间步数
        max_episode = 9
        for episode in range(max_episode):    
            print('------ episode ', episode)  
            env = Environment(settings, "EPRIReward")    
            print('------ reset ')   
            obs = env.reset()
            #obs.    
            time_hour = 0
            score = 0
            time_step = 0
            non_zero_actions = 0
            reward = 0.0
            done = False
            # while not done:
            for timestep in range(max_timestep):
                action = self.get_action(obs, reward, done)
                adjust_gen_p_action_space = obs.action_space['adjust_gen_p']
                if(not adjust_gen_p_action_space.contains(action['adjust_gen_p'])):
                    score = -99999
                    #go to train---------------- update parames
                #action = self.get_action(obs, reward, done)
                #action = my_agent.act(self, obs, reward, done)
                print("adjust_gen_p: ", action['adjust_gen_p'])
                print("adjust_gen_v: ", action['adjust_gen_v'])
                obs, reward, done, info = env.step(action)
                #state = env.reset()

                #state_obs = observation_space.array_to_observation(state)

                #state = self.useful_state(state)

                next_state, reward, done, flag = env.step(valid_actions_array_uniq[action,:])

                if done:
                    score += -1000 # this is the penalty for grid failure.

                    self.memory(state, action, -1000)
                else:
                    state_obs = observation_space.array_to_observation(next_state)
                    time_hour = state_obs.date_day*10000 + state_obs.date_hour * 100+ state_obs.date_minute

                    current_lim_factor = 0.85

                    over_current = 50 * sum(((state_obs.ampere_flows - current_lim_factor * state_obs.thermal_limits ) / (state_obs.thermal_limits))[

                        state_obs.ampere_flows > current_lim_factor * state_obs.thermal_limits]) # # penalizing lines close to the limit

                    score += (reward-over_current)

                    self.memory(state, action, (reward - over_current))

                non_zero_actions += 0 if action==0 else 1

                state = self.useful_state(next_state) if not done else np.zeros([1, state_size])

                time_step += 1

                if time_step % training_batch_size ==0:

                    print("Continue Thread:", self.index, "/ train episode: ", episode, "/ score : ", int(score),

                          "/ with recent time:", time_step, "/ with recent action", action,"/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)

                    self.train_episode(score < 2000000) # max score = 80000

                if done or time_step > time_step_end:

                    if done:

                        print("----STOPPED Thread:", self.index, "/ train episode: ", episode, "/ score : ", int(score),

                              "/ with final time:", time_step, "/ with final action", action,

                              "/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)

                    if time_step > time_step_end:

                        print("End Thread:", self.index, "/ train episode: ", episode, "/ score : ", int(score),

                              "/ with final time:", time_step, "/ with final action", action,

                              "/ number of non-zero actions", non_zero_actions, "/ day_hour_min:", time_hour)

                    scores.append(score)
                    episode += 1
                    self.train_episode(score < 2000000) # max score = 80000
                    break    


if __name__ == "__main__":
    max_timestep = 9990  # 最大时间步数
    max_episode = 99  # 回合数
    #my_agent = Agent(settings,'./model_sv')

    #my_agent=DoNothingAgent(settings.num_gen)
    #run_task(my_agent)
    
    my_agent=Agent(settings,'./model_sv')
    my_agent.train()

