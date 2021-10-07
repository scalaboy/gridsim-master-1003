#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:54:25 2021

@author: deeplp
"""

import os, sys, random
from itertools import count 
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import numpy as np

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings

TAU = 0.005
LR = 1e-3
GAMMA = 0.99
MEMORY_CAPACITY = 3
BATCH_SIZE = 64
MAX_EPISODE = 100000
MODE = 'train' # or 'test'

sample_frequency = 256
log_interval = 50
render_interval = 100
exploration_noise = 0.1
max_length_of_trajectory = 2000
target_update_interval = 1
test_iteration = 10
update_iteration = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'


min_Val = torch.tensor(1e-7).float().to(device)

directory = './runs'

class Replay_buffer():
    def __init__(self,max_size=MEMORY_CAPACITY):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self,data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr+1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self,batch_size):
        ind = np.random.randint(0,len(self.storage),size=batch_size)
        x,x1,x2,y,u,r,d = [],[],[],[],[],[],[]

        for i in ind:
            X,X1,X2,Y,U,R,D = self.storage[i]
            x.append(np.array(X,copy=False))
            x1.append(np.array(X1,copy=False))
            x2.append(np.array(X2,copy=False))
            y.append(np.array(Y,copy=False))
            u.append(np.array(U,copy=False))
            r.append(np.array(R,copy=False))
            d.append(np.array(D,copy=False))
        return np.array(x),np.array(x1),np.array(x2),np.array(y),np.array(u),np.array(r),np.array(d)
    
class Actor(nn.Module):
    """docstring for Actor"""
    def __init__(self, state_dim,action_dim,max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim,400)
        self.l2 = nn.Linear(400,300)
        self.l3 = nn.Linear(300,action_dim)
        self.max_action = max_action

    def forward(self,x,min_action,gap_action):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        #x = self.max_action + torch.tanh(self.l3(x))
        #x=torch.sigmoid(self.l3(x))
        x = min_action+gap_action*torch.sigmoid(self.l3(x))
        return x

class Critic(nn.Module):
    """docstring for Critic"""
    def __init__(self, state_dim,action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim+action_dim,400)
        self.l2 = nn.Linear(400,300)
        self.l3 = nn.Linear(300,1)
        
    def forward(self,x,u):
        x = F.relu(self.l1(torch.cat([x,u],1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    """docstring for DDPG"""
    def __init__(self, state_dim,action_dim,max_action):
        super(DDPG, self).__init__()
        
        self.actor = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(),LR)

        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),LR)

        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self,input_tensor3,min_a,gap_a):
        #state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        #state = torch.FloatTensor(state).to(device)
        min_a=  torch.FloatTensor(min_a).to(device)
        gap_a=  torch.FloatTensor(gap_a).to(device)
        #input_tensor2 = np.append(state,min_a)
        #input_tensor3 = np.append(input_tensor2,gap_a)
        input_tensor3=  torch.FloatTensor(input_tensor3).to(device)
        #gap_a = torch.FloatTensor(gap_a).to(device)
        #min_a=  torch.FloatTensor(min_a).to(device)
        return self.actor(input_tensor3,min_a,gap_a).cpu().data.numpy().flatten()

    def update(self):
        for it in range(update_iteration):
            # sample replay buffer
            x,x1,x2,y,u,r,d = self.replay_buffer.sample(BATCH_SIZE)
            #print('x,y,u,r,d',x,y,u,r,d)
            state = torch.FloatTensor(x).to(device)
            min_a = torch.FloatTensor(x1).to(device)
            gap_a = torch.FloatTensor(x2).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # compute the target Q value
            target_Q = self.critic_target(next_state,self.actor_target(next_state,min_a,gap_a))
            target_Q = reward + ((1-done)*GAMMA*target_Q).detach()

            # get current Q estimate
            current_Q = self.critic(state,action)

            # compute critic loss
            critic_loss = F.mse_loss(current_Q,target_Q)
            self.writer.add_scalar('Loss/critic_loss',critic_loss,global_step=self.num_critic_update_iteration)

            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute actor loss
            #max_a = state.action_space['adjust_gen_p'].high
            #min_a = state.action_space['adjust_gen_p'].low
            actor_loss = - self.critic(state,self.actor(state,min_a,gap_a)).mean()
            self.writer.add_scalar('Loss/actor_loss',actor_loss,global_step=self.num_actor_update_iteration)

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the frozen target models
            for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

            for param,target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data) 

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1


    def save(self):
        torch.save(self.actor.state_dict(),directory+'actor.pth')
        torch.save(self.critic.state_dict(),directory+'critic.pth')
        print('model has been saved...')

    def load(self):
        self.actor.load_state_dict(torch.load(directory+'actor.pth'))
        self.critic.load_state_dict(torch.load(directory+'critic.pth'))
        print('model has been loaded...')


def inf_shot(x):
    if(x<-9999):
        return -9999
    elif(x>9999):
        return 9999
    else:
        return x

def main():
    #env = gym.make('Pendulum-v0').unwrapped
    env = Environment(settings, "EPRIReward")
    print('------ reset ')
    state = env.reset()
    #state_dim = 3*settings.num_gen
    state_dim = settings.num_gen
    action_dim = settings.num_gen
    #max_action = float(state.action_space['adjust_gen_p'].high)
    ep_r = 0
    if MODE == 'test':
        
        for i in range(test_iteration):
            state = env.reset()
            max_a = state.action_space['adjust_gen_p'].high
            min_a = state.action_space['adjust_gen_p'].low
            max_action = (max_a+min_a)*0.1
            max_action, = map (torch.tensor, (max_action, ))
            agent = DDPG(state_dim,action_dim,max_action)
            agent.load()
            for t in count():
                action = agent.select_action(state)
                print(action)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t>=max_length_of_trajectory:
                    print('Episode:{}, Return:{:0.2f}, Step:{}'.format(i,ep_r,t))
                    ep_r = 0
                    break
                state = next_state

    elif MODE == 'train':
        print('Collection Experience...')
        state = env.reset()
        max_a = state.action_space['adjust_gen_p'].high
        min_a = state.action_space['adjust_gen_p'].low
        max_action = (max_a+min_a)*0.1
        max_action, = map (torch.tensor, (max_action, ))
        agent = DDPG(state_dim,action_dim,max_action)
        agent.load()
        for i in range(MAX_EPISODE):
            state = env.reset()
            max_a = state.action_space['adjust_gen_p'].high
            min_a = state.action_space['adjust_gen_p'].low
            max_action = (max_a+min_a)*0.1
            max_action, = map (torch.tensor, (max_action, ))
            agent = DDPG(state_dim,action_dim,max_action)
            #agent.load()
            for t in count():
                max_a = state.action_space['adjust_gen_p'].high
                min_a = state.action_space['adjust_gen_p'].low
                gap_a = max_a - min_a
                gap_a = np.array(list( map(inf_shot,gap_a))).astype('float32')
                min_a = np.array(list( map(inf_shot,min_a))).astype('float32')
                input_tensor1 = state.gen_v
                #input_tensor2 = obs.load_p
                 
                #input_tensor3 = np.append(input_tensor1,min_a,max_a)
                input_tensor2 = np.append(input_tensor1,min_a).astype('float32')
                input_tensor3 = np.append(input_tensor2,max_a)
                input_tensor3 = input_tensor3.astype('float32')
                #print(min_a,max_a,input_tensor3,)
                input_tensor3, = map (torch.tensor, (input_tensor3, ))
                action = agent.select_action(input_tensor1,min_a,gap_a)
                #print('action is , info' ,action,'\n')
                # issue 3 add noise to action
                #if(action )

                #action = (action + np.random.normal(0,exploration_noise,size=env.action_space.shape[0])).clip(env.action_space.low,env.action_space.high)
                adjust_gen_v_action_space = state.action_space['adjust_gen_v']
                adjust_gen_p_action_space=state.action_space['adjust_gen_p']
                adjust_gen_v = adjust_gen_v_action_space.sample()
                adjust_gen_p = adjust_gen_p_action_space.sample()
                if(random.randint(0,10) >4):
                    action = adjust_gen_p
                action_step = {'adjust_gen_p': action, 'adjust_gen_v': adjust_gen_v}
                #obs, reward, done, info = env.step(action)
                if(not adjust_gen_p_action_space.contains(action_step['adjust_gen_p'])):
                    print('errororrrrrrrrrrrrrrrrrrrrrrrrrrrrr \n')
                    reward = -99999
                    next_state = state
                    done = False
                else:
                    next_state, reward, done, info = env.step(action_step)
                    print('next iter reward,done, info',reward,done, info)
                    if(done==True):
                        reward = -999
                    if(reward >0):
                        reward = 99*reward
                ep_r += reward
                agent.replay_buffer.push((input_tensor1,min_a,gap_a,next_state.gen_v,action,reward,np.float(done)))
                
                state = next_state
                if done or t>=max_length_of_trajectory:
                    agent.writer.add_scalar('ep_r',ep_r,global_step=i)
                    if i % 10 ==0:
                        print('Episode:{}, Return:{:0.2f}, Step:{}'.format(i,ep_r,t))
                    ep_r = 0
                    break

            if (i+1) % 100 == 0:
                print('Episode:{}, Memory size:{}'.format(i,len(agent.replay_buffer.storage)))

            if i % log_interval == 0:
                agent.save()

            if len(agent.replay_buffer.storage) >= MEMORY_CAPACITY-1:
                agent.update()

    else:
        raise NameError('model is wrong!!!')

if __name__ == '__main__':
    main()