from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
import random

import train
import buffer
import os
from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


MAX_EPISODES = 25000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300



ram = buffer.MemoryBuffer(MAX_BUFFER)

#trainer.load_models(1900)
def inf_shot(x):
    if(x<-9999):
        return -9999
    elif(x>9999):
        return 9999
    else:
        return x
def main():
    env = Environment(settings, "EPRIReward")
    state = env.reset()
    max_a = state.action_space['adjust_gen_p'].high
    state_dim = settings.num_gen+len(state.load_p)*2+len(state.rho)+ \
    len(state.a_or) + len(state.a_ex) +len(state.count_soft_overflow_steps) +\
    len(state.curstep_renewable_gen_p_max) + len(state.nextstep_renewable_gen_p_max) +\
    len(settings.max_gen_p) + len(settings.min_gen_p) + len(settings.max_gen_v) + len(settings.min_gen_v)+\
    len(state.target_dispatch) + len(state.actual_dispatch)
    action_dim = 2*settings.num_gen
    #action_dim = settings.num_gen
    trainer = train.Trainer(state_dim, action_dim, max_a, ram)
    #trainer.load_models(2400)
    for _ep in range(MAX_EPISODES):
        	state = env.reset()
        	print ('EPISODE :- ', _ep)
        	for r in range(MAX_STEPS):
        		#env.render()
        		#state = np.float32(observation)
        		max_a = state.action_space['adjust_gen_p'].high
        		min_a = state.action_space['adjust_gen_p'].low
        		gap_a = (max_a - min_a)*0.98
        		max_v = state.action_space['adjust_gen_v'].high
        		min_v = state.action_space['adjust_gen_v'].low
        		gap_v = (max_v - min_v)*0.98                
        		gap_a = np.array(list( map(inf_shot,gap_a))).astype('float32')
        		min_a = np.array(list( map(inf_shot,min_a))).astype('float32')
        		gap_v = np.array(list( map(inf_shot,gap_v))).astype('float32')
        		min_v = np.array(list( map(inf_shot,min_v))).astype('float32')
        		state_gen_v = np.array(state.gen_v).astype('double')
        		load_p = np.array(state.load_p).astype('double')
        		nextstep_load_p =  np.array(state.nextstep_load_p).astype('double')
        		rho = np.array(state.rho).astype('double')
        		a_or = np.array(state.a_or).astype('double')
        		a_ex =  np.array(state.a_ex).astype('double')
        		count_soft_overflow_steps = np.array(state.count_soft_overflow_steps).astype('double')
        		curstep_renewable_gen_p_max = np.array(state.curstep_renewable_gen_p_max).astype('double')
        		nextstep_renewable_gen_p_max =  np.array(state.nextstep_renewable_gen_p_max).astype('double')
        		max_gen_p =  np.array(settings.max_gen_p).astype('double')
        		min_gen_p = np.array(settings.min_gen_p).astype('double')
        		max_gen_v = np.array(settings.max_gen_v).astype('double')
        		min_gen_v =  np.array(settings.min_gen_v).astype('double')
        		target_dispatch = np.array(state.target_dispatch).astype('double')
        		actual_dispatch =  np.array(state.actual_dispatch).astype('double')
        		#input_tensor2 = np.append(state_gen_v,load_p)               
        		#input_tensor4 = np.append(state_gen_v,load_p)   
        		input_tensor9 = np.concatenate((state_gen_v,load_p,nextstep_load_p,rho,a_or,a_ex,count_soft_overflow_steps,\
            curstep_renewable_gen_p_max,nextstep_renewable_gen_p_max,max_gen_p,min_gen_p,max_gen_v,min_gen_v,\
            target_dispatch,actual_dispatch),axis=0)   
        		#input_tensor8 = np.append(state_gen_v,load_p)   
        		#input_tensor3 = np.append(input_tensor3,nextstep_load_p)                  
        		#action = trainer.get_exploration_action(state)
        		adjust_gen_v_action_space = state.action_space['adjust_gen_v']
        		adjust_gen_p_action_space=state.action_space['adjust_gen_p']
        		adjust_gen_v = adjust_gen_v_action_space.sample()
        		adjust_gen_p = adjust_gen_p_action_space.sample()   
        		random1 = random.randint(0,10)
        		if(random1 >-18):
        		# 	# validate every 5th episode
        		 	action = trainer.get_exploitation_action(input_tensor9,min_a,gap_a,min_v,gap_v)
        		else:
        		# 	# get action based on observation, use exploration policy here
        		 	print('in get_exploration_action=============')
        		 	#action = trainer.get_exploration_action(state_gen_v,min_a,gap_a)
        		 	action = adjust_gen_p
        		#@if(random.randint(0,10) >4):
        		 #   action = adjust_gen_p
        		action_step = {'adjust_gen_p': action[0:settings.num_gen], 'adjust_gen_v': action[settings.num_gen:2*settings.num_gen]}   
        		if(not adjust_gen_p_action_space.contains(action_step['adjust_gen_p'])):
        		     print('errororrrrrrrrrrrrrrrrrrrrrrrrrrrrr \n',random1,action)
                     
        		     reward = -99999                
        		new_observation, reward, done, info = env.step(action_step)
        		if(reward>0):
        		   print('rrrrrrrrrrrrrrrrrrrrrrreward',reward)
        		   reward = reward*99   
        
        		# # dont update if this is validation
        		# if _ep%50 == 0 or _ep>450:
        		# 	continue
        
        		if done:
        			new_state_gen_v = None
        			print('in train_main r count is ',r,reward, done, info)
        			reward = -999   
        			ram.add(input_tensor9,min_a,gap_a,min_v,gap_v, action, reward, input_tensor9)
        			trainer.optimize()                    
        			break
        		else:
        			new_state_gen_v = np.float32(new_observation.gen_v)
        			new_load_p = np.array(new_observation.load_p).astype('double')
        			new_nextstep_load_p =  np.array(new_observation.nextstep_load_p).astype('double')
        			new_rho = np.array(new_observation.rho).astype('double')
        			new_a_or = np.array(new_observation.a_or).astype('double')
        			new_a_ex =  np.array(new_observation.a_ex).astype('double')
        			new_count_soft_overflow_steps = np.array(new_observation.count_soft_overflow_steps).astype('double')
        			new_curstep_renewable_gen_p_max = np.array(new_observation.curstep_renewable_gen_p_max).astype('double')
        			new_nextstep_renewable_gen_p_max =  np.array(new_observation.nextstep_renewable_gen_p_max).astype('double')
        			new_max_gen_p =  np.array(settings.max_gen_p).astype('double')
        			new_min_gen_p = np.array(settings.min_gen_p).astype('double')
        			new_max_gen_v = np.array(settings.max_gen_v).astype('double')
        			new_min_gen_v =  np.array(settings.min_gen_v).astype('double')
        			new_target_dispatch = np.array(new_observation.target_dispatch).astype('double')
        			new_actual_dispatch =  np.array(new_observation.actual_dispatch).astype('double')
            		#input_tensor2 = np.append(state_gen_v,load_p)               
            		#input_tensor4 = np.append(state_gen_v,load_p)   
        			new_input_tensor9 = np.concatenate((new_state_gen_v,new_load_p,new_nextstep_load_p,new_rho,new_a_or,new_a_ex,new_count_soft_overflow_steps,\
                     new_curstep_renewable_gen_p_max,new_nextstep_renewable_gen_p_max,\
                    new_max_gen_p,new_min_gen_p, new_max_gen_v,new_min_gen_v,new_target_dispatch,new_actual_dispatch),axis=0)                
        			# push this exp in ram
        			ram.add(input_tensor9,min_a,gap_a,min_v,gap_v, action, reward, new_input_tensor9)
        			trainer.optimize()
        		state = new_observation
        
        		# perform optimization
        		
        		if done:
        			break
        
        	# check memory consumption and clear memory
        	gc.collect()
        	# process = psutil.Process(os.getpid())
        	# print(process.memory_info().rss)
        
        	if _ep%100 == 0 and _ep >9:
        		trainer.save_models(_ep)
    
    
        	print ('Completed episodes')

if __name__ =='__main__':
    main()
