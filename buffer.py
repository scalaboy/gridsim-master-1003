import numpy as np
import random
from collections import deque


class MemoryBuffer:

	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)

		s_arr = np.float32([arr[0] for arr in batch])
		min_arr = np.float32([arr[1] for arr in batch])
		gap_arr = np.float32([arr[2] for arr in batch])
		min_v = np.float32([arr[3] for arr in batch])
		gap_v = np.float32([arr[4] for arr in batch])        
		a_arr = np.float32([arr[5] for arr in batch])
		r_arr = np.float32([arr[6] for arr in batch])
		s1_arr = np.float32([arr[7] for arr in batch])

		return s_arr,min_arr,gap_arr,min_v,gap_v, a_arr, r_arr, s1_arr

	def len(self):
		return self.len

	def add(self, s,min_a,gap_a,min_v,gap_v, a, r, s1):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""
		transition = (s,min_a,gap_a,min_v,gap_v,a,r,s1)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)
