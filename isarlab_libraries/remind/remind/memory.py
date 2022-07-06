from collections import deque
import numpy as np
import random


class Memory():

	new_samples = 0

	def __init__(self, lock, max_size, key_list, sequence_length=None, seed=None):
		super(Memory, self).__init__()
		self.lock = lock
		self.__max_size = max_size
		self.__key_list = key_list + ['memory_id']
		self.__sequence_length = sequence_length
		self.__can_be_picked_vector = deque(maxlen=max_size)
		self.__done_counter = self.__sequence_length
		self.__buffer = deque(maxlen=max_size)
		self.__indexes = []
		if seed is not None:
			np.random.seed(seed)
			random.seed(seed)

	@property
	def max_size(self):
		return self.__max_size

	@property
	def key_list(self):
		return self.__key_list

	@property
	def sequence_length(self):
		return self.__sequence_length

	@property
	def buffer(self):
		return self.__buffer

	@property
	def available_batches(self):
		return sum(self.__can_be_picked_vector)

	def get_indices_for_sampling(self, batch_size, mode):
		if mode == 'random':
			pickable_smaples_num = sum(self.__can_be_picked_vector)
			probabilities = [k / pickable_smaples_num for k in self.__can_be_picked_vector]
			indexes = np.random.choice(self.__indexes, batch_size, p=probabilities)
		elif mode == 'newest':
			indexes = []
			for i in reversed(self.__indexes):
				if self.__can_be_picked_vector[i] == 1:
					indexes.append(i)
				if len(indexes) == batch_size:
					break
		else:
			raise Exception('Invalid mode: "{}"'.format(mode))
		return indexes

	def insert(self, data):
		if set(data.keys()) != set(self.__key_list):
			raise Exception('Data to save has invalid keys.')
		else:
			if self.__max_size != len(self.__buffer):
				self.__indexes.append(len(self.__buffer))
			self.__buffer.append(data)
			if self.__sequence_length is not None:
				self.__done_counter -= 1
				self.__can_be_picked_vector.append(0)
				if self.__done_counter <= 0:
					self.__can_be_picked_vector[-1] = 1
				if data['done_exp'] == 1 or data['done_track'] == 1:
					self.__done_counter = self.__sequence_length
			Memory.new_samples += 1

	def get(self, batch_size, data_keys, mode):
		if any(seq_len > self.__sequence_length for seq_len in data_keys.values()):
			raise Exception('Requested sequence length is greater than specified sequence length.')
		else:
			indexes = self.get_indices_for_sampling(batch_size, mode)

			# list of list of dict (batch_size, key_len, sequence_length)
			batch_samples_list = [{k: [self.__buffer[i - j][k] for j in reversed(range(seq_len))] for k, seq_len in data_keys.items()} for i in indexes]
			
			Memory.new_samples = 0

			return batch_samples_list
