# from isarsocket.isarsocket import IsarSocket
from isarsocket.simple_socket import SimpleSocket
from remind.memory import Memory
import multiprocessing as mp
import numpy as np
import threading
import random
import time


class Remind(mp.Process):
	def __init__(self, first_address, num_agents, seed=None):
		super(Remind, self).__init__()
		self.address_list = [(first_address[0], first_address[1] + i) for i in range(num_agents + 1)]
		print(len(self.address_list))
		self.learner_address = [(self.address_list[-1][0], self.address_list[-1][1] + 1)]
		# self.isar_socket = IsarSocket(server=True, address=address, port=port, name='remind')
		self.__memories = {}
		self.in_socket = None
		self.out_socket = None
		if seed is not None:
			np.random.seed(seed)
			random.seed(seed)

	def run(self):
		self.in_socket = SimpleSocket(address_list=self.address_list, server=True, name='remind_in')
		self.out_socket = SimpleSocket(address_list=self.learner_address, server=True, name='remind_out')
		# for data in self.isar_socket.run_server():
		for data in self.in_socket.server_receive():
			command, settings = data
			settings['sock'] = None
			method_to_call = getattr(self, command)
			thread1 = threading.Thread(target=method_to_call, kwargs=settings)
			thread1.start()

	def init_memory(self, sock, memory_id, max_size, key_list, sequence_length=None, seed=None):
		self.__memories[memory_id] = Memory(lock=threading.Lock(), max_size=max_size, key_list=key_list, sequence_length=sequence_length, seed=seed)

	def insert(self, sock, memory_id, data):
		data['memory_id'] = memory_id
		self.__memories[memory_id].lock.acquire()
		self.__memories[memory_id].insert(data)
		self.__memories[memory_id].lock.release()

	def __check_if_ready(self, batch_size, min_new_samples, memory_ids):
		ready = False
		while not ready:
			if Memory.new_samples >= min_new_samples:
				try:
					batch_size_base = int(batch_size / len(memory_ids))
					num_remaining_batches = batch_size % len(memory_ids)

					indexes = random.sample(range(len(memory_ids)), num_remaining_batches)

					ready = True
					batch_sizes = {}
					for memory_id in memory_ids:
						bs = batch_size_base + 1 if memory_id in indexes else batch_size_base
						batch_sizes[memory_id] = bs
						if bs > self.__memories[memory_id].available_batches:
							ready = False
							break
				except:
					pass
			if not ready:
				time.sleep(0.01)
		return batch_sizes

	def get(self, sock, batch_size, data_keys, min_new_samples, mode, memory_ids=None):
		if memory_ids is None:
			memory_ids = self.__memories.keys()
		batch_sizes = self.__check_if_ready(batch_size, min_new_samples, memory_ids)

		# list of list of dict (batch_size, key_len, sequence_length)
		batch_samples_list = []
		for memory_id in memory_ids:
			self.__memories[memory_id].lock.acquire()
			batch_samples_list += self.__memories[memory_id].get(batch_sizes[memory_id], data_keys, mode)
			self.__memories[memory_id].lock.release()

		# dict of numpy array (key_len, sequence_length, batch_size)
		batch_samples_dict = {k: np.expand_dims([[batch_samples_list[b][k][s] for b in range(batch_size)] for s in range(seq_len)], -1) if np.isscalar(batch_samples_list[0][k][0]) else np.array([[batch_samples_list[b][k][s] for b in range(batch_size)] for s in range(seq_len)]) for k, seq_len in data_keys.items()}

		# self.isar_socket.send(batch_samples_dict, sock)
		self.out_socket.server_send(batch_samples_dict)
