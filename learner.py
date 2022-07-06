from isarsocket.simple_socket import SimpleSocket
import torch.multiprocessing as mp
import numpy as np
import importlib
import threading
import random
import torch
import wandb
import time
import os


class ReadWriteLock:
    """ A lock object that allows many simultaneous "read locks", but
    only one "write lock." """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        """ Acquire a read lock. Blocks only if a thread has
        acquired the write lock. """
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """ Release a read lock. """
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """ Acquire a write lock. Blocks until there are no
        acquired read or write locks. """
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """ Release a write lock. """
        self._read_ready.release()


def get_sockets(num_agents, learner_address, memory_address, main_address):
    agents_in_address = [(learner_address[0], learner_address[1] + i) for i in range(num_agents + 1)]
    agents_out_address = [(learner_address[0], learner_address[1] + 100 + i) for i in range(num_agents + 1)]

    main_address = [(main_address[0], main_address[1] + num_agents + 1)]

    main_socket = SimpleSocket(address_list=main_address, server=False, name='learner_main')
    memory_in_socket = SimpleSocket(address_list=[memory_address], server=False, name='learner_mem_in')
    memory_out_socket = SimpleSocket(address_list=[(memory_address[0], memory_address[1] + 1)], server=False,
                                     name='learner_mem_out')

    agents_in_socket = SimpleSocket(address_list=agents_in_address, server=True, name='learner_in')
    agents_out_socket = SimpleSocket(address_list=agents_out_address, server=True, name='learner_out')

    return memory_in_socket, memory_out_socket, agents_in_socket, agents_out_socket, main_socket


class MyLearner(mp.Process):
    def __init__(self, settings):
        super(MyLearner, self).__init__()
        # self.daemon = True
        self.settings = settings

        self.agent = None
        self.update_data = {}
        self.model_update = 0
        self.lock = None
        self.fps = 0
        self.main_socket = None
        self.memory_in_socket = None
        self.memory_out_socket = None
        self.agents_in_socket = None
        self.agents_out_socket = None

        print(self.settings['learner_gpu_id'])

    def run(self):
        print('learner', os.getpid())

        torch.manual_seed(self.settings['learner_rank'])
        torch.cuda.manual_seed(self.settings['learner_rank'])
        np.random.seed(self.settings['learner_rank'])
        random.seed(self.settings['learner_rank'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.lock = ReadWriteLock()

        self.memory_in_socket, self.memory_out_socket, self.agents_in_socket, self.agents_out_socket, self.main_socket = \
            get_sockets(self.settings['num_agents'], self.settings['learner_address'],
                        self.settings['memory_address'], self.settings['main_to_agent_address'])

        # shared_model = copy.deepcopy(self.settings['model_init_queue'].get())
        shared_model = self.main_socket.client_receive()

        if self.settings['WandB']:
            if self.settings['IMPALA']:
                wandb.init(project="AT_{}".format(self.settings['workstation_name']), entity='pegaso', name='IMPALA_Perseo', config=self.settings)
                wandb.watch(shared_model['net'])
            else:
                if self.settings['TD3']:
                    wandb.init(project="exploration", entity='pegaso', name='TD3', config=self.settings)
                else:
                    wandb.init(project="exploration", entity='pegaso', name='DDPG', config=self.settings)
                wandb.watch(shared_model['net'])

        self.settings['net'] = shared_model
        module = importlib.import_module(self.settings['algorithm'])
        class_ = getattr(module, self.settings['algorithm'])
        self.agent = class_(self.settings)

        if not self.settings['on_policy']:
            update_thread = threading.Thread(target=self.update)
            update_thread.start()

        for data in self.agents_in_socket.server_receive():
            command, settings = data
            method_to_call = getattr(self, command)
            thread1 = threading.Thread(target=method_to_call, kwargs=settings)
            thread1.start()

    def update(self):
        torch.manual_seed(self.settings['learner_rank'])
        torch.cuda.manual_seed(self.settings['learner_rank'])
        np.random.seed(self.settings['learner_rank'])
        random.seed(self.settings['learner_rank'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        while True:
            self.memory_in_socket.client_send(['get', self.settings['update_settings']])
            samples = self.memory_out_socket.client_receive()

            if self.model_update % 100 == 0:
                print('model_update', self.model_update)

            update_data = self.agent.update(samples)

            self.lock.acquire_write()
            self.update_data['log'] = update_data
            self.agent.model.cpu()
            self.update_data['last_state_dict_te'] = self.agent.model.state_dict()
            self.agent.model.to(self.agent.device)
            self.model_update += 1
            self.update_data['model_update'] = self.model_update
            self.lock.release_write()

    def save_log(self, training_log):
        if self.settings['WandB']:
            try:
                wandb.log(training_log)
            except:
                print("WANDB EXCEPTION")

    def send_last_state_dict(self, model_update, socket_id):
        while model_update == self.model_update:
            time.sleep(0.01)
        self.lock.acquire_read()
        data = self.update_data
        self.lock.release_read()
        self.agents_out_socket.server_send(data, socket_id)

    def on_policy_update(self, agent_state_dict, memory_id, socket_id):
        if not self.settings['on_policy']:
            raise Exception(
                'Cannot use "on_policy_update" for off-policy training: call "send_last_state_dict" instead.')
        self.settings['update_settings']['memory_ids'] = [memory_id]
        self.lock.acquire_write()
        self.memory_in_socket.client_send(['get', self.settings['update_settings']])
        samples = self.memory_out_socket.client_receive()
        update_data = self.agent.on_policy_update(agent_state_dict, samples)
        self.update_data['log'] = update_data
        self.agent.model.cpu()
        self.update_data['last_state_dict'] = self.agent.model.state_dict()
        self.agent.model.to(self.agent.device)
        self.model_update += 1
        self.update_data['model_update'] = self.model_update
        data = self.update_data
        self.agents_out_socket.server_send(data, socket_id)
        self.lock.release_write()

    def off_policy_update(self, memory_id, socket_id):
        if not self.settings['on_policy']:
            raise Exception(
                'Cannot use "off_policy_update" for off-policy training: call "send_last_state_dict" instead.')
        self.settings['update_settings']['memory_ids'] = [memory_id]
        self.lock.acquire_write()
        self.memory_in_socket.client_send(['get', self.settings['update_settings']])
        samples = self.memory_out_socket.client_receive()
        update_data = self.agent.update(samples)
        self.update_data['log'] = update_data
        self.agent.model.cpu()
        self.update_data['last_state_dict'] = self.agent.model.state_dict()
        self.agent.model.to(self.agent.device)
        self.model_update += 1
        self.update_data['model_update'] = self.model_update
        data = self.update_data
        self.agents_out_socket.server_send(data, socket_id)
        self.lock.release_write()
