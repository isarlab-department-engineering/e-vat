import selectors
import threading
import socket
import pickle
import types
import time


class IsarSocket:

	def __init__(self, address, port, server, name=None):
		super(IsarSocket, self).__init__()
		self.server = server
		self.name = name
		if server:
			self.sel = None
			self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
			self.server_socket.bind((address, port))
			self.server_socket.setblocking(False)
			self.num_clients = 0
		else:
			check = False
			while not check:
				try:
					self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
					self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
					self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
					self.sock.connect((address, port))
					check = True
				except:
					pass

	def __service_connection(self, key, mask):
		sock = key.fileobj
		data = key.data
		if mask & selectors.EVENT_READ:
			return self.receive(sock), sock

	def send(self, data, sock=None):
		if self.server:
			self.lock.acquire()
			if sock is not None:
				self.__send(data, sock)
			else:
				events = self.sel.select(timeout=None)
				for key, mask in events:
					sock = key.fileobj
					if mask & selectors.EVENT_WRITE:
						print('learner')
						self.__send(data, sock)
			self.lock.release()
		else:
			self.__send(data, self.sock)

	def __flush(self, sock):
		while True:
			try:
				sock.recv(data_len - num_bytes)
			except:
				return

	def flush(self):
		if self.server:
			self.lock.acquire()
			events = self.sel.select(timeout=None)
			for key, mask in events:
				sock = key.fileobj
				if mask & selectors.EVENT_WRITE:
					self.__flush(sock)
			self.lock.release()

	def __accept_wrapper(self, sock, num_clients):
		if num_clients == -1 or num_clients > self.num_clients:
			self.lock.acquire()
			conn, addr = sock.accept()
			conn.setblocking(False)
			data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'')
			events = selectors.EVENT_READ | selectors.EVENT_WRITE
			self.sel.register(conn, events, data=data)
			self.lock.release()
			self.num_clients += 1

		if num_clients == self.num_clients:
			return False
		else:
			return True

	def __send_bytes(self, sock, data):
		data_len = len(data)
		num_bytes = 0
		# print('dentro')
		while num_bytes < data_len:
			try:
				sent = sock.send(data[num_bytes:])
				num_bytes += sent
				# print('num_bytes', num_bytes, data_len)
			except BlockingIOError:
				pass

	def __send(self, data, sock):
		data_string = pickle.dumps(data, -1)
		data_len = str(len(data_string))

		data_len = data_len.ljust(15)[:15]

		# print('invio')
		self.__send_bytes(sock, bytes(data_len, 'utf8'))
		# print('metà')
		self.__send_bytes(sock, data_string)
		# print('finito')

		# data = self.__recv_bytes(sock, 6)
		#
		# ack = [x.decode('utf-8') for x in data][0]
		# if ack != 'my_ack':
		# 	raise Exception('Acknowledgment failed (my_ack != {}).'.format(ack))

	def __recv_bytes(self, sock, data_len):
		data = []
		num_bytes = 0
		while num_bytes < data_len:
			try:
				packet = sock.recv(data_len - num_bytes)
				num_bytes += len(packet)
				data.append(packet)
			except BlockingIOError:
				pass
		return data

	def receive(self, sock=None):
		if not self.server:
			sock = self.sock
		else:
			self.lock.acquire()

		data = self.__recv_bytes(sock, 15)
		data_len = [x.decode('utf-8') for x in data][0]
		data_len = int(data_len)

		data = self.__recv_bytes(sock, data_len)
		data = pickle.loads(b"".join(data))

		# ack = 'my_ack'
		# self.__send_bytes(sock, bytes(ack, 'utf8'))

		if self.server:
			self.lock.release()

		return data

	def run_server(self, num_clients=-1):
		if not self.server:
			raise Exception('The socket is not a server.')
		else:
			self.server_socket.listen()
			self.lock = threading.Lock()
			self.sel = selectors.DefaultSelector()
			self.sel.register(self.server_socket, selectors.EVENT_READ, data=None)
			while True:
				self.lock.acquire()
				events = self.sel.select(timeout=None)
				self.lock.release()
				for key, mask in events:
					if key.data is None:
						if not self.__accept_wrapper(key.fileobj, num_clients):
							return
					else:
						data = self.__service_connection(key, mask)
						if data is not None:
							yield data

	def close(self):
		if self.server:
			self.server_socket.shutdown(socket.SHUT_RDWR)
			self.server_socket.close()
		else:
			self.sock.shutdown(socket.SHUT_RDWR)
			self.sock.close()
		print ("Isarsocket closed.")
