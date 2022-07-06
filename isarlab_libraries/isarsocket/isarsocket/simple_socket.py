import socket
import pickle
import select


class SimpleSocket:

    def __init__(self, address_list, server, name=None):
        super(SimpleSocket, self).__init__()
        self.server = server
        self.name = name
        if server:
            self.clients = []
            for address, port in address_list:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((address, port))
                self.server_socket.setblocking(False)
                self.server_socket.listen()
                while True:
                    try:
                        conn, addr = self.server_socket.accept()
                        break
                    except BlockingIOError:
                        pass
                self.clients.append(conn)
        else:
            assert len(address_list) == 1
            address, port = address_list[0]
            check = False
            while not check:
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.sock.connect((address, port))
                    check = True
                except:
                    pass

    @staticmethod
    def __recv_bytes(sock, data_len):
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

    @staticmethod
    def __send_bytes(sock, data):
        data_len = len(data)
        num_bytes = 0
        while num_bytes < data_len:
            try:
                sent = sock.send(data[num_bytes:])
                num_bytes += sent
            except BlockingIOError:
                pass

    @staticmethod
    def __sock_receive(sock):
        data = SimpleSocket.__recv_bytes(sock, 15)
        data_len = [x.decode('utf-8') for x in data][0]
        data_len = int(data_len)

        data = SimpleSocket.__recv_bytes(sock, data_len)
        data = pickle.loads(b"".join(data))
        return data

    @staticmethod
    def __sock_send(sock, data):
        data_string = pickle.dumps(data, -1)
        data_len = str(len(data_string)).ljust(15)[:15]

        SimpleSocket.__send_bytes(sock, bytes(data_len, 'utf8'))
        SimpleSocket.__send_bytes(sock, data_string)

    def client_receive(self):
        assert not self.server
        return SimpleSocket.__sock_receive(self.sock)

    def client_send(self, data):
        assert not self.server
        SimpleSocket.__sock_send(self.sock, data)

    def server_receive(self):
        assert self.server
        while 1:
            for client in self.clients:
                ready = select.select([client], [], [], 0.01)
                if ready[0]:
                    yield SimpleSocket.__sock_receive(client)

    def server_send(self, data, socket_id=None):
        assert self.server
        if socket_id is not None:
            SimpleSocket.__sock_send(self.clients[socket_id], data)
        else:
            for client in self.clients:
                SimpleSocket.__sock_send(client, data)
