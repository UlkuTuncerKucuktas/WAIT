import json
import os
import socket
import time

from wait.probe import monotonic_s

PORT = 47811
CONNECT_TIMEOUT_S = 30.0
# A barrier holds until the slowest rank arrives, which across a write phase is
# minutes.  The connect timeout must not outlive the connect.
BARRIER_TIMEOUT_S = 600.0


class RankError(RuntimeError):
    pass


def rank():
    return int(os.environ.get("SLURM_PROCID", "0"))


def world():
    # WAIT_SCALE first: srun sets SLURM_NTASKS per step, so a single-task prepare
    # sees 1 while its multi-task measure sees N, and the two phases compute
    # different ledger keys for the same cell.
    return int(os.environ.get("WAIT_SCALE") or os.environ.get("SLURM_NTASKS", "1"))


class _Channel:
    # Barrier bytes and gather lines share one socket, so a read that takes
    # whatever the kernel hands it will swallow the next message: rank 0 asked
    # for a JSON line and got the line plus the barrier byte behind it.  Every
    # read goes through this buffer, and what it does not need it keeps.
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def _fill(self):
        block = self.sock.recv(65536)
        if not block:
            raise RankError("peer closed the connection")
        self.buf += block

    def recv_exact(self, n):
        while len(self.buf) < n:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv_line(self):
        while b"\n" not in self.buf:
            self._fill()
        line, _, self.buf = self.buf.partition(b"\n")
        return line.decode()

    def sendall(self, data):
        self.sock.sendall(data)

    def close(self):
        self.sock.close()


class Barrier:
    # Over TCP rather than the filesystem: a barrier that creates and stats files
    # puts its own metadata traffic inside the phenomenon being measured, and the
    # rank count it synchronises is exactly the axis under test.
    def __init__(self, coordinator, size=None, me=None, port=PORT):
        self.size = world() if size is None else size
        self.rank = rank() if me is None else me
        self.peers = []
        self.chan = None
        if self.size == 1:
            return
        if self.rank == 0:
            self._listen(port)
        else:
            self._connect(coordinator, port)

    def _listen(self, port):
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen(self.size)
        for _ in range(self.size - 1):
            conn, _addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(BARRIER_TIMEOUT_S)
            self.peers.append(_Channel(conn))
        server.close()

    def _connect(self, coordinator, port):
        deadline = monotonic_s() + CONNECT_TIMEOUT_S
        while monotonic_s() < deadline:
            try:
                sock = socket.create_connection((coordinator, port), timeout=5)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(BARRIER_TIMEOUT_S)
                self.chan = _Channel(sock)
                return
            except OSError:
                time.sleep(0.05)
        raise RankError("rank %d could not reach %s:%d" % (self.rank, coordinator, port))

    def wait(self):
        if self.size == 1:
            return
        if self.rank == 0:
            for conn in self.peers:
                conn.recv_exact(1)
            for conn in self.peers:
                conn.sendall(b"1")
        else:
            self.chan.sendall(b"1")
            self.chan.recv_exact(1)

    def broadcast(self, value=0):
        """Release everyone, carrying rank 0's value with the release.

        A deadline the workload can act on has to be known while the work is
        happening, not computed afterwards -- and it must not come from the
        consumer's own pace, or a slow consumer widens its own budget and can
        never miss.
        """
        if self.size == 1:
            return value
        width = 20
        if self.rank == 0:
            for conn in self.peers:
                conn.recv_exact(1)
            payload = ("%0*d" % (width, value)).encode()
            for conn in self.peers:
                conn.sendall(payload)
            return value
        self.chan.sendall(b"1")
        return int(self.chan.recv_exact(width).decode())

    def gather(self, payload):
        # Rank 0 needs every rank's timings to take a maximum per round, and the
        # rendezvous socket is already open -- writing them to the filesystem
        # under test would add metadata traffic to the measurement.
        if self.size == 1:
            return [payload]
        if self.rank == 0:
            collected = [payload]
            for conn in self.peers:
                collected.append(json.loads(conn.recv_line()))
            return collected
        self.chan.sendall((json.dumps(payload) + "\n").encode())
        return []

    def close(self):
        for conn in self.peers:
            conn.close()
        if self.chan:
            self.chan.close()
        self.peers, self.chan = [], None
