"""Plaintext UDP transport for NibeGW."""

from __future__ import annotations

import asyncio
import errno
from ipaddress import ip_address
import logging
import socket
import struct
from typing import Any, Callable, Optional, TypeVar

from nibe.connection.mixins import ConnectionStatus
from nibe.connection.nibegw import (
    Address,
    Command,
    NibeGWDatagramTransport,
    NibeGWProtocol,
)
from nibe.exceptions import AddressInUseException

logger = logging.getLogger("nibe").getChild(__name__)


class NibeGWUdpTransport(NibeGWDatagramTransport):
    """Plaintext UDP transport, with separate ports for reads and writes."""

    def __init__(
        self,
        protocol: NibeGWProtocol,
        remote_ip: Optional[str] = None,
        remote_read_port: int = 9999,
        remote_write_port: int = 10000,
        listening_ip: str = "0.0.0.0",
        listening_port: int = 9999,
    ) -> None:
        """Internal, use create_nibegw_udp_connection()."""
        super().__init__(protocol)
        self._remote_ip = remote_ip
        self._listening_ip = listening_ip
        self._listening_port = listening_port
        modbus40 = int(Address.MODBUS40)
        self._remote_ports = {
            (modbus40, int(Command.MODBUS_READ_REQ)): remote_read_port,
            (modbus40, int(Command.MODBUS_WRITE_REQ)): remote_write_port,
        }

    @property
    def remote_ip(self) -> Optional[str]:
        return self._remote_ip

    async def _open(self) -> None:
        logger.info(f"Starting UDP server on port {self._listening_port}")

        family, type, proto, _, sockaddr = socket.getaddrinfo(
            self._listening_ip,
            self._listening_port,
            type=socket.SOCK_DGRAM,
            proto=socket.IPPROTO_UDP,
            family=socket.AddressFamily.AF_INET,
        )[0]

        sock = socket.socket(family, type, proto)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        try:
            if ip_address(sockaddr[0]).is_multicast:
                group_bin = socket.inet_pton(family, sockaddr[0])
                if family == socket.AF_INET:  # IPv4
                    sock.bind(("", sockaddr[1]))
                    mreq = group_bin + struct.pack("=I", socket.INADDR_ANY)
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                else:
                    sock.bind(("", sockaddr[1]))
                    mreq = group_bin + struct.pack("@I", 0)
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
            elif self._listening_ip:
                sock.bind(sockaddr)
            else:
                sock.bind(("", sockaddr[1]))
        except OSError as exception:
            if exception.errno == errno.EADDRINUSE:
                raise AddressInUseException(f"Address in use {sockaddr}")
            raise

        await asyncio.get_event_loop().create_datagram_endpoint(lambda: self, sock=sock)

    def send_request(self, address: int, frame: bytes) -> None:
        assert self._transport, "Transport is closed"
        port = self._remote_ports.get((address, frame[1]))
        if port is None:
            raise ValueError(f"No port for request 0x{address:x}:0x{frame[1]:x}")
        self._transport.sendto(frame, (self._remote_ip, port))

    def datagram_received(self, data: bytes, addr: Any) -> None:
        self._protocol.frame_received(data)

        # Without a configured address, the first pump answering is used
        if not self._remote_ip and self._protocol.status == ConnectionStatus.CONNECTED:
            logger.debug("Pump discovered at %s", addr)
            self._remote_ip = addr[0]

    def error_received(self, exc: Exception) -> None:
        logger.error(exc)


_P = TypeVar("_P", bound=NibeGWProtocol)


async def create_nibegw_udp_connection(
    protocol_factory: Callable[[], _P],
    remote_ip: Optional[str] = None,
    remote_read_port: int = 9999,
    remote_write_port: int = 10000,
    listening_ip: str = "0.0.0.0",
    listening_port: int = 9999,
) -> tuple[NibeGWUdpTransport, _P]:
    """Connect a NibeGW protocol to a gateway over plaintext UDP.

    Like loop.create_datagram_endpoint(): returns the connected transport
    and the protocol created by protocol_factory.
    """
    protocol = protocol_factory()
    transport = NibeGWUdpTransport(
        protocol,
        remote_ip,
        remote_read_port,
        remote_write_port,
        listening_ip,
        listening_port,
    )
    await transport._open()
    return transport, protocol
