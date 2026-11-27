"""Encrypted NibeGW connection (ESPHome nibegw `udp_noise` transport).

The protocol follows the WireGuard data plane. The handshake is
Noise_NNpsk0_25519_ChaChaPoly_SHA256 with prologue "nibegw-udp-noise-v1",
initiated by the client. All integers are little endian and every message
starts with a type byte and three zero bytes:

    1 HANDSHAKE_INITIATION  type | sender_index | noise message 1 (48 bytes)
    2 HANDSHAKE_RESPONSE    type | sender_index | receiver_index | noise message 2
    4 TRANSPORT_DATA        type | receiver_index | counter (u64) | ciphertext

Transport data is ChaCha20-Poly1305 (IETF) with nonce = 4 zero bytes | counter
and no associated data. Counters must increase. The key of each direction is
derived from the Noise transport key k as ENCRYPT(k, 2^64 - 2, "", zeros[32])[0:32].

The plaintext is empty for a keepalive, which the gateway echoes. Otherwise it
is a request (address u16 | token | frame) to the gateway, or a telegram (frame
as received on RS485) from the gateway.
"""

from __future__ import annotations

import asyncio
import binascii
import logging
import secrets
import struct
from typing import Any, Callable, Literal, Optional, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from noise.connection import NoiseConnection

from nibe.coil import Coil, CoilData
from nibe.connection import (
    DEFAULT_TIMEOUT,
    READ_PRODUCT_INFO_TIMEOUT,
    Connection,
    verify_connectivity_read_write_alarm,
)
from nibe.connection.mixins import ConnectionStatus, ConnectionStatusMixin
from nibe.connection.nibegw import NibeGWProtocol
from nibe.event_server import EventServer
from nibe.heatpump import HeatPump, ProductInfo

logger = logging.getLogger("nibe").getChild(__name__)

PROTOCOL_NAME = b"Noise_NNpsk0_25519_ChaChaPoly_SHA256"
PROLOGUE = b"nibegw-udp-noise-v1"
KEY_DERIVE_NONCE = 2**64 - 2

TYPE_HANDSHAKE_INITIATION = 1
TYPE_HANDSHAKE_RESPONSE = 2
TYPE_TRANSPORT_DATA = 4

RESPONSE_SIZE = 60
DATA_HEADER_SIZE = 16
MAC_SIZE = 16

HANDSHAKE_RETRY = 2.0
KEEPALIVE_INTERVAL = 15.0
# Session considered lost without any packet received in this time
SESSION_TIMEOUT = 3 * KEEPALIVE_INTERVAL
START_TIMEOUT = 5.0


def decode_key(key: Union[str, bytes]) -> bytes:
    """Decode a base64 pre-shared key, the same format as ESPHome uses."""
    if isinstance(key, str):
        try:
            key = binascii.a2b_base64(key)
        except binascii.Error as err:
            raise ValueError("Key must be base64 encoded") from err
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")
    return key


def _nonce(counter: int) -> bytes:
    return b"\x00" * 4 + counter.to_bytes(8, "little")


def _header(message_type: int) -> bytes:
    return bytes([message_type, 0, 0, 0])


def derive_key(noise_key: bytes) -> bytes:
    """Transport data key from a Noise transport key."""
    return ChaCha20Poly1305(noise_key).encrypt(
        _nonce(KEY_DERIVE_NONCE), bytes(32), None
    )[:32]


class Session:
    """Keys and counters of one session."""

    def __init__(
        self, tx_key: bytes, rx_key: bytes, local_index: int, remote_index: int
    ) -> None:
        self._tx = ChaCha20Poly1305(tx_key)
        self._rx = ChaCha20Poly1305(rx_key)
        self.local_index = local_index
        self.remote_index = remote_index
        self._tx_counter = 0
        self._rx_counter = 0  # next acceptable counter

    def seal(self, plaintext: bytes) -> bytes:
        """Encrypt into a transport data packet."""
        counter = self._tx_counter
        self._tx_counter += 1
        header = _header(TYPE_TRANSPORT_DATA) + struct.pack(
            "<IQ", self.remote_index, counter
        )
        return header + self._tx.encrypt(_nonce(counter), plaintext, None)

    def open(self, packet: bytes) -> Optional[bytes]:
        """Decrypt a transport data packet, None if invalid, old or replayed."""
        if len(packet) < DATA_HEADER_SIZE + MAC_SIZE or packet[:4] != _header(
            TYPE_TRANSPORT_DATA
        ):
            return None
        receiver_index, counter = struct.unpack("<IQ", packet[4:DATA_HEADER_SIZE])
        if receiver_index != self.local_index or counter < self._rx_counter:
            return None
        try:
            plaintext = self._rx.decrypt(
                _nonce(counter), packet[DATA_HEADER_SIZE:], None
            )
        except InvalidTag:
            return None
        self._rx_counter = counter + 1
        return plaintext


class Handshake:
    """Initiator side of the handshake."""

    def __init__(self, psk: bytes) -> None:
        self.sender_index = secrets.randbits(32)
        self._noise = NoiseConnection.from_name(PROTOCOL_NAME)
        self._noise.set_as_initiator()
        self._noise.set_psks(psk)
        self._noise.set_prologue(PROLOGUE)
        self._noise.start_handshake()

    def initiation(self) -> bytes:
        return (
            _header(TYPE_HANDSHAKE_INITIATION)
            + struct.pack("<I", self.sender_index)
            + bytes(self._noise.write_message())
        )

    def finish(self, packet: bytes) -> Optional[Session]:
        """Session from a handshake response, None if not for this handshake."""
        if len(packet) != RESPONSE_SIZE or packet[:4] != _header(
            TYPE_HANDSHAKE_RESPONSE
        ):
            return None
        remote_index, receiver_index = struct.unpack("<II", packet[4:12])
        if receiver_index != self.sender_index:
            return None
        self._noise.read_message(packet[12:])
        proto = self._noise.noise_protocol
        return Session(
            derive_key(proto.cipher_state_encrypt.k),
            derive_key(proto.cipher_state_decrypt.k),
            self.sender_index,
            remote_index,
        )


def _cancel_timer(timer: Optional[asyncio.TimerHandle]) -> None:
    """Cancel a timer, also safe if it already fired or was cancelled."""
    if timer:
        timer.cancel()


def _restart_timer(
    timer: Optional[asyncio.TimerHandle], delay: float, callback: Callable[[], None]
) -> asyncio.TimerHandle:
    """Cancel a running timer and start it again."""
    _cancel_timer(timer)
    return asyncio.get_running_loop().call_later(delay, callback)


class NoiseTransport(asyncio.DatagramTransport, asyncio.DatagramProtocol):
    """Encrypted datagram transport on top of a UDP socket.

    To the application protocol it is a DatagramTransport whose sendto
    encrypts, and which delivers decrypted payloads. To the UDP socket it is
    the DatagramProtocol that receives the encrypted packets. The session is
    bound to the gateway, so the sendto address is ignored like for a
    connected datagram socket.
    """

    def __init__(self, protocol: asyncio.DatagramProtocol, psk: bytes) -> None:
        super().__init__()
        self._protocol = protocol
        self._psk = psk
        self._socket: Optional[asyncio.DatagramTransport] = None
        self._session: Optional[Session] = None
        self._handshake: Optional[Handshake] = None
        self._handshake_timer: Optional[asyncio.TimerHandle] = None
        self._keepalive_timer: Optional[asyncio.TimerHandle] = None
        self._session_timer: Optional[asyncio.TimerHandle] = None
        self.established = asyncio.Event()

    # Transport side, used by the application protocol

    def sendto(self, data, addr=None) -> None:
        if not self._session:
            logger.debug("No encrypted session, dropping payload")
            return
        self._send(self._session, data)

    def close(self) -> None:
        self._cancel_timers()
        if self._socket:
            self._socket.close()

    def abort(self) -> None:
        self.close()

    def is_closing(self) -> bool:
        return self._socket is None or self._socket.is_closing()

    def get_extra_info(self, name, default=None):
        assert self._socket, "Transport is closed"
        return self._socket.get_extra_info(name, default)

    def get_protocol(self) -> asyncio.BaseProtocol:
        return self._protocol

    def set_protocol(self, protocol: asyncio.BaseProtocol) -> None:
        assert isinstance(protocol, asyncio.DatagramProtocol)
        self._protocol = protocol

    # Protocol side, called by the UDP socket

    def connection_made(self, transport) -> None:
        self._socket = transport
        self._start_handshake()
        self._protocol.connection_made(self)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._cancel_timers()
        self._socket = None
        self._protocol.connection_lost(exc)

    def datagram_received(self, data: bytes, addr: Any) -> None:
        if self._handshake and data[:1] == bytes([TYPE_HANDSHAKE_RESPONSE]):
            self._on_handshake_response(self._handshake, data)
            return

        plaintext = self._session.open(data) if self._session else None
        if plaintext is None:
            logger.debug(f"Dropped invalid packet from {addr}")
            return

        self._session_timer = _restart_timer(
            self._session_timer, SESSION_TIMEOUT, self._session_lost
        )
        if plaintext:
            self._protocol.datagram_received(plaintext, addr)

    def error_received(self, exc: Exception) -> None:
        # An unreachable gateway is handled by handshake retries
        logger.debug(exc)

    def _start_handshake(self) -> None:
        """Send a handshake initiation, repeated until the gateway answers."""
        assert self._socket, "Transport is closed"
        self._handshake = Handshake(self._psk)
        self._socket.sendto(self._handshake.initiation())
        self._handshake_timer = _restart_timer(
            self._handshake_timer, HANDSHAKE_RETRY, self._start_handshake
        )

    def _on_handshake_response(self, handshake: Handshake, data: bytes) -> None:
        try:
            session = handshake.finish(data)
        except Exception as err:
            logger.debug(f"Invalid handshake response: {err}")
            return
        if session is None:
            return

        _cancel_timer(self._handshake_timer)
        self._handshake = None
        self._session = session
        self._session_timer = _restart_timer(
            self._session_timer, SESSION_TIMEOUT, self._session_lost
        )

        # The gateway uses the session once it sees data with the new keys
        self._send(session, b"")
        self.established.set()
        logger.info("Encrypted session established")

    def _send(self, session: Session, plaintext: bytes) -> None:
        assert self._socket, "Transport is closed"
        self._socket.sendto(session.seal(plaintext))
        self._keepalive_timer = _restart_timer(
            self._keepalive_timer, KEEPALIVE_INTERVAL, self._send_keepalive
        )

    def _send_keepalive(self) -> None:
        if self._session:
            self._send(self._session, b"")

    def _session_lost(self) -> None:
        logger.warning("Encrypted session lost, reconnecting")
        self._session = None
        self.established.clear()
        _cancel_timer(self._keepalive_timer)
        self._start_handshake()

    def _cancel_timers(self) -> None:
        _cancel_timer(self._handshake_timer)
        _cancel_timer(self._keepalive_timer)
        _cancel_timer(self._session_timer)


class NibeGWNoise(
    asyncio.DatagramProtocol, Connection, EventServer, ConnectionStatusMixin
):
    """NibeGW connection over the encrypted ESPHome `udp_noise` transport.

    The session is initiated from this side, so the gateway address is
    required and no local port is opened.
    """

    PRODUCT_INFO_EVENT = "product_info"

    def __init__(
        self,
        heatpump: HeatPump,
        remote_ip: str,
        key: Union[str, bytes],
        remote_port: int = 9999,
        read_retries: int = 3,
        write_retries: int = 3,
        table_processing_mode: Literal["permissive", "strict"] = "permissive",
    ) -> None:
        super().__init__()
        self._heatpump = heatpump
        self._remote_ip = remote_ip
        self._remote_port = remote_port
        self._psk = decode_key(key)
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol = NibeGWProtocol(
            heatpump,
            send=self._send_request,
            on_response=self._on_response,
            on_product_info=self._on_product_info,
            read_retries=read_retries,
            write_retries=write_retries,
            table_processing_mode=table_processing_mode,
        )

    async def start(self):
        self.status = ConnectionStatus.INITIALIZING
        logger.info(
            f"Connecting encrypted session to {self._remote_ip}:{self._remote_port}"
        )
        _, noise = await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: NoiseTransport(self, self._psk),
            remote_addr=(self._remote_ip, self._remote_port),
        )

        try:
            await asyncio.wait_for(noise.established.wait(), START_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("No encrypted session established yet, retrying")

        if self._heatpump.word_swap is None:
            await self._protocol.detect_word_swap()

    def connection_made(self, transport) -> None:
        self.status = ConnectionStatus.LISTENING
        self._transport = transport

    def datagram_received(self, data: bytes, addr: Any) -> None:
        self._protocol.frame_received(data, addr)

    def _send_request(self, address: int, data: bytes) -> None:
        assert self._transport, "Transport is closed"
        token = data[1]
        self._transport.sendto(struct.pack("<HB", address, token) + data)

    def _on_response(self, addr: Any) -> None:
        self.status = ConnectionStatus.CONNECTED

    def _on_product_info(self, product_info: ProductInfo) -> None:
        self.notify_event_listeners(self.PRODUCT_INFO_EVENT, product_info=product_info)

    @property
    def remote_ip(self) -> Optional[str]:
        return self._remote_ip

    async def read_coil(self, coil: Coil, timeout: float = DEFAULT_TIMEOUT) -> CoilData:
        return await self._protocol.read_coil(coil, timeout)

    async def write_coil(
        self, coil_data: CoilData, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        await self._protocol.write_coil(coil_data, timeout)

    async def read_product_info(
        self, timeout: float = READ_PRODUCT_INFO_TIMEOUT
    ) -> ProductInfo:
        return await self._protocol.read_product_info(timeout)

    async def verify_connectivity(self):
        await verify_connectivity_read_write_alarm(self, self._heatpump)

    async def stop(self):
        if self._transport:
            self._transport.close()
            self._transport = None
        await asyncio.sleep(0)
        self.status = ConnectionStatus.DISCONNECTED
