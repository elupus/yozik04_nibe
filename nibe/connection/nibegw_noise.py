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
from contextlib import suppress
import logging
import secrets
import struct
import time
from typing import Optional, Union

from nibe.connection.mixins import ConnectionStatus
from nibe.connection.nibegw import NibeGWTransport
from nibe.exceptions import NibeException

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

MODBUS40 = 0x20

HANDSHAKE_RETRY = 2.0
KEEPALIVE_INTERVAL = 15.0
# Session considered lost without any packet received in this time
SESSION_TIMEOUT = 3 * KEEPALIVE_INTERVAL
START_TIMEOUT = 5.0
TICK = 1.0


def _import_crypto():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        from noise.connection import NoiseConnection
    except ImportError as err:  # pragma: no cover
        raise NibeException(
            "Encrypted NibeGW requires the 'noise' extra: pip install nibe[noise]"
        ) from err
    return ChaCha20Poly1305, NoiseConnection


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
    ChaCha20Poly1305, _ = _import_crypto()
    return ChaCha20Poly1305(noise_key).encrypt(
        _nonce(KEY_DERIVE_NONCE), bytes(32), None
    )[:32]


class Session:
    """Keys and counters of one session."""

    def __init__(
        self, tx_key: bytes, rx_key: bytes, local_index: int, remote_index: int
    ) -> None:
        ChaCha20Poly1305, _ = _import_crypto()
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
        except Exception:  # cryptography.exceptions.InvalidTag
            return None
        self._rx_counter = counter + 1
        return plaintext


class Handshake:
    """Initiator side of the handshake."""

    def __init__(self, psk: bytes, started: float = 0.0) -> None:
        _, NoiseConnection = _import_crypto()
        self.sender_index = secrets.randbits(32)
        self.started = started
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


class NibeGWNoiseTransport(asyncio.DatagramProtocol, NibeGWTransport):
    """Encrypted transport for the ESPHome nibegw `udp_noise` transport.

    The session is initiated from this side, so the gateway address is
    required and no local port is opened. Use it with NibeGW:

        NibeGW(heatpump, transport=NibeGWNoiseTransport(remote_ip, key))
    """

    def __init__(
        self, remote_ip: str, key: Union[str, bytes], remote_port: int = 9999
    ) -> None:
        NibeGWTransport.__init__(self)
        self._psk = decode_key(key)
        self._remote_ip = remote_ip
        self._remote_port = remote_port
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._session: Optional[Session] = None
        self._handshake: Optional[Handshake] = None
        self._established = asyncio.Event()
        self._last_rx = 0.0
        self._last_tx = 0.0
        self._task: Optional[asyncio.Task] = None

    @property
    def remote_ip(self) -> Optional[str]:
        return self._remote_ip

    async def start(self) -> None:
        logger.info(
            f"Connecting encrypted session to {self._remote_ip}:{self._remote_port}"
        )
        await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: self, remote_addr=(self._remote_ip, self._remote_port)
        )
        self._task = asyncio.create_task(self._run())

        try:
            await asyncio.wait_for(self._established.wait(), START_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("No encrypted session established yet, retrying")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._transport:
            self._transport.close()
            self._transport = None
        self._session = None
        self._handshake = None
        self._established.clear()

    def connection_made(self, transport):
        self._transport = transport
        self._on_status(ConnectionStatus.LISTENING)

    def datagram_received(self, data: bytes, addr):
        if self._handshake and data[:1] == bytes([TYPE_HANDSHAKE_RESPONSE]):
            self._on_handshake_response(data)
            return

        plaintext = self._session.open(data) if self._session else None
        if plaintext is None:
            logger.debug(f"Dropped invalid packet from {addr}")
            return

        self._last_rx = time.monotonic()
        if plaintext:
            self._on_frame(plaintext, addr)

    def error_received(self, exc):
        # An unreachable gateway is handled by handshake retries
        logger.debug(exc)

    def send_request(self, data: bytes, write: bool) -> None:
        if not self._session:
            logger.debug("No encrypted session, dropping request")
            return
        token = data[1]
        self._send(struct.pack("<HB", MODBUS40, token) + data)

    def _on_handshake_response(self, data: bytes) -> None:
        try:
            session = self._handshake.finish(data)
        except Exception as err:
            logger.debug(f"Invalid handshake response: {err}")
            return
        if session is None:
            return

        self._handshake = None
        self._session = session
        self._last_rx = time.monotonic()
        # The gateway uses the session once it sees data with the new keys
        self._send(b"")
        self._established.set()
        logger.info("Encrypted session established")

    def _send(self, plaintext: bytes) -> None:
        if self._session and self._transport:
            self._transport.sendto(self._session.seal(plaintext))
            self._last_tx = time.monotonic()

    def _tick(self, now: float) -> None:
        if self._session and now - self._last_rx >= SESSION_TIMEOUT:
            logger.warning("Encrypted session lost, reconnecting")
            self._session = None
            self._established.clear()
            self._on_status(ConnectionStatus.LISTENING)

        if self._session is None:
            if (
                self._handshake is None
                or now - self._handshake.started >= HANDSHAKE_RETRY
            ):
                assert self._transport, "Transport is closed"
                self._handshake = Handshake(self._psk, now)
                self._transport.sendto(self._handshake.initiation())
        elif now - self._last_tx >= KEEPALIVE_INTERVAL:
            self._send(b"")

    async def _run(self) -> None:
        while True:
            self._tick(time.monotonic())
            await asyncio.sleep(TICK)
