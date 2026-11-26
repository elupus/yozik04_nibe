"""Encrypted NibeGW connection (ESPHome nibegw `udp_noise` transport).

The protocol follows the WireGuard data plane. The handshake is
Noise_NNpsk0_25519_ChaChaPoly_SHA256 with prologue "nibegw-udp-noise-v1",
initiated by the client. All integers are little endian and every message
starts with a type byte and three zero bytes:

    1 HANDSHAKE_INITIATION  type | sender_index | noise message 1 (48 bytes)
    2 HANDSHAKE_RESPONSE    type | sender_index | receiver_index | noise message 2
    3 HANDSHAKE_REJECT      type | receiver_index | reason (ascii)
    4 TRANSPORT_DATA        type | receiver_index | counter (u64) | ciphertext

Transport data is ChaCha20-Poly1305 (IETF) with nonce = 4 zero bytes | counter
and no associated data. The key of each direction is derived from the Noise
transport key k as ENCRYPT(k, 2^64 - 2, "", zeros[32])[0:32].

The plaintext is empty for a keepalive, which the gateway echoes, otherwise:

    REQUEST   0x01 | address (u16) | token | frame     client -> gateway
    TELEGRAM  0x02 | frame as received on RS485        gateway -> client

The client confirms a new session by sending transport data right after the
handshake. The gateway drops sessions without traffic after two minutes.
"""

from __future__ import annotations

import asyncio
import binascii
from contextlib import suppress
import logging
import secrets
import struct
import time
from typing import Literal, Optional, Union

from nibe.connection.mixins import ConnectionStatus
from nibe.connection.nibegw import NibeGW
from nibe.exceptions import NibeException
from nibe.heatpump import HeatPump

logger = logging.getLogger("nibe").getChild(__name__)

PROTOCOL_NAME = b"Noise_NNpsk0_25519_ChaChaPoly_SHA256"
PROLOGUE = b"nibegw-udp-noise-v1"
KEY_DERIVE_NONCE = 2**64 - 2

TYPE_HANDSHAKE_INITIATION = 1
TYPE_HANDSHAKE_RESPONSE = 2
TYPE_HANDSHAKE_REJECT = 3
TYPE_TRANSPORT_DATA = 4

INNER_REQUEST = 0x01
INNER_TELEGRAM = 0x02

NOISE_MESSAGE_SIZE = 48
RESPONSE_SIZE = 12 + NOISE_MESSAGE_SIZE
DATA_HEADER_SIZE = 16
MAC_SIZE = 16
REPLAY_WINDOW = 64

MODBUS40 = 0x20

REJECT_AFTER_MESSAGES = 2**64 - 2**13 - 1
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
    """Keys, counters and replay protection of one session."""

    def __init__(
        self, tx_key: bytes, rx_key: bytes, local_index: int, remote_index: int
    ) -> None:
        ChaCha20Poly1305, _ = _import_crypto()
        self._tx = ChaCha20Poly1305(tx_key)
        self._rx = ChaCha20Poly1305(rx_key)
        self.local_index = local_index
        self.remote_index = remote_index
        self.created = time.monotonic()
        self._tx_counter = 0
        self._rx_highest = -1
        self._rx_seen: set[int] = set()

    def seal(self, inner: bytes) -> bytes:
        """Encrypt a message into a transport data packet."""
        counter = self._tx_counter
        if counter >= REJECT_AFTER_MESSAGES:
            raise NibeException("Session exhausted")
        self._tx_counter += 1
        header = _header(TYPE_TRANSPORT_DATA) + struct.pack(
            "<IQ", self.remote_index, counter
        )
        return header + self._tx.encrypt(_nonce(counter), inner, None)

    def open(self, packet: bytes) -> Optional[bytes]:
        """Decrypt a transport data packet, None if invalid or replayed."""
        if len(packet) < DATA_HEADER_SIZE + MAC_SIZE or packet[:4] != _header(
            TYPE_TRANSPORT_DATA
        ):
            return None
        receiver_index, counter = struct.unpack("<IQ", packet[4:DATA_HEADER_SIZE])
        if receiver_index != self.local_index or counter >= REJECT_AFTER_MESSAGES:
            return None
        if counter <= self._rx_highest - REPLAY_WINDOW or counter in self._rx_seen:
            return None
        try:
            inner = self._rx.decrypt(_nonce(counter), packet[DATA_HEADER_SIZE:], None)
        except Exception:  # cryptography.exceptions.InvalidTag
            return None
        self._rx_seen.add(counter)
        if counter > self._rx_highest:
            self._rx_highest = counter
            self._rx_seen = {
                c for c in self._rx_seen if c > self._rx_highest - REPLAY_WINDOW
            }
        return inner


class Handshake:
    """Initiator side of the handshake."""

    def __init__(self, psk: bytes) -> None:
        _, NoiseConnection = _import_crypto()
        self.sender_index = secrets.randbits(32)
        self.started = time.monotonic()
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

    def matches(self, packet: bytes) -> bool:
        return (
            len(packet) == RESPONSE_SIZE
            and packet[:4] == _header(TYPE_HANDSHAKE_RESPONSE)
            and struct.unpack("<I", packet[8:12])[0] == self.sender_index
        )

    def rejected(self, packet: bytes) -> Optional[str]:
        """Reason if the packet rejects this handshake."""
        if (
            len(packet) >= 8
            and packet[:4] == _header(TYPE_HANDSHAKE_REJECT)
            and struct.unpack("<I", packet[4:8])[0] == self.sender_index
        ):
            return packet[8:].decode(errors="replace")
        return None

    def finish(self, packet: bytes) -> Session:
        """Process the handshake response, raises if invalid."""
        (remote_index,) = struct.unpack("<I", packet[4:8])
        self._noise.read_message(packet[12:])
        if not self._noise.handshake_finished:
            raise NibeException("Handshake not finished")
        proto = self._noise.noise_protocol
        return Session(
            derive_key(proto.cipher_state_encrypt.k),
            derive_key(proto.cipher_state_decrypt.k),
            self.sender_index,
            remote_index,
        )


class NibeGWNoise(NibeGW):
    """NibeGW connection over the encrypted ESPHome `udp_noise` transport.

    Unlike the plaintext protocol the session is initiated from this side, so
    the gateway address is required and no local port is opened.
    """

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
        super().__init__(
            heatpump,
            remote_ip=remote_ip,
            remote_read_port=remote_port,
            remote_write_port=remote_port,
            read_retries=read_retries,
            write_retries=write_retries,
            table_processing_mode=table_processing_mode,
        )
        self._psk = decode_key(key)
        self._remote_port = remote_port
        self._current: Optional[Session] = None
        self._handshake: Optional[Handshake] = None
        self._established = asyncio.Event()
        self._last_rx = 0.0
        self._last_tx = 0.0
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        logger.info(
            f"Connecting encrypted session to {self._remote_ip}:{self._remote_port}"
        )
        self.status = ConnectionStatus.INITIALIZING

        await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: self, remote_addr=(self._remote_ip, self._remote_port)
        )
        self._task = asyncio.create_task(self._run())

        try:
            await asyncio.wait_for(self._established.wait(), START_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("No encrypted session established yet, retrying")

        if self._heatpump.word_swap is None:
            await self.detect_word_swap()

    async def stop(self):
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._current = None
        self._handshake = None
        self._established.clear()
        await super().stop()

    def datagram_received(self, data: bytes, addr):
        if len(data) < 4:
            return

        if data[0] == TYPE_HANDSHAKE_RESPONSE:
            self._on_handshake_response(data)
            return

        if data[0] == TYPE_HANDSHAKE_REJECT:
            if self._handshake and (reason := self._handshake.rejected(data)):
                logger.error(f"Gateway rejected handshake: {reason}")
            return

        inner = self._current.open(data) if self._current else None
        if inner is None:
            logger.debug(f"Dropped invalid packet from {addr}")
            return

        self._last_rx = time.monotonic()
        if inner and inner[0] == INNER_TELEGRAM:
            super().datagram_received(inner[1:], addr)

    def _on_handshake_response(self, data: bytes) -> None:
        if self._handshake is None or not self._handshake.matches(data):
            return
        try:
            session = self._handshake.finish(data)
        except Exception as err:
            logger.debug(f"Invalid handshake response: {err}")
            return

        self._handshake = None
        self._current = session
        self._last_rx = time.monotonic()
        # The gateway uses the session once it sees data with the new keys
        self._send_inner(b"")
        self._established.set()
        logger.info("Encrypted session established")

    def _send_inner(self, inner: bytes) -> None:
        if self._current and self._transport:
            self._transport.sendto(self._current.seal(inner))
            self._last_tx = time.monotonic()

    def _send_request(self, data: bytes, port: int) -> None:
        if not self._current:
            logger.debug("No encrypted session, dropping request")
            return
        token = data[1]
        self._send_inner(
            bytes([INNER_REQUEST]) + struct.pack("<HB", MODBUS40, token) + data
        )

    def _initiate_handshake(self, now: float) -> None:
        assert self._transport, "Transport is closed"
        self._handshake = Handshake(self._psk)
        self._handshake.started = now
        self._transport.sendto(self._handshake.initiation())

    def _tick(self, now: float) -> None:
        if self._current and now - self._last_rx >= SESSION_TIMEOUT:
            logger.warning("Encrypted session lost, reconnecting")
            self._current = None
            self._established.clear()
            self.status = ConnectionStatus.LISTENING

        if self._current is None:
            if (
                self._handshake is None
                or now - self._handshake.started >= HANDSHAKE_RETRY
            ):
                self._initiate_handshake(now)
        elif now - self._last_tx >= KEEPALIVE_INTERVAL:
            self._send_inner(b"")

    async def _run(self) -> None:
        while True:
            self._tick(time.monotonic())
            await asyncio.sleep(TICK)

    def error_received(self, exc):
        # An unreachable gateway is handled by handshake retries
        logger.debug(exc)
