import asyncio
import binascii
import secrets
import struct
from unittest.mock import Mock

from noise.connection import NoiseConnection
import pytest

from nibe.coil import CoilData
from nibe.connection.mixins import ConnectionStatus
from nibe.connection.nibegw_noise import (
    HANDSHAKE_RETRY,
    INNER_REQUEST,
    INNER_TELEGRAM,
    KEEPALIVE_INTERVAL,
    PROLOGUE,
    PROTOCOL_NAME,
    SESSION_TIMEOUT,
    TYPE_HANDSHAKE_INITIATION,
    TYPE_HANDSHAKE_REJECT,
    TYPE_HANDSHAKE_RESPONSE,
    Handshake,
    NibeGWNoise,
    Session,
    decode_key,
    derive_key,
)
from nibe.heatpump import HeatPump, Model

KEY = "8ewZqBXo5fMYjnKkKUu1vyGQqEMNa4hsNTN3j2O/9N4="
PSK = decode_key(KEY)
GW_ADDR = ("127.0.0.1", 9999)


class FakeGateway:
    """Responder side of the protocol, mirroring the ESPHome implementation."""

    def __init__(self, psk: bytes = PSK) -> None:
        self.psk = psk
        self.session: Session | None = None

    def respond(self, packet: bytes) -> bytes | None:
        assert packet[:4] == bytes([TYPE_HANDSHAKE_INITIATION, 0, 0, 0])
        (remote_index,) = struct.unpack("<I", packet[4:8])
        noise = NoiseConnection.from_name(PROTOCOL_NAME)
        noise.set_as_responder()
        noise.set_psks(self.psk)
        noise.set_prologue(PROLOGUE)
        noise.start_handshake()
        try:
            noise.read_message(packet[8:])
        except Exception:
            return (
                bytes([TYPE_HANDSHAKE_REJECT, 0, 0, 0])
                + struct.pack("<I", remote_index)
                + b"Invalid key"
            )
        local_index = secrets.randbits(32)
        response = (
            bytes([TYPE_HANDSHAKE_RESPONSE, 0, 0, 0])
            + struct.pack("<II", local_index, remote_index)
            + bytes(noise.write_message())
        )
        proto = noise.noise_protocol
        self.session = Session(
            derive_key(proto.cipher_state_encrypt.k),
            derive_key(proto.cipher_state_decrypt.k),
            local_index,
            remote_index,
        )
        return response

    def telegram(self, frame: str) -> bytes:
        return self.session.seal(bytes([INNER_TELEGRAM]) + binascii.unhexlify(frame))


@pytest.fixture(name="heatpump")
async def fixture_heatpump() -> HeatPump:
    heatpump = HeatPump(Model.F1255)
    heatpump.word_swap = True
    await heatpump.initialize()
    return heatpump


@pytest.fixture(name="transport")
async def fixture_transport():
    return Mock()


@pytest.fixture(name="nibegw")
async def fixture_nibegw(heatpump: HeatPump, transport: Mock):
    nibegw = NibeGWNoise(heatpump, "127.0.0.1", KEY)
    nibegw.connection_made(transport)
    yield nibegw
    await nibegw.stop()


def _last_sent(transport: Mock) -> bytes:
    return transport.sendto.call_args.args[0]


def _connect(nibegw: NibeGWNoise, transport: Mock, gateway: FakeGateway, now=0.0):
    nibegw._tick(now)
    response = gateway.respond(_last_sent(transport))
    nibegw.datagram_received(response, GW_ADDR)
    # The client confirms the session with a keepalive
    assert gateway.session.open(_last_sent(transport)) == b""
    transport.sendto.reset_mock()


def test_decode_key():
    assert decode_key(KEY) == binascii.a2b_base64(KEY)
    with pytest.raises(ValueError):
        decode_key("YWJj")


def test_derive_key_vector():
    # First 32 bytes of ChaCha20-Poly1305(k=00..1f, n=2^64-2, zeros), checked
    # against libsodium which the gateway uses
    assert derive_key(bytes(range(32))).hex() == (
        "50173a052c3d81945afe3a058a090370a553adb5ea3e3247dce01d5256cbd6e0"
    )


def test_initiation_format():
    handshake = Handshake(PSK)
    packet = handshake.initiation()
    assert len(packet) == 56
    assert packet[:4] == b"\x01\x00\x00\x00"
    assert struct.unpack("<I", packet[4:8])[0] == handshake.sender_index


def _pair():
    a = Session(b"\x01" * 32, b"\x02" * 32, local_index=1, remote_index=2)
    b = Session(b"\x02" * 32, b"\x01" * 32, local_index=2, remote_index=1)
    return a, b


def test_session_format():
    a, b = _pair()
    packet = a.seal(b"abc")
    assert packet[:4] == b"\x04\x00\x00\x00"
    assert struct.unpack("<IQ", packet[4:16]) == (2, 0)
    assert len(packet) == 16 + 3 + 16
    assert b.open(packet) == b"abc"
    assert a.open(packet) is None  # wrong receiver index


def test_session_replay_and_reorder():
    a, b = _pair()
    p0 = a.seal(b"zero")
    p1 = a.seal(b"one")
    p2 = a.seal(b"two")

    assert b.open(p2) == b"two"
    assert b.open(p0) == b"zero"  # reordered within window
    assert b.open(p1) == b"one"
    assert b.open(p1) is None  # replay

    tampered = bytearray(a.seal(b"data"))
    tampered[-1] ^= 1
    assert b.open(bytes(tampered)) is None

    # The counter is the nonce, changing it breaks authentication
    packet = bytearray(a.seal(b"data"))
    packet[8] ^= 1
    assert b.open(bytes(packet)) is None


def test_session_window():
    a, b = _pair()
    old = a.seal(b"old")
    for _ in range(70):
        assert b.open(a.seal(b"x")) == b"x"
    assert b.open(old) is None  # outside the replay window


async def test_handshake_and_telegram(
    nibegw: NibeGWNoise, transport: Mock, heatpump: HeatPump
):
    gateway = FakeGateway()
    _connect(nibegw, transport, gateway)

    on_coil_update = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, on_coil_update)

    nibegw.datagram_received(gateway.telegram("5c00206a0698a9ad0c0000dc"), GW_ADDR)

    assert nibegw.status == ConnectionStatus.CONNECTED
    coil_data = on_coil_update.call_args.args[0]
    assert coil_data.coil.address == 43416
    assert coil_data.value == 3245


async def test_wrong_key_rejected_and_retried(
    nibegw: NibeGWNoise, transport: Mock, caplog
):
    gateway = FakeGateway(bytes(32))
    nibegw._tick(0)
    nibegw.datagram_received(gateway.respond(_last_sent(transport)), GW_ADDR)
    assert "Invalid key" in caplog.text
    assert nibegw._current is None
    transport.sendto.reset_mock()

    nibegw._tick(HANDSHAKE_RETRY - 1)
    transport.sendto.assert_not_called()
    nibegw._tick(HANDSHAKE_RETRY)
    assert _last_sent(transport)[0] == TYPE_HANDSHAKE_INITIATION


async def test_ignores_unrelated_response(nibegw: NibeGWNoise, transport: Mock):
    nibegw._tick(0)
    other = Handshake(PSK)
    response = FakeGateway().respond(other.initiation())
    nibegw.datagram_received(response, GW_ADDR)
    assert nibegw._current is None


async def test_ignores_forged_data(nibegw: NibeGWNoise, transport: Mock):
    gateway = FakeGateway()
    _connect(nibegw, transport, gateway)
    forged = Session(b"\x05" * 32, b"\x06" * 32, 0, nibegw._current.local_index)
    nibegw.datagram_received(forged.seal(bytes([INNER_TELEGRAM]) + b"\x5c"), GW_ADDR)
    assert nibegw.status != ConnectionStatus.CONNECTED


async def test_keepalive(nibegw: NibeGWNoise, transport: Mock):
    gateway = FakeGateway()
    _connect(nibegw, transport, gateway)
    nibegw._last_rx = nibegw._last_tx = 100
    nibegw._tick(100 + KEEPALIVE_INTERVAL - 1)
    transport.sendto.assert_not_called()
    nibegw._tick(100 + KEEPALIVE_INTERVAL)
    assert gateway.session.open(_last_sent(transport)) == b""


async def test_lost_session_reconnects(nibegw: NibeGWNoise, transport: Mock):
    gateway = FakeGateway()
    _connect(nibegw, transport, gateway)
    nibegw._last_rx = 100
    nibegw._tick(100 + SESSION_TIMEOUT)
    assert nibegw._current is None
    assert nibegw.status == ConnectionStatus.LISTENING
    assert _last_sent(transport)[0] == TYPE_HANDSHAKE_INITIATION

    # A new session replaces the lost one
    nibegw.datagram_received(gateway.respond(_last_sent(transport)), GW_ADDR)
    assert nibegw._current is not None


async def test_read_coil(nibegw: NibeGWNoise, transport: Mock, heatpump: HeatPump):
    gateway = FakeGateway()
    _connect(nibegw, transport, gateway)
    coil = heatpump.get_coil_by_address(43416)

    async def respond():
        await asyncio.sleep(0)
        request = gateway.session.open(_last_sent(transport))
        assert request[0] == INNER_REQUEST
        assert struct.unpack("<HB", request[1:4]) == (0x20, 0x69)
        assert request[4:] == binascii.unhexlify("c0690298a99a")
        nibegw.datagram_received(gateway.telegram("5c00206a0698a9ad0c0000dc"), GW_ADDR)

    asyncio.create_task(respond())
    data = await nibegw.read_coil(coil)
    assert data.value == 3245


async def test_write_coil(nibegw: NibeGWNoise, transport: Mock, heatpump: HeatPump):
    gateway = FakeGateway()
    _connect(nibegw, transport, gateway)
    coil = heatpump.get_coil_by_address(48132)

    async def respond():
        await asyncio.sleep(0)
        request = gateway.session.open(_last_sent(transport))
        assert struct.unpack("<HB", request[1:4]) == (0x20, 0x6B)
        assert request[4:] == binascii.unhexlify("c06b0604bc0400000011")
        nibegw.datagram_received(gateway.telegram("5c00206c01014c"), GW_ADDR)

    asyncio.create_task(respond())
    await nibegw.write_coil(CoilData(coil, "One time increase"))


async def test_request_dropped_without_session(nibegw: NibeGWNoise, transport: Mock):
    nibegw._send_request(b"\xc0\x69\x02\x98\xa9\x00", 9999)
    transport.sendto.assert_not_called()
