import asyncio
import binascii
import secrets
import struct
from unittest.mock import Mock

from noise.connection import NoiseConnection
import pytest

from nibe.coil import CoilData
from nibe.connection.mixins import ConnectionStatus
from nibe.connection.nibegw import NibeGWProtocol
from nibe.connection.nibegw_noise import (
    PROLOGUE,
    PROTOCOL_NAME,
    TYPE_HANDSHAKE_INITIATION,
    TYPE_HANDSHAKE_RESPONSE,
    Handshake,
    NibeGWNoiseTransport,
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
            return None  # dropped
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
        return self.session.seal(binascii.unhexlify(frame))


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
async def fixture_nibegw(heatpump: HeatPump):
    nibegw = NibeGWProtocol(heatpump)
    yield nibegw
    await nibegw.stop()


@pytest.fixture(name="noise")
async def fixture_noise(nibegw: NibeGWProtocol, transport: Mock):
    noise = NibeGWNoiseTransport(nibegw, "127.0.0.1", KEY)
    noise.connection_made(transport)
    return noise


def _last_sent(transport: Mock) -> bytes:
    return transport.sendto.call_args.args[0]


def _connect(noise: NibeGWNoiseTransport, transport: Mock, gateway: FakeGateway):
    # connection_made has sent the handshake initiation
    noise.datagram_received(gateway.respond(_last_sent(transport)), GW_ADDR)
    # The client confirms the session with a keepalive
    assert gateway.session.open(_last_sent(transport)) == b""
    transport.sendto.reset_mock()


def _pair():
    a = Session(b"\x01" * 32, b"\x02" * 32, local_index=1, remote_index=2)
    b = Session(b"\x02" * 32, b"\x01" * 32, local_index=2, remote_index=1)
    return a, b


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


def test_session_format():
    a, b = _pair()
    packet = a.seal(b"abc")
    assert packet[:4] == b"\x04\x00\x00\x00"
    assert struct.unpack("<IQ", packet[4:16]) == (2, 0)
    assert len(packet) == 16 + 3 + 16
    assert b.open(packet) == b"abc"
    assert a.open(packet) is None  # wrong receiver index


def test_session_counters():
    a, b = _pair()
    p0 = a.seal(b"zero")
    p1 = a.seal(b"one")

    assert b.open(p1) == b"one"
    assert b.open(p0) is None  # older counter
    assert b.open(p1) is None  # replay

    tampered = bytearray(a.seal(b"data"))
    tampered[-1] ^= 1
    assert b.open(bytes(tampered)) is None

    # The counter is the nonce, changing it breaks authentication
    packet = bytearray(a.seal(b"data"))
    packet[8] ^= 1
    assert b.open(bytes(packet)) is None


async def test_handshake_and_telegram(
    nibegw: NibeGWProtocol,
    noise: NibeGWNoiseTransport,
    transport: Mock,
    heatpump: HeatPump,
):
    gateway = FakeGateway()
    _connect(noise, transport, gateway)

    on_coil_update = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, on_coil_update)

    noise.datagram_received(gateway.telegram("5c00206a0698a9ad0c0000dc"), GW_ADDR)

    assert nibegw.status == ConnectionStatus.CONNECTED
    coil_data = on_coil_update.call_args.args[0]
    assert coil_data.coil.address == 43416
    assert coil_data.value == 3245


async def test_handshake_sent_and_retried(
    nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport, transport: Mock
):
    assert _last_sent(transport)[0] == TYPE_HANDSHAKE_INITIATION
    assert noise._handshake_timer.when() > asyncio.get_running_loop().time()

    # Wrong key, the gateway does not answer
    gateway = FakeGateway(bytes(32))
    assert gateway.respond(_last_sent(transport)) is None
    transport.sendto.reset_mock()

    # The retry timer sends a new initiation
    noise._start_handshake()
    assert _last_sent(transport)[0] == TYPE_HANDSHAKE_INITIATION
    assert noise._session is None


async def test_timers_after_handshake(
    nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport, transport: Mock
):
    _connect(noise, transport, FakeGateway())
    assert noise._handshake_timer.cancelled()
    assert not noise._keepalive_timer.cancelled()
    assert not noise._session_timer.cancelled()

    await nibegw.stop()
    assert noise._keepalive_timer.cancelled()
    assert noise._session_timer.cancelled()


async def test_ignores_unrelated_response(
    nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport, transport: Mock
):
    response = FakeGateway().respond(Handshake(PSK).initiation())
    noise.datagram_received(response, GW_ADDR)
    assert noise._session is None


async def test_ignores_forged_data(
    nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport, transport: Mock
):
    gateway = FakeGateway()
    _connect(noise, transport, gateway)
    forged = Session(b"\x05" * 32, b"\x06" * 32, 0, noise._session.local_index)
    noise.datagram_received(forged.seal(b"\x5c"), GW_ADDR)
    assert nibegw.status != ConnectionStatus.CONNECTED


async def test_keepalive(
    nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport, transport: Mock
):
    gateway = FakeGateway()
    _connect(noise, transport, gateway)
    noise._send_keepalive()
    assert gateway.session.open(_last_sent(transport)) == b""


async def test_lost_session_reconnects(
    nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport, transport: Mock
):
    gateway = FakeGateway()
    _connect(noise, transport, gateway)
    noise._session_lost()
    assert noise._session is None
    assert noise._keepalive_timer.cancelled()
    assert _last_sent(transport)[0] == TYPE_HANDSHAKE_INITIATION

    noise.datagram_received(gateway.respond(_last_sent(transport)), GW_ADDR)
    assert noise._session is not None


async def test_read_coil(
    nibegw: NibeGWProtocol,
    noise: NibeGWNoiseTransport,
    transport: Mock,
    heatpump: HeatPump,
):
    gateway = FakeGateway()
    _connect(noise, transport, gateway)
    coil = heatpump.get_coil_by_address(43416)

    async def respond():
        await asyncio.sleep(0)
        request = gateway.session.open(_last_sent(transport))
        assert struct.unpack("<HB", request[:3]) == (0x20, 0x69)
        assert request[3:] == binascii.unhexlify("c0690298a99a")
        noise.datagram_received(gateway.telegram("5c00206a0698a9ad0c0000dc"), GW_ADDR)

    asyncio.create_task(respond())
    data = await nibegw.read_coil(coil)
    assert data.value == 3245


async def test_write_coil(
    nibegw: NibeGWProtocol,
    noise: NibeGWNoiseTransport,
    transport: Mock,
    heatpump: HeatPump,
):
    gateway = FakeGateway()
    _connect(noise, transport, gateway)
    coil = heatpump.get_coil_by_address(48132)

    async def respond():
        await asyncio.sleep(0)
        request = gateway.session.open(_last_sent(transport))
        assert struct.unpack("<HB", request[:3]) == (0x20, 0x6B)
        assert request[3:] == binascii.unhexlify("c06b0604bc0400000011")
        noise.datagram_received(gateway.telegram("5c00206c01014c"), GW_ADDR)

    asyncio.create_task(respond())
    await nibegw.write_coil(CoilData(coil, "One time increase"))


async def test_request_dropped_without_session(
    nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport, transport: Mock
):
    transport.sendto.reset_mock()  # the handshake initiation
    noise.send_request(0x20, b"\xc0\x69\x02\x98\xa9\x00")
    transport.sendto.assert_not_called()


async def test_transport_protocol(nibegw: NibeGWProtocol, noise: NibeGWNoiseTransport):
    assert nibegw._transport is noise
    assert noise.get_protocol() is nibegw
    other = NibeGWProtocol(nibegw._heatpump)
    noise.set_protocol(other)
    assert noise.get_protocol() is other
