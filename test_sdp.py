import asyncio
from aiortc import RTCPeerConnection, RTCIceServer, RTCConfiguration

async def main():
    config = RTCConfiguration([
        RTCIceServer(
            urls=["turns:global.relay.metered.ca:443?transport=tcp"],
            username="test",
            credential="password"
        )
    ])
    pc = RTCPeerConnection(config)
    channel = pc.createDataChannel("chat")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    print("SDP:\n", pc.localDescription.sdp)
    await pc.close()

if __name__ == "__main__":
    asyncio.run(main())
