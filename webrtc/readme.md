WebRTC 三大线程模型（组件&调用线程分布图）
                                  [你的应用线程 / UI线程]
                                             │
         ┌───────────────────────────────────┴───────────────────────────────────┐
         │                                调用 WebRTC API                        │
         ▼                                                                      ▼
   ┌────────────┐                                                      ┌────────────────┐
   │ Signaling  │◀───控制流（Offer/Answer、AddTrack、ICE）───┐        │   Worker Thread │
   │  Thread    │                                              │        │（音视频编解码等）│
   └────────────┘                                              │        └────────────────┘
         │                                                    │                 ▲
         ▼                                                    │                 │
[PeerConnection / PeerConnectionFactory]                      │         ┌───────────────┐
         │                                                    │         │ VideoEncoder  │
         │                                                    │         │ AudioEncoder  │
         │                                                    │         │ MediaSource   │
         ▼                                                    │         └───────────────┘
  ┌────────────────────┐                                      │
  │ PeerConnectionImpl │                                      │
  └────────────────────┘                                      │
         │                                                    │
         ▼                                                    │
 ┌─────────────────────────────┐                              │
 │   MediaController / Session │◀────────数据轨 & 媒体流控制──┘
 └─────────────────────────────┘
         │
         ▼
 ┌─────────────┐
 │ Network     │◀───────────────RTP / RTCP / STUN / TURN / DTLS / ICE────────────┐
 │ Thread      │                                                                  │
 └─────────────┘                                                                  ▼
                                                                              Internet


🧠 各线程的职责简要回顾
线程	主要职责
Signaling	信令控制：SDP协商、ICE candidate处理、Track添加/移除、回调事件（如 OnIceCandidate）
Worker	编解码、回调 OnFrame、音频处理、MediaEngine、帧处理、FrameTransformer、音视频采集渲染等
Network	RTP/RTCP 收发，socket I/O，ICE连接，DTLS握手、NAT打洞、STUN/TURN

🔧 开发者常见线程操作行为
操作	所在线程
peer_connection->CreateOffer() / SetRemoteDesc	Signaling Thread
peer_connection->AddTrack()	Signaling Thread
PeerConnectionObserver::OnTrack 回调	Signaling Thread
AudioSinkInterface::OnData()	Worker Thread
VideoSinkInterface::OnFrame()	Worker Thread
IceTransport::OnPacketReceived()	Network Thread
自定义 Encoder/Decoder	Worker Thread
