# WebRTC 模块线程结构图

本结构图展示了 WebRTC 的 3 个核心线程及其对应的模块与数据流。

```
                   ┌──────────────────────────────────┐
                   │    应用 / UI线程     │
                   └─────────────────────────┘
                            │
         ┌──────────────────────────────────────┐
         │            Signaling Thread          │
         │--------------------------------------│
         │ - CreateOffer / SetRemoteDescription │
         │ - AddTrack / RemoveTrack             │
         │ - ICE 候选协商                       │
         │ - PeerConnectionObserver 回调        │
         └───────────────────────────────────┘
                            │
                            ▼
                     ┌──────────────────────────────┐
                     │ PeerConnection │
                     └──────────────────────────┘
                            │
   ┌──────────────────────────────────────┐
   ▼                        ▼                        ▼
Worker Thread         Network Thread           Media Transport
-------------         --------------           ----------------
- 编解码 (Opus/H264)  - STUN/TURN/ICE 收发     - RTP/RTCP 发送
- SRTP 加密           - DTLS 握手              - 音视频传输
- 音视频处理         - socket 管理            - Bandwidth control
- AudioDeviceModule   - UDP/TCP                - Congestion control
- OnData / OnFrame

   ▲                        ▲
   └───────────────────────────────────┘
                  ▼
          外部设备/网络
```

---

## 各线程模块职责

| 线程类型      | 职责描述                                               |
| --------- | -------------------------------------------------- |
| Signaling | 信令协商，控制流操作，PeerConnection API 调用，所有控制类回调发生线程       |
| Worker    | 编解码、音视频处理、Frame 回调、AudioDevice 接入、MediaEngine 初始化等 |
| Network   | 网络 I/O，ICE/DTLS/STUN 等握手与收发包，socket 管理             |

---

## 常见模块与线程分布

| 模块                            | 所属线程               |
| ----------------------------- | ------------------ |
| `PeerConnection::CreateOffer` | Signaling Thread   |
| `AddTrack / OnTrack`          | Signaling Thread   |
| `VideoSinkInterface::OnFrame` | Worker Thread      |
| `AudioSinkInterface::OnData`  | Worker Thread      |
| `OnIceCandidate` 回调           | Signaling Thread   |
| RTP/RTCP/STUN socket I/O      | Network Thread     |
| `AudioDeviceModule::Create()` | Worker Thread (推荐) |

---

## 线程间调度建议

* 如果你从 UI 调用 PeerConnection API，请使用 `rtc::Thread::PostTask()` 转发到 Signaling Thread。
* 不要在 Worker 或 Network Thread 执行耗时任务。
* 使用 `RTC_DCHECK_RUN_ON()` 验证线程一致性。
