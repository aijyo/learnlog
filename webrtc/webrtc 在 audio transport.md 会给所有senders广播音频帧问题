在同一个 PeerConnection / VoiceEngine / AudioTransport 里，AudioTransportImpl::SendProcessedData() 会把 ADM 采到并经 APM 处理后的 10ms 帧广播给所有已注册的 audio senders。这些 sender 是在媒体层创建/启用发送流时注册进去的，因此一旦 B 的 sender 被创建且处于发送态，它也会被放进 audio_senders_，从而收到 ADM 帧。这是当前语音引擎的单源广播设计（fan-out），不是你用法的问题。

关键点复盘

WebRtcVoiceMediaChannel::AddSendStream() 创建 WebRtcAudioSendStream 时，会把该 sender 注册进 AudioTransportImpl（通常通过 AddAudioSender(this) 一类调用）。

AudioTransportImpl::SendProcessedData() 每 10ms 遍历 audio_senders_：

// comments in English
if (audio_senders_.empty()) return;
for (auto it = ++audio_senders_.begin(); it != audio_senders_.end(); ++it) {
  (*it)->SendAudioData(copy_of_frame);
}
(*audio_senders_.begin())->SendAudioData(original_frame);


也就是说，只要 sender 在列表里且没有额外的“屏蔽”逻辑，它就会收到来自 ADM 的帧。

在上游未打补丁的前提下，想在“同一个 PC”里既保留 ADM 采麦（A），又同时发送另一条独立外部源（B），并且不让 B 收到 ADM 的广播帧，基本做不到（设计如此）。

可行的工程方案
方案 1（强烈推荐）：拆分为两个 PeerConnection

PC_Mic：使用带 ADM 的工厂，仅承载 A（麦克风）。

PC_Sys：使用不采集的 ADM（Dummy/Null ADM，或 TestADM 不提供 Capturer），仅承载 B（外部源）。

这样两个 PC 各有自己的 VoiceEngine / AudioTransport：

PC_Mic 的 AudioTransportImpl 有 ADM 回调，会广播给它自己的 audio_senders_（只有 A）。

PC_Sys 的 AudioTransportImpl 没有任何 ADM 采集帧（因为 Dummy ADM 不会上报），所以不会发生“ADM → B”的误喂；B 的音频从你的 ExternalAudioSource → AudioRtpSender 直达它自己的 send stream。

简化骨架（仅展示“两个工厂/两个 PC”的要点；注释英文）：

// All comments in English.
// Build two factories (or reuse one factory but two PCs), isolating ADM fan-out.
auto tqf = webrtc::CreateDefaultTaskQueueFactory();

// Factory/PC for MIC (with real ADM)
auto real_adm = webrtc::AudioDeviceModule::Create(
    webrtc::AudioDeviceModule::kPlatformDefaultAudio, tqf.get());
webrtc::PeerConnectionFactoryDependencies depsA;
depsA.audio_device_module = real_adm;
// ... set threads/encoders/decoders/APM ...
auto factoryA = webrtc::CreateModularPeerConnectionFactory(std::move(depsA));
auto pcMic = factoryA->CreatePeerConnection(cfgA, nullptr, nullptr, obsA);

// Factory/PC for SYS (with dummy/no capture ADM)
auto dummy_adm = /* a dummy ADM that never captures */;
webrtc::PeerConnectionFactoryDependencies depsB;
depsB.audio_device_module = dummy_adm;
// ... threads/encoders/decoders/APM ...
auto factoryB = webrtc::CreateModularPeerConnectionFactory(std::move(depsB));
auto pcSys = factoryB->CreatePeerConnection(cfgB, nullptr, nullptr, obsB);

// PC_Mic: add ADM-backed track A
auto mic_src = factoryA->CreateAudioSource(cricket::AudioOptions{});
auto mic_trk = factoryA->CreateAudioTrack("mic_A", mic_src);
pcMic->AddTransceiver(mic_trk, {.direction = webrtc::RtpTransceiverDirection::kSendOnly});

// PC_Sys: add external-push track B (never touches ADM)
auto ext_src = rtc::make_ref_counted<ExternalAudioSource>();
auto sys_trk = factoryB->CreateAudioTrack("sys_B", ext_src);
pcSys->AddTransceiver(sys_trk, {.direction = webrtc::RtpTransceiverDirection::kSendOnly});


Dummy/Null ADM 怎么做？

最简单：用 test::TestAudioDeviceModule::Create(/*capturer=*/nullptr, /*renderer=*/nullptr)（或传只 renderer），让它不提供录音；这样 AudioTransportImpl::SendProcessedData() 根本不会被调用。

或者写一个最小的自定义 ADM，RecordingIsAvailable=false，InitRecording/StartRecording 返回错误/不触发回调。

代价：要做两路信令（两套 SDP/ICE）。利：彻底隔离，稳定可靠。

方案 2：两条轨都走外部源（绕开 ADM）

如果你可以接受麦克风也不用 ADM，那就把 A=Mic 和 B=System 都用 WASAPI 自采，各自推给 ExternalAudioSource，在同一个 PC 内也能避免 ADM fan-out（因为你不给 PC 注入会采集的 ADM）。

你仍可把 APM（AEC/AGC/NS）留在 PeerConnectionFactoryDependencies.audio_processing 里，但注意 AEC 的反向参考（render path）接线。

这条路对接 AEC 比较麻烦，但避免了两 PC 的信令成本。

方案 3（临时过渡）：占位但不发、绑定后再切发

这只能缩短 ADM 误喂的窗口，不能从根上避免广播：

先 AddTransceiver(MEDIA_TYPE_AUDIO, {.direction=kInactive}) 占位；

sender->SetTrack(external_track)；

再 transceiver->SetDirection(kSendOnly)。
一旦进入发送态且 sender 注册进 audio_senders_，ADM 有帧时仍会广播。

方案 4（需要改源码）：给 sender 加“拒收 ADM 帧”的开关

如果你允许 patch WebRTC，可以在 WebRtcAudioSendStream 增加一个标记，例如：

bool accepts_transport_input_ = true;

当 AudioRtpSender 绑定的是外部源时，调用 send_stream->SetAcceptsTransportInput(false);

AudioTransportImpl::SendProcessedData() 遍历时跳过 !accepts_transport_input_ 的 sender（或在 WebRtcAudioSendStream::SendAudioData() 内部直接忽略）。

这是最干净的单 PC 方案，但需要维护私有补丁。

小结

你观察的广播行为是当前语音引擎的既定设计（单 ADM 源 → 所有 sender）。

不想让 B 收到 ADM 帧，工程上最佳是两个 PC 隔离（或两条都走外部源、不用 ADM）。

在同一个 PC 内，不修改源码基本无法彻底规避。

若能打补丁，可在 WebRtcAudioSendStream/AudioTransportImpl 上加“过滤 ADM 帧”的开关，区分“来自 ADM 的输入”与“来自外部源的输入”
