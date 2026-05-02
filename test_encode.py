import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio

# 1. 加载模型（选择 24kHz / 48kHz 都可以）
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 1.5 / 3 / 6 / 12 kbps 可选
model.eval()

# 2. 读入音频
wav, sr = torchaudio.load(r"E:\\phd\\DL\\p2\\final_proj_wsj0\\wsj0\\si_dt_05\\22g\\22ga010a.wav")
# 3. 转采样 + 通道处理（EnCodec要求）
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)  # [B, C, T]

# 4. 编码
with torch.no_grad():
    encoded_frames = model.encode(wav)

# 5. 解码
with torch.no_grad():
    decoded = model.decode(encoded_frames)

# 6. 保存结果
torchaudio.save("reconstructed.wav", decoded[0], model.sample_rate)

print("Done! reconstructed.wav generated.")