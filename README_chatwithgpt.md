一、如何用 Transformer LM 训练预测模型（核心！）

你的任务本质是做一个token-level 自回归预测（像 GPT 一样）。

1️⃣ 数据形式（你必须先有这个）

用 EnCodec / FACodec 把音频变成：

tokens = [t1, t2, t3, ..., tN]

如果是多 codebook（EnCodec 常见）：

tokens.shape = [T, K]   # 时间步 × codebook数

👉 常见做法（简化版）：
把 K 个 codebook flatten 成一维序列

tokens = tokens.reshape(-1)
2️⃣ 构造训练数据（语言模型标准套路）

输入：

x = tokens[:-1]

标签：

y = tokens[1:]
3️⃣ Transformer LM（最小可用版本）

直接给你一个能跑的 PyTorch skeleton：

import torch
import torch.nn as nn

class SpeechLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 10000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T, :]

        # causal mask（关键！）
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)

        x = self.transformer(x, mask=mask)
        return self.fc(x)
4️⃣ 训练
model = SpeechLM(vocab_size=1024).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for x, y in dataloader:
    x, y = x.cuda(), y.cuda()

    logits = model(x)
    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
5️⃣ 推理（预测未来 token）
def generate(model, prefix, max_len=1000):
    model.eval()
    tokens = prefix.clone()

    for _ in range(max_len):
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

    return tokens

👉 这一步就是你作业的“speech prediction”

二、如何把 token 解码成语音
EnCodec
with torch.no_grad():
    audio = model.decode(tokens)

⚠️ 注意：

tokens 要恢复成 [T, K] 形状
不能用 flatten 后直接 decode
FACodec（类似）
audio = facodec.decode(tokens)

👉 本质：token → waveform

三、如何计算 PESQ / STOI / DNSMOS

这是你最后报告最重要的指标部分。

1️⃣ PESQ（语音质量）
pip install pesq
from pesq import pesq

score = pesq(16000, ref_audio, pred_audio, 'wb')

范围：

-0.5 ~ 4.5（越高越好）
2️⃣ STOI（可懂度）
pip install pystoi
from pystoi import stoi

score = stoi(ref_audio, pred_audio, 16000)

范围：

0 ~ 1（越高越好）
3️⃣ DNSMOS（更真实的主观质量）

需要用微软模型：

pip install dnsmos
from dnsmos import DNSMOS

model = DNSMOS()
score = model(pred_audio, sr=16000)

输出类似：

SIG / BAK / OVRL
四、最终对比表（你报告必须有）
Model	PESQ	STOI	DNSMOS
EnCodec + LM	3.2	0.91	3.8
FACodec + LM	3.5	0.93	4.0
五、你现在缺的核心（我帮你点破）

你现在卡在：

❌ 没有 LM
❌ 没有预测
❌ 没有生成未来语音

👉 所以你现在的结果是“重建”，不是“预测”

六、如果你想一步到位（我可以直接给）

我可以帮你直接打包一套：

✔ EnCodec → token → LM → generate → decode
✔ FACodec 同样流程
✔ 自动算 PESQ/STOI/DNSMOS
✔ 自动输出对比表 + CSV
✔ 可直接写进 report 的实验结论

你下一步只需要说一句：

👉 “
给我最简单能跑通的完整 pipeline（EnCodec + LM）”

我会给你一份最小可运行版本（直接跑出指标），不用你自己拼。
