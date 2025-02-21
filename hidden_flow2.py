import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import os
import re

import sys
sys.path.append('/huyang/hidden_flow2/flash-linear-attention')
from fla.layers import RWKV7Attention


scale_factor = 0.5

@dataclass
class Config:
    dim_in: int = 4096

    n_embd: int = 512
    n_head: int = 16
    n_layer: int = 6
    dropout: float = 0.1
    bias: bool = True
    
    # 时间编码参数
    time_emb_dim: int = 256
    
    # 训练参数
    learning_rate: float = 3e-4
    weight_decay: float = 0.001
    grad_clip: float = 1.0
    batch_size: int = 8  # 总批次大小
    num_epochs: int = 10000
    wiki_batch_size: int = 8  # 每批处理的wiki文章数量
    window_size: int = 2048

    sequence_length: int = window_size+2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据集配置
    data_dir: str = 'data/tinyshakespeare'
    chunk_size: int = 10000


class WikiDataset(Dataset):
    def __init__(self, config):
        self.config = config
        
        # 加载完整的 wikitext 数据集
        self.dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir="/huyang/hidden_flow2/cache")
            
        def clean_text(text):
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
            
        # 将数据集转换为文本块
        self.text_chunks = []
        chunk_size = config.chunk_size
        current_chunk = ""
        
        # 处理完整数据集
        for item in self.dataset:
            text = clean_text(item['text'])
            if not text.strip():  # 跳过空文本
                continue
                
            current_chunk += text
            
            if len(current_chunk) >= chunk_size:
                self.text_chunks.append(current_chunk[:chunk_size])
                current_chunk = current_chunk[chunk_size:]

        if current_chunk:  # 添加最后一个不完整的块
            self.text_chunks.append(current_chunk)

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
        
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-7B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        # 缓存管理
        self.cached_text = ""
        self.min_length = config.window_size + 2
        self.current_chunk_idx = 0

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        # 如果缓存的文本为空，添加新的文本块
        if not self.cached_text:
            self.cached_text = "\n" + "-"*50 + "\n"
            
        # 如果缓存文本长度不够，继续添加新文本
        while len(self.cached_text) < self.min_length:
            text = self.text_chunks[self.current_chunk_idx]
            if text.strip():
                self.cached_text += text + "\n" + "-"*3 + "\n"
            self.current_chunk_idx = (self.current_chunk_idx + 1) % len(self.text_chunks)
        
        # 对文本进行编码
        inputs = self.tokenizer(
            self.cached_text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.config.sequence_length
        )
        
        # 更新缓存
        encoded_length = len(inputs['input_ids'][0])
        decoded_text = self.tokenizer.decode(inputs['input_ids'][0][:encoded_length//2])
        self.cached_text = self.cached_text[len(decoded_text):]
        
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].squeeze(0)
            attention_mask = inputs['attention_mask'].squeeze(0)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            
            hidden_states = hidden_states.float()
        
        return hidden_states




class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dim = config.time_emb_dim
        
        position = torch.arange(self.dim//2).float()
        div_term = torch.exp(position * (-math.log(10000.0) / (self.dim//2)))
        self.register_buffer('div_term', div_term)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, 4 * self.n_embd)
        )
  
    def forward(self, t):
        # t shape: [batch_size]
        t = t.view(-1, 1)  # [batch_size, 1]
        
        emb = torch.zeros(t.shape[0], self.dim, device=t.device)
        omega = t * self.div_term
        emb[:, 0::2] = torch.sin(omega)
        emb[:, 1::2] = torch.cos(omega)
        
        # [batch_size, 4 * n_embd]
        return self.mlp(emb)

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
    def forward(self, x, scale, shift):
        x = self.norm(x)
        return x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        norm_x = x.norm(2, -1, keepdim=True)
        rms_x = norm_x * x.size(-1) ** (-0.5) 
        x_normed = x / (rms_x + self.eps)
        return self.weight * x_normed



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = nn.Dropout(config.dropout)
        
        # 将QKV合并为一个线性层以提高效率
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 添加RMS归一化
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # 添加RoPE位置编码
        self.max_seq_len = 2048
        
        # 直接在初始化时构建RoPE缓存
        theta = 10000.0
        pos = torch.arange(self.max_seq_len).unsqueeze(1)
        dim = torch.arange(0, self.head_dim, 2).float()
        freq = pos / (theta ** (dim / self.head_dim))
        
        emb = torch.zeros(self.max_seq_len, self.head_dim)
        emb[:, 0::2] = freq.sin()
        emb[:, 1::2] = freq.cos()
        self.register_buffer('rope_cache', emb)

    def _apply_rope(self, x, seq_len):
        # x: [batch, heads, seq_len, head_dim]
        rope = self.rope_cache[:seq_len]  # [seq_len, head_dim]
        
        # 将x分成实部和虚部
        x_real, x_imag = x[..., ::2], x[..., 1::2]
        
        # 获取旋转角度
        cos = rope[:, ::2].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = rope[:, 1::2].unsqueeze(0).unsqueeze(0)
        
        # 应用复数旋转
        x_out = torch.zeros_like(x)
        x_out[..., ::2] = x_real * cos - x_imag * sin
        x_out[..., 1::2] = x_real * sin + x_imag * cos
        
        return x_out

    def forward(self, x, v_first=None):
        B, L, C = x.shape  # batch_size, sequence_length, n_embd
        
        # 计算QKV
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, self.n_head, self.head_dim).transpose(1, 2), qkv)
        
        # 应用RMS归一化
        # q = self.q_norm(q)
        # k = self.k_norm(k)
        
        # 应用RoPE位置编码
        q = self._apply_rope(q, L)
        k = self._apply_rope(k, L)
        
        # 计算注意力分数
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 因果mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        
        # 应用softmax并dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out, None  # 返回None以匹配RWKV的接口

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.adaln1 = AdaLN(config.n_embd)
        
        # 根据配置选择注意力机制
        attn_type = os.environ.get('hucfg_attention', 'attn')  # 默认使用RWKV
        if attn_type == 'rwkv':
            self.attn = RWKV7Attention(
                hidden_size=config.n_embd,
                layer_idx=layer_idx
            )
        else:  # causal
            self.attn = CausalAttention(config)
            
        self.adaln2 = AdaLN(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, time_emb, v_first=None):
        h = x
        time_emb = time_emb.view(time_emb.size(0), 4, -1)
        scale, shift = time_emb[:, 0], time_emb[:, 1]
        
        h = self.adaln1(h, scale, shift)
        h = self.mlp(h)
        
        # 修改这里：解包注意力层的返回值
        attn_out, _ = self.attn(h, v_first)
        h = attn_out
        
        h = h + x
        
        # 第二组操作
        x2 = h
        scale, shift = time_emb[:, 2], time_emb[:, 3]
        
        h = self.adaln2(h, scale, shift)
        h = self.mlp(h)
        h = h + x2
        
        return h

class FeatureGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_mlp = SinusoidalTimeEncoding(config)
        self.blocks = nn.ModuleList([
            Block(config, layer_idx=i) 
            for i in range(config.n_layer)
        ])
        
        self.apply(self._init_weights)
        print(f"FeatureGPT parameters: {sum(p.numel() for p in self.parameters()):,}")

        self.in_proj = nn.Linear(config.dim_in, config.n_embd)
        self.in_ln = nn.LayerNorm(config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.dim_in)

        self.adaln_final = AdaLN(config.n_embd)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, t):
        x = self.in_proj(x)
        x = self.in_ln(x)
        
        time_emb = self.time_mlp(t)
        v_first = torch.empty_like(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x, time_emb, v_first=v_first)
        
        time_emb = time_emb.view(time_emb.size(0), 4, -1)
        scale, shift = time_emb[:, 0], time_emb[:, 1]
        x = self.adaln_final(x, scale, shift)
        
        x = self.out_proj(x)
        
        return x

def create_training_batch(features_list, config):
    """
    Args:
        features_list: list of tensors, each [seq_len, dim]
        config: Config object
    Returns:
        tuple of (interpolated, target, time) tensors
    """
    total_batches = config.batch_size // len(features_list)
    interpolated_list = []
    target_list = []
    time_list = []
    
    for features in features_list:
        seq_len = features.size(0)
        
        for _ in range(total_batches):
            # 随机选择窗口起始位置
            idx = torch.randint(0, seq_len - config.window_size - 1, (1,)).item()
            
            # 获取当前和未来序列
            current_seq = features[idx:idx+config.window_size]  # [window_size, dim]
            
            
            # 生成随机时间步长
            # t = torch.rand(1, device=features.device)
            t = torch.sigmoid(torch.randn(1, device=features.device))
            
            # 计算插值和目标
            use_randn = os.environ.get('hucfg_use_randn', 'True')
            if use_randn=='True':
                current_seq = current_seq * scale_factor

                randn_seq = torch.randn_like(current_seq)
                # import ipdb; ipdb.set_trace()
                interpolated = (t) * current_seq + (1-t) * randn_seq  # [window_size, dim]
                target = current_seq - randn_seq  # [window_size, dim]
            else:
                future_seq = features[idx+1:idx+config.window_size+1]  # [window_size, dim]
                interpolated = (1-t) * current_seq + t * future_seq  # [window_size, dim]
                target = future_seq - current_seq  # [window_size, dim]
            

            
            interpolated_list.append(interpolated)
            target_list.append(target)
            time_list.append(t)
    
    return (
        torch.stack(interpolated_list),  # [batch_size, window_size, dim]
        torch.stack(target_list),        # [batch_size, window_size, dim]
        torch.cat(time_list)             # [batch_size]
    )



def train_epoch(model, dataloader, optimizer, config, base_model, tokenizer):
    model.train()
    total_loss = 0
    num_batches = 0
    
    grad_norms = []
    pred_norms = []
    target_norms = []

    eval_prompts = [
        "克莱恩默默 ",
        "玛丽太太",
        "而是",
        "Ancient Rome",
        "architecture made extensive use "
    ]

    for batch_idx, hidden_states in enumerate(dataloader):
        hidden_states = [h.to(config.device) for h in hidden_states]
        x_train, y_train, t_train = create_training_batch(hidden_states, config)
        
        pred = model(x_train, t_train)
        
        # 将预测的差值和目标差值都通过head投影到logits空间
        pred_logits = base_model.lm_head(pred.to(torch.bfloat16))
        target_logits = base_model.lm_head(y_train.to(torch.bfloat16))
        
        # 在logits空间计算MSE损失
        loss = F.mse_loss(pred_logits, target_logits) 
        loss += F.mse_loss(pred, y_train)
        
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        grad_norms.append(grad_norm.item())
        
        with torch.no_grad():
            pred_norms.append(pred.norm().item())
            target_norms.append(y_train.norm().item())
        
        optimizer.step()
        
        num_batches += 1
        print(f'batch {num_batches}, loss: {loss.item():.4f}')
    
    # 记录评估结果
    use_randn = os.environ.get('hucfg_use_randn', 'True')
    with open(f'log_{use_randn}.txt', 'a+') as f:
        f.write(f"\n=== Evaluation at batch {num_batches} ===\n")
        
        # 如果是randn模型，评估噪声到特征的生成
        
        if use_randn == 'True':
            model.eval()
            with torch.no_grad():
                # 生成一个样本
                x = torch.randn(1, config.window_size, config.dim_in, device=config.device)
                steps = 100
                t = torch.linspace(0, 1.0, steps, device=config.device)
                
                for time in t:
                    pred_noise = model(x, time.unsqueeze(0)) 
                    x = x + pred_noise * (1 / steps)
                x = x / scale_factor
                
                # 解码生成的特征
                logits = base_model.lm_head(x.to(torch.bfloat16))
                gen_tokens = torch.argmax(logits, dim=-1)
                generated_text = tokenizer.decode(gen_tokens[0])
                f.write(f"Generated from noise: {generated_text}...\n")
        
        # 如果是预测下一状态的模型，评估文本生成
        else:
            model.eval()
            for prompt in eval_prompts:
                with torch.no_grad():
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
                    inputs = {k: v.to(config.device) for k, v in inputs.items()}
                    outputs = base_model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    
                    generated_text = prompt
                    for _ in range(20):  # 生成20个token
                        if hidden_states.size(1) > config.window_size:
                            hidden_states = hidden_states[:, -config.window_size:]
                        
                        steps = 100
                        t = torch.linspace(0, 1.0, steps, device=config.device)
                        next_state = hidden_states
                   
                        for time in t:
                            t = t
                            next_state_float = next_state.float()
                            pred_noise = model(next_state_float, time.unsqueeze(0))
                            next_state = next_state + pred_noise * (1 / steps)

                        
                        logits = base_model.lm_head(next_state[:, -1:].to(torch.bfloat16))
                        next_token = torch.argmax(logits, dim=-1)
                        generated_text += tokenizer.decode(next_token[0])
                        hidden_states = torch.cat([hidden_states, next_state[:, -1:]], dim=1)
                    
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Generated: {generated_text}\n")
                    f.write("-" * 50 + "\n")
    
    model.train()

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return {
        'avg_loss': avg_loss,
        'avg_grad_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
        'avg_pred_norm': sum(pred_norms) / len(pred_norms) if pred_norms else 0,
        'avg_target_norm': sum(target_norms) / len(target_norms) if target_norms else 0,
    }

def collate_fn(batch):
    # 直接返回hidden_states列表，不进行stack操作
    return batch

def train(config):
    print("Initializing dataset and model...")
    dataset = WikiDataset(config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.wiki_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    model = FeatureGPT(config).to(config.device)
    base_model = dataset.model  # 使用WikiDataset中已加载的模型
    tokenizer = dataset.tokenizer  # 使用WikiDataset中已加载的tokenizer
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    print("Starting training...")
    best_loss = float('inf')
    training_stats = []
    
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        epoch_stats = train_epoch(model, dataloader, optimizer, config, base_model, tokenizer)
        training_stats.append(epoch_stats)
        
        # 所有监控信息写入log.txt
        with open('log.txt', 'a+') as f:
            hucfg_envs = {k:v for k,v in os.environ.items() if k.startswith('hucfg_')}
            f.write(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===\n")
            f.write(f"Average Loss: {epoch_stats['avg_loss']:.4f}\n")
            f.write(f"Average Gradient Norm: {epoch_stats['avg_grad_norm']:.4f}\n")
            f.write(f"Average Pred Norm: {epoch_stats['avg_pred_norm']:.4f}\n")
            f.write(f"Average Target Norm: {epoch_stats['avg_target_norm']:.4f}\n")
            f.write(f"Pred/Target Ratio: {epoch_stats['avg_pred_norm']/epoch_stats['avg_target_norm']:.4f}\n")
            f.write(f"Config: {hucfg_envs}\n")
            
            # 添加训练趋势
            if len(training_stats) > 1:
                last_5_loss = [stats['avg_loss'] for stats in training_stats[-5:]]
                if len(last_5_loss) > 1:
                    loss_change = (last_5_loss[-1] - last_5_loss[0]) / last_5_loss[0] * 100
                    f.write(f"Loss change over last 5 epochs: {loss_change:+.2f}%\n")
        
        if epoch_stats['avg_loss'] < best_loss:
            best_loss = epoch_stats['avg_loss']
            hucfg_envs = {k:v for k,v in os.environ.items() if k.startswith('hucfg_')}
            hucfg_str = '_'.join([f'{k}_{v}' for k,v in hucfg_envs.items()])
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_stats['avg_loss'],
                'epoch': epoch,
                'training_stats': training_stats,
            }, f'best_model_future_{hucfg_str}.pt')
            print(f'Saved best model, loss: {epoch_stats["avg_loss"]:.4f}')

if __name__ == '__main__':
    config = Config()
    train(config)
