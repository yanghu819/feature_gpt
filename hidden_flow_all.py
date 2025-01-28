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

@dataclass
class Config:
    n_embd: int = 4096
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
    num_epochs: int = 10
    wiki_batch_size: int = 2  # 每批处理的wiki文章数量
    window_size: int = 512

    sequence_length: int = 576
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0

class WikiDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset = load_dataset("wikipedia", "20220301.en", split="train[:5000]")
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
            padding='max_length',
            max_length=self.config.sequence_length
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].squeeze(0)  # [seq_len, dim]
            attention_mask = inputs['attention_mask'].squeeze(0)
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            
            # 转换为 float32 类型
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
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        norm_x = x.norm(2, -1, keepdim=True)
        rms_x = norm_x * x.size(-1) ** (-0.5) 
        x_normed = x / (rms_x + self.eps)
        return self.weight * x_normed

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.kqv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

        self.q_norm = RMSNorm(self.head_dim)  # 需实现RMSNorm
        self.k_norm = RMSNorm(self.head_dim)


    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_embeddings(self, q, k, seq_len):
        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        freqs = freqs.unsqueeze(0).unsqueeze(1)
        q_rot = q * torch.cos(freqs) + self._rotate_half(q) * torch.sin(freqs)
        k_rot = k * torch.cos(freqs) + self._rotate_half(k) * torch.sin(freqs)
        return q_rot, k_rot

    def forward(self, x):
        B, T, C = x.size()
        
        k, q, v = self.kqv(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = self.apply_rotary_embeddings(q, k, T)
        
        eps = 1e-6

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(scores, dim=-1)

        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.proj(y))
        return y

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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adaln1 = AdaLN(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.adaln2 = AdaLN(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, time_emb):
        # time_emb: [batch_size, 4 * n_embd]
        B = time_emb.size(0)
        time_emb = time_emb.view(B, 4, -1)  # [batch_size, 4, n_embd]
        scale1, shift1, scale2, shift2 = time_emb.unbind(dim=1)
        
        x = x + self.attn(self.adaln1(x, scale1, shift1))
        x = x + self.mlp(self.adaln2(x, scale2, shift2))
        return x

class FeatureGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_mlp = SinusoidalTimeEncoding(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.adaln_final = AdaLN(config.n_embd)
        self.apply(self._init_weights)
        print(f"FeatureGPT parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, t):
        time_emb = self.time_mlp(t)  # [batch_size, 4 * n_embd]
        
        for block in self.blocks:
            x = block(x, time_emb)
            
        time_emb = time_emb.view(time_emb.size(0), 4, -1)
        scale, shift = time_emb[:, 0], time_emb[:, 1]
        x = self.adaln_final(x, scale, shift)
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
            future_seq = features[idx+1:idx+config.window_size+1]  # [window_size, dim]
            
            # 生成随机时间步长
            # t = torch.rand(1, device=features.device)
            t = torch.sigmoid(torch.randn(1, device=features.device))
            
            # 计算插值和目标
            use_randn = os.environ.get('hucfg_use_randn', 'True')
            if use_randn=='True':
                randn_seq = torch.randn_like(current_seq)
                interpolated = (t) * current_seq + (1-t) * randn_seq  # [window_size, dim]
                target = current_seq - randn_seq  # [window_size, dim]
            else:
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

def train_epoch(model, dataloader, optimizer, config):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, hidden_states in enumerate(dataloader):
        # hidden_states是一个列表，包含wiki_batch_size个张量，每个张量形状为[seq_len, dim]
        hidden_states = [h.to(config.device) for h in hidden_states]
        x_train, y_train, t_train = create_training_batch(hidden_states, config)
        
        pred = model(x_train, t_train)
        loss = F.mse_loss(pred, y_train)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        num_batches += 1
        if num_batches % 10 == 0:
            print(f'batch {num_batches}, loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    print(f'epoch complete, avg loss: {avg_loss:.4f}')

    return avg_loss

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    print("Starting training...")
    best_loss = float('inf')
    for epoch in range(config.num_epochs):

        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        loss = train_epoch(model, dataloader, optimizer, config)
        
        with open('log.txt', 'a+') as f:
            hucfg_envs = {k:v for k,v in os.environ.items() if k.startswith('hucfg_')}
            f.write(f'Epoch {epoch+1}/{config.num_epochs}, avg loss: {loss:.4f}, hucfg_params: {hucfg_envs}\n')
            
        if loss < best_loss:
            best_loss = loss
            hucfg_envs = {k:v for k,v in os.environ.items() if k.startswith('hucfg_')}
            hucfg_str = '_'.join([f'{k}_{v}' for k,v in hucfg_envs.items()])
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'epoch': epoch,
            }, f'best_model_future_{hucfg_str}.pt')
            print(f'Saved best model, loss: {loss:.4f}')

if __name__ == '__main__':
    config = Config()
    train(config)
