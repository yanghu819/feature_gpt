import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import numpy as np
from hidden_flow_all import *

@dataclass
class EvalConfig:
    n_embd: int = 4096
    n_head: int = 16
    n_layer: int = 6
    dropout: float = 0.0  # Set to 0 for evaluation
    bias: bool = True
    time_emb_dim: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    window_size: int = 512

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def evaluate_randn_true_model(model, base_model, tokenizer, config, num_samples=5):
    """评估从随机噪声生成特征的模型"""
    model.eval()
    base_model.eval()
    
    print("\nEvaluating noise-to-feature generation...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # 从随机噪声开始
            x = torch.randn(1, config.window_size, config.n_embd, device=config.device)
            
            # 生成时间序列 t: 1.0 -> 0.0
            steps = 100
            t = torch.linspace(0, 1.0, steps, device=config.device)
            
            for time in t:
                # 预测噪声
                pred_noise = model(x, time.unsqueeze(0))
                # 更新x
                x = x + pred_noise * (1 / steps)
            
            # 使用生成的特征进行解码
            # 将特征转换为logits
            # hidden_states = x.float()
            hidden_states = x.to(torch.bfloat16)
            logits = base_model.lm_head(hidden_states)
            
            # 生成文本
            gen_tokens = torch.argmax(logits, dim=-1)
            generated_text = tokenizer.decode(gen_tokens[0])
            
            print(f"\nSample {i+1}:")
            print(f"Generated text: {generated_text[:200]}...")
            print("-" * 50)

def evaluate_randn_false_model(model, base_model, tokenizer, config, prompt="The quick brown fox", num_steps=50):
    """评估预测下一个隐藏状态的模型"""
    model.eval()
    base_model.eval()
    
    print("\nEvaluating next-state prediction...")
    
    with torch.no_grad():
        # 编码输入文本
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(config.device) for k, v in inputs.items()}
        
        # 获取初始隐藏状态
        outputs = base_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len, dim]
        
        generated_text = prompt
        
        for _ in range(num_steps):
            # 如果序列太长，只保留最后window_size个token
            if hidden_states.size(1) > config.window_size:
                hidden_states = hidden_states[:, -config.window_size:]
            
            # 预测下一个状态的差值
            steps = 100
            t = torch.linspace(0, 1.0, steps, device=config.device)
            next_state = hidden_states
            for time in t:
                # 预测噪声
                pred_noise = model(next_state, time.unsqueeze(0))
                # 更新x
                next_state = next_state + pred_noise * (1 / steps)

                                            
            
            # 解码最后一个token
            logits = base_model.lm_head(next_state[:, -1:].to(torch.bfloat16))
            next_token = torch.argmax(logits, dim=-1)
            
            # 更新生成的文本
            generated_text += tokenizer.decode(next_token[0])
            
            # 为下一步准备隐藏状态
            hidden_states = torch.cat([hidden_states, next_state[:, -1:]], dim=1)
            
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    config = EvalConfig()
    base_model, tokenizer = load_base_model()
    
    # 加载并评估 randn=True 的模型
    model_true = FeatureGPT(config).to(config.device)
    checkpoint_true = torch.load('best_model_future_hucfg_use_randn_True.pt')
    model_true.load_state_dict(checkpoint_true['model_state_dict'])
    evaluate_randn_true_model(model_true, base_model, tokenizer, config)
    
    # 加载并评估 randn=False 的模型
    model_false = FeatureGPT(config).to(config.device)
    checkpoint_false = torch.load('best_model_future_hucfg_use_randn_False.pt')
    model_false.load_state_dict(checkpoint_false['model_state_dict'])
    
    # 使用不同的提示评估
    prompts = [
        "The quick brown fox",
        "In a world where",
        "Scientists have discovered",
        "Once upon a time",
    ]
    
    for prompt in prompts:
        evaluate_randn_false_model(model_false, base_model, tokenizer, config, prompt=prompt)
