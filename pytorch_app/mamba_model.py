import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from config import Config


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dropout = dropout
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=False
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        self.B_proj = nn.Linear(d_state, self.d_inner, bias=False)
        
        self.A_log = nn.Parameter(torch.randn(d_state, self.d_inner) * 0.1)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        B, L, D = x.shape
        
        x_proj = self.x_proj(x)
        dt, B_raw = x_proj.chunk(2, dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))
        B_proj = self.B_proj(B_raw)
        
        A = -torch.exp(self.A_log)
        
        y = self.selective_scan(x, dt, A, B_proj, self.D)
        
        y = y * F.silu(z)
        
        y = self.out_proj(y)
        y = self.dropout_layer(y)
        
        return y
    
    def selective_scan(self, u: torch.Tensor, delta: torch.Tensor, 
                       A: torch.Tensor, B: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        B_batch, L, D_in = u.shape
        
        delta = delta.transpose(1, 2)
        u = u.transpose(1, 2)
        B = B.transpose(1, 2)
        
        delta = torch.clamp(delta, min=1e-4, max=1.0)
        
        y = torch.zeros_like(u)
        
        h = torch.zeros(B_batch, D_in, dtype=u.dtype, device=u.device)
        
        for i in range(L):
            h = h * (1.0 - delta[:, :, i]) + u[:, :, i] * B[:, :, i]
            y[:, :, i] = h
        
        y = y + D.unsqueeze(0).unsqueeze(-1) * u
        
        y = y.transpose(1, 2)
        
        return y


class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = x + residual
        return x


class MambaMIDIGenerator(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        model_config = config.MODEL_CONFIG
        
        self.d_model = model_config['d_model']
        self.d_state = model_config['d_state']
        self.d_conv = model_config['d_conv']
        self.expand = model_config['expand']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.max_seq_length = model_config['max_seq_length']
        self.vocab_size = model_config['vocab_size']
        
        self.num_emotions = len(config.EMOTIONS)
        self.num_styles = len(config.STYLES)
        self.num_keys = len(config.KEYS)
        self.num_modes = len(config.MODES)
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        self.emotion_embedding = nn.Embedding(self.num_emotions, self.d_model)
        self.style_embedding = nn.Embedding(self.num_styles, self.d_model)
        self.key_embedding = nn.Embedding(self.num_keys, self.d_model)
        self.mode_embedding = nn.Embedding(self.num_modes, self.d_model)
        
        self.bpm_embedding = nn.Linear(1, self.d_model)
        
        self.condition_projection = nn.Linear(self.d_model * 5, self.d_model)
        
        self.layers = nn.ModuleList([
            MambaLayer(self.d_model, self.d_state, self.d_conv, self.expand, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.d_model)
        
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        
    def encode_condition(self, emotion: torch.Tensor, style: torch.Tensor,
                         key: torch.Tensor, mode: torch.Tensor, 
                         bpm: torch.Tensor) -> torch.Tensor:
        emotion_emb = self.emotion_embedding(emotion)
        style_emb = self.style_embedding(style)
        key_emb = self.key_embedding(key)
        mode_emb = self.mode_embedding(mode)
        
        bpm_emb = self.bpm_embedding(bpm.unsqueeze(-1).float())
        
        condition = torch.cat([emotion_emb, style_emb, key_emb, mode_emb, bpm_emb], dim=-1)
        condition = self.condition_projection(condition)
        
        return condition
    
    def forward(self, tokens: torch.Tensor, emotion: torch.Tensor, 
                style: torch.Tensor, key: torch.Tensor, 
                mode: torch.Tensor, bpm: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        
        token_emb = self.token_embedding(tokens)
        
        condition = self.encode_condition(emotion, style, key, mode, bpm)
        condition = condition.unsqueeze(1).expand(-1, L, -1)
        
        positions = torch.arange(L, dtype=torch.long, device=tokens.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)
        
        x = token_emb + condition + pos_emb
        x = self.dropout_layer(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, emotion: int, style: int, key: int, mode: int, 
                 bpm: int, max_length: int = 512, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.9, seed: Optional[int] = None) -> torch.Tensor:
        self.eval()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        emotion_tensor = torch.tensor([emotion], dtype=torch.long)
        style_tensor = torch.tensor([style], dtype=torch.long)
        key_tensor = torch.tensor([key], dtype=torch.long)
        mode_tensor = torch.tensor([mode], dtype=torch.long)
        bpm_tensor = torch.tensor([bpm], dtype=torch.float)
        
        device = next(self.parameters()).device
        emotion_tensor = emotion_tensor.to(device)
        style_tensor = style_tensor.to(device)
        key_tensor = key_tensor.to(device)
        mode_tensor = mode_tensor.to(device)
        bpm_tensor = bpm_tensor.to(device)
        
        condition = self.encode_condition(emotion_tensor, style_tensor, key_tensor, mode_tensor, bpm_tensor)
        
        tokens = torch.zeros([1, 1], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                B, L = tokens.shape
                
                token_emb = self.token_embedding(tokens)
                condition_expanded = condition.unsqueeze(1).expand(-1, L, -1)
                
                positions = torch.arange(L, dtype=torch.long, device=tokens.device).unsqueeze(0).expand(B, -1)
                pos_emb = self.position_embedding(positions)
                
                x = token_emb + condition_expanded + pos_emb
                x = self.dropout_layer(x)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                logits = self.output_projection(x)
                
                logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.where(logits < values[:, -1:], torch.full_like(logits, float('-inf')), logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.topk(logits, k=logits.shape[-1])
                    sorted_logits = sorted_logits[0]
                    sorted_indices = sorted_indices[0]
                    
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove[0, sorted_indices[sorted_indices_to_remove]] = True
                    logits = torch.where(indices_to_remove, torch.full_like(logits, float('-inf')), logits)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                tokens = torch.cat([tokens, next_token], dim=1)
                
                if tokens.shape[1] >= max_length:
                    break
        
        tokens = tokens.squeeze(0)
        return tokens
    
    def generate_multi_track(self, emotion: int, style: int, key: int, mode: int, 
                            bpm: int, num_tracks: int = 4, max_length: int = 512,
                            temperature: float = 1.0, top_k: int = 50, 
                            top_p: float = 0.9, seed: Optional[int] = None) -> Dict[int, torch.Tensor]:
        self.eval()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        emotion_tensor = torch.tensor([emotion] * num_tracks, dtype=torch.long)
        style_tensor = torch.tensor([style] * num_tracks, dtype=torch.long)
        key_tensor = torch.tensor([key] * num_tracks, dtype=torch.long)
        mode_tensor = torch.tensor([mode] * num_tracks, dtype=torch.long)
        bpm_tensor = torch.tensor([bpm] * num_tracks, dtype=torch.float)
        
        device = next(self.parameters()).device
        emotion_tensor = emotion_tensor.to(device)
        style_tensor = style_tensor.to(device)
        key_tensor = key_tensor.to(device)
        mode_tensor = mode_tensor.to(device)
        bpm_tensor = bpm_tensor.to(device)
        
        condition = self.encode_condition(emotion_tensor, style_tensor, key_tensor, mode_tensor, bpm_tensor)
        
        tokens = torch.zeros([num_tracks, 1], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                B, L = tokens.shape
                
                token_emb = self.token_embedding(tokens)
                condition_expanded = condition.unsqueeze(1).expand(-1, L, -1)
                
                positions = torch.arange(L, dtype=torch.long, device=tokens.device).unsqueeze(0).expand(B, -1)
                pos_emb = self.position_embedding(positions)
                
                x = token_emb + condition_expanded + pos_emb
                x = self.dropout_layer(x)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                logits = self.output_projection(x)
                
                logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.where(logits < values[:, -1:], torch.full_like(logits, float('-inf')), logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.topk(logits, k=logits.shape[-1])
                    sorted_logits = sorted_logits[0]
                    sorted_indices = sorted_indices[0]
                    
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove[0, sorted_indices[sorted_indices_to_remove]] = True
                    logits = torch.where(indices_to_remove, torch.full_like(logits, float('-inf')), logits)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                tokens = torch.cat([tokens, next_token], dim=1)
                
                if tokens.shape[1] >= max_length:
                    break
        
        tracks = {}
        for i in range(num_tracks):
            track_tokens = tokens[i]
            tracks[i] = track_tokens.cpu()
        
        return tracks


def create_model(config: Config) -> MambaMIDIGenerator:
    model = MambaMIDIGenerator(config)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad or p.requires_grad)
