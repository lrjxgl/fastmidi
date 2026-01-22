import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Optional, Tuple, Dict
from config import Config


class MambaBlock(nn.Layer):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dropout = dropout
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias_attr=False)
        
        self.conv1d = nn.Conv1D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias_attr=False
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias_attr=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias_attr=True)
        self.B_proj = nn.Linear(d_state, self.d_inner, bias_attr=False)
        
        self.A_log = paddle.create_parameter(
            shape=[d_state, self.d_inner],
            dtype='float32',
            default_initializer=nn.initializer.Uniform()
        )
        self.A_log.set_value(paddle.log(paddle.rand([d_state, self.d_inner])))
        
        self.D = paddle.create_parameter(
            shape=[self.d_inner],
            dtype='float32',
            default_initializer=nn.initializer.Constant(1.0)
        )
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias_attr=False)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, L, D = x.shape
        
        xz = self.in_proj(x)
        x, z = paddle.chunk(xz, 2, axis=-1)
        
        x = x.transpose([0, 2, 1])
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose([0, 2, 1])
        
        x = F.silu(x)
        
        B, L, D = x.shape
        
        x_proj = self.x_proj(x)
        dt, B_raw = paddle.chunk(x_proj, 2, axis=-1)
        
        dt = F.softplus(self.dt_proj(dt))
        B_proj = self.B_proj(B_raw)
        
        A = -paddle.exp(self.A_log)
        
        y = self.selective_scan(x, dt, A, B_proj, self.D)
        
        y = y * F.silu(z)
        
        y = self.out_proj(y)
        y = self.dropout_layer(y)
        
        return y
    
    def selective_scan(self, u: paddle.Tensor, delta: paddle.Tensor, 
                       A: paddle.Tensor, B: paddle.Tensor, D: paddle.Tensor) -> paddle.Tensor:
        B_batch, L, D_in = u.shape
        
        delta = delta.transpose([0, 2, 1])
        u = u.transpose([0, 2, 1])
        B = B.transpose([0, 2, 1])
        
        delta = paddle.clip(delta, min=1e-4, max=1.0)
        
        y = paddle.zeros_like(u)
        
        log_delta = paddle.log(delta + 1e-8)
        log_one_minus_delta = paddle.log(1.0 - delta + 1e-8)
        
        h = paddle.zeros([B_batch, D_in], dtype=u.dtype)
        
        for i in range(L):
            h = h * paddle.exp(log_one_minus_delta[:, :, i]) + u[:, :, i] * B[:, :, i]
            y[:, :, i] = h
        
        y = y + D.unsqueeze(0).unsqueeze(-1) * u
        
        y = y.transpose([0, 2, 1])
        
        return y


class MambaLayer(nn.Layer):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = x + residual
        return x


class MambaMIDIGenerator(nn.Layer):
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
        
        self.layers = nn.LayerList([
            MambaLayer(self.d_model, self.d_state, self.d_conv, self.expand, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.d_model)
        
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        
    def encode_condition(self, emotion: paddle.Tensor, style: paddle.Tensor,
                         key: paddle.Tensor, mode: paddle.Tensor, 
                         bpm: paddle.Tensor) -> paddle.Tensor:
        emotion_emb = self.emotion_embedding(emotion)
        style_emb = self.style_embedding(style)
        key_emb = self.key_embedding(key)
        mode_emb = self.mode_embedding(mode)
        
        bpm_emb = self.bpm_embedding(bpm.unsqueeze(-1).astype('float32'))
        
        condition = paddle.concat([emotion_emb, style_emb, key_emb, mode_emb, bpm_emb], axis=-1)
        condition = self.condition_projection(condition)
        
        return condition
    
    def forward(self, tokens: paddle.Tensor, emotion: paddle.Tensor, 
                style: paddle.Tensor, key: paddle.Tensor, 
                mode: paddle.Tensor, bpm: paddle.Tensor) -> paddle.Tensor:
        B, L = tokens.shape
        
        token_emb = self.token_embedding(tokens)
        
        condition = self.encode_condition(emotion, style, key, mode, bpm)
        condition = condition.unsqueeze(1).expand([-1, L, -1])
        
        positions = paddle.arange(L, dtype='int64').unsqueeze(0).expand([B, -1])
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
                 top_k: int = 50, top_p: float = 0.9, seed: Optional[int] = None) -> paddle.Tensor:
        self.eval()
        
        if seed is not None:
            paddle.seed(seed)
        else:
            import random
            import numpy as np
            seed = random.randint(0, 2**32 - 1)
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        emotion_tensor = paddle.to_tensor([emotion], dtype='int64')
        style_tensor = paddle.to_tensor([style], dtype='int64')
        key_tensor = paddle.to_tensor([key], dtype='int64')
        mode_tensor = paddle.to_tensor([mode], dtype='int64')
        bpm_tensor = paddle.to_tensor([bpm], dtype='int64')
        
        emotion_tensor = emotion_tensor.cuda() if paddle.is_compiled_with_cuda() else emotion_tensor
        style_tensor = style_tensor.cuda() if paddle.is_compiled_with_cuda() else style_tensor
        key_tensor = key_tensor.cuda() if paddle.is_compiled_with_cuda() else key_tensor
        mode_tensor = mode_tensor.cuda() if paddle.is_compiled_with_cuda() else mode_tensor
        bpm_tensor = bpm_tensor.cuda() if paddle.is_compiled_with_cuda() else bpm_tensor
        
        condition = self.encode_condition(emotion_tensor, style_tensor, key_tensor, mode_tensor, bpm_tensor)
        
        tokens = paddle.zeros([1, 1], dtype='int64')
        tokens = tokens.cuda() if paddle.is_compiled_with_cuda() else tokens
        
        with paddle.no_grad():
            for step in range(max_length):
                B, L = tokens.shape
                
                token_emb = self.token_embedding(tokens)
                condition_expanded = condition.unsqueeze(1).expand([-1, L, -1])
                
                positions = paddle.arange(L, dtype='int64').unsqueeze(0).expand([B, -1])
                pos_emb = self.position_embedding(positions)
                
                x = token_emb + condition_expanded + pos_emb
                x = self.dropout_layer(x)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                logits = self.output_projection(x)
                
                logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    values, indices = paddle.topk(logits, top_k)
                    logits = paddle.where(logits < values[:, -1:], paddle.full_like(logits, float('-inf')), logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = paddle.topk(logits, k=logits.shape[-1])
                    sorted_logits = sorted_logits[0]
                    sorted_indices = sorted_indices[0]
                    
                    cumulative_probs = paddle.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = paddle.zeros_like(logits, dtype='bool')
                    indices_to_remove[0, sorted_indices[sorted_indices_to_remove]] = True
                    logits = paddle.where(indices_to_remove, paddle.full_like(logits, float('-inf')), logits)
                
                probs = F.softmax(logits, axis=-1)
                next_token = paddle.multinomial(probs, num_samples=1)
                
                tokens = paddle.concat([tokens, next_token], axis=1)
                
                if tokens.shape[1] >= max_length:
                    break
        
        tokens = tokens.squeeze(0)
        return tokens.cpu() if paddle.is_compiled_with_cuda() else tokens
    
    def generate_multi_track(self, emotion: int, style: int, key: int, mode: int, 
                            bpm: int, num_tracks: int = 4, max_length: int = 512,
                            temperature: float = 1.0, top_k: int = 50, 
                            top_p: float = 0.9, seed: Optional[int] = None) -> Dict[int, paddle.Tensor]:
        if num_tracks == 1:
            tokens = self.generate(emotion, style, key, mode, bpm, 
                                   max_length, temperature, top_k, top_p, seed)
            return {0: tokens}
        
        return self._generate_multi_track_batch(emotion, style, key, mode, bpm,
                                                 num_tracks, max_length, temperature, 
                                                 top_k, top_p, seed)
    
    def _generate_multi_track_batch(self, emotion: int, style: int, key: int, mode: int, 
                                    bpm: int, num_tracks: int, max_length: int,
                                    temperature: float, top_k: int, top_p: float, 
                                    seed: Optional[int]) -> Dict[int, paddle.Tensor]:
        self.eval()
        
        if seed is not None:
            paddle.seed(seed)
        else:
            import random
            import numpy as np
            seed = random.randint(0, 2**32 - 1)
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        emotion_tensor = paddle.to_tensor([emotion] * num_tracks, dtype='int64')
        style_tensor = paddle.to_tensor([style] * num_tracks, dtype='int64')
        key_tensor = paddle.to_tensor([key] * num_tracks, dtype='int64')
        mode_tensor = paddle.to_tensor([mode] * num_tracks, dtype='int64')
        bpm_tensor = paddle.to_tensor([bpm] * num_tracks, dtype='int64')
        
        emotion_tensor = emotion_tensor.cuda() if paddle.is_compiled_with_cuda() else emotion_tensor
        style_tensor = style_tensor.cuda() if paddle.is_compiled_with_cuda() else style_tensor
        key_tensor = key_tensor.cuda() if paddle.is_compiled_with_cuda() else key_tensor
        mode_tensor = mode_tensor.cuda() if paddle.is_compiled_with_cuda() else mode_tensor
        bpm_tensor = bpm_tensor.cuda() if paddle.is_compiled_with_cuda() else bpm_tensor
        
        condition = self.encode_condition(emotion_tensor, style_tensor, key_tensor, mode_tensor, bpm_tensor)
        
        tokens = paddle.zeros([num_tracks, 1], dtype='int64')
        tokens = tokens.cuda() if paddle.is_compiled_with_cuda() else tokens
        
        with paddle.no_grad():
            for step in range(max_length):
                B, L = tokens.shape
                
                token_emb = self.token_embedding(tokens)
                condition_expanded = condition.unsqueeze(1).expand([-1, L, -1])
                
                positions = paddle.arange(L, dtype='int64').unsqueeze(0).expand([B, -1])
                pos_emb = self.position_embedding(positions)
                
                x = token_emb + condition_expanded + pos_emb
                x = self.dropout_layer(x)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                logits = self.output_projection(x)
                
                logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    values, indices = paddle.topk(logits, top_k)
                    logits = paddle.where(logits < values[:, -1:], paddle.full_like(logits, float('-inf')), logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = paddle.topk(logits, k=logits.shape[-1])
                    sorted_logits = sorted_logits[0]
                    sorted_indices = sorted_indices[0]
                    
                    cumulative_probs = paddle.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = paddle.zeros_like(logits, dtype='bool')
                    indices_to_remove[0, sorted_indices[sorted_indices_to_remove]] = True
                    logits = paddle.where(indices_to_remove, paddle.full_like(logits, float('-inf')), logits)
                
                probs = F.softmax(logits, axis=-1)
                next_token = paddle.multinomial(probs, num_samples=1)
                
                tokens = paddle.concat([tokens, next_token], axis=1)
                
                if tokens.shape[1] >= max_length:
                    break
        
        tracks = {}
        for i in range(num_tracks):
            track_tokens = tokens[i]
            tracks[i] = track_tokens.cpu() if paddle.is_compiled_with_cuda() else track_tokens
        
        return tracks


def create_model(config: Config) -> MambaMIDIGenerator:
    model = MambaMIDIGenerator(config)
    return model


def count_parameters(model: nn.Layer) -> int:
    return sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
