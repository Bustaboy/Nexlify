# /home/netrunner/neuralink_project/utils/tensor_ops.py
"""
Tensor Operations - Optimized neural computation primitives
Hardware-accelerated operations for the streets
When every nanosecond counts in the data stream
"""

import numpy as np
import numba as nb
from typing import Tuple, List, Optional, Union
from functools import lru_cache

# JIT-compiled operations for maximum chrome
@nb.jit(nopython=True, cache=True, fastmath=True)
def fast_sigmoid(x: np.ndarray) -> np.ndarray:
    """Ultra-fast sigmoid activation"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

@nb.jit(nopython=True, cache=True, fastmath=True)
def fast_tanh(x: np.ndarray) -> np.ndarray:
    """Ultra-fast tanh activation"""
    return np.tanh(np.clip(x, -350, 350))

@nb.jit(nopython=True, cache=True, fastmath=True)
def fast_relu(x: np.ndarray) -> np.ndarray:
    """Ultra-fast ReLU activation"""
    return np.maximum(0, x)

@nb.jit(nopython=True, cache=True, fastmath=True)
def fast_softmax(x: np.ndarray) -> np.ndarray:
    """Ultra-fast softmax for probability distributions"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

@nb.jit(nopython=True, cache=True, fastmath=True)
def fast_layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Fast layer normalization"""
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)

class TensorOps:
    """
    High-performance tensor operations
    Optimized for neural network computations
    """
    
    def __init__(self):
        # Cache for frequently used operations
        self.cache = {}
        
        # Operation counters for profiling
        self.op_counts = {
            'matmul': 0,
            'conv2d': 0,
            'attention': 0,
            'activation': 0
        }
    
    @staticmethod
    @nb.jit(nopython=True, cache=True, parallel=True)
    def matmul_optimized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Optimized matrix multiplication
        Uses parallel processing when available
        """
        m, k = a.shape
        k2, n = b.shape
        
        assert k == k2, "Matrix dimensions must align"
        
        c = np.zeros((m, n), dtype=a.dtype)
        
        # Parallel loop for outer dimension
        for i in nb.prange(m):
            for j in range(n):
                sum_val = 0.0
                for l in range(k):
                    sum_val += a[i, l] * b[l, j]
                c[i, j] = sum_val
        
        return c
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def conv1d_optimized(x: np.ndarray, kernel: np.ndarray, 
                        stride: int = 1) -> np.ndarray:
        """
        Optimized 1D convolution for sequence processing
        """
        seq_len = x.shape[0]
        kernel_size = kernel.shape[0]
        out_len = (seq_len - kernel_size) // stride + 1
        
        output = np.zeros(out_len, dtype=x.dtype)
        
        for i in range(out_len):
            start_idx = i * stride
            output[i] = np.sum(x[start_idx:start_idx + kernel_size] * kernel)
        
        return output
    
    @staticmethod
    @nb.jit(nopython=True, cache=True, parallel=True)
    def attention_scores(query: np.ndarray, key: np.ndarray, 
                        value: np.ndarray, scale: float) -> np.ndarray:
        """
        Optimized attention mechanism computation
        """
        # Compute attention scores
        scores = TensorOps.matmul_optimized(query, key.T) * scale
        
        # Apply softmax
        attn_weights = np.zeros_like(scores)
        for i in nb.prange(scores.shape[0]):
            attn_weights[i] = fast_softmax(scores[i])
        
        # Apply attention to values
        output = TensorOps.matmul_optimized(attn_weights, value)
        
        return output
    
    @staticmethod
    def batch_operation(func, data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Process data in batches for memory efficiency
        """
        n_samples = data.shape[0]
        results = []
        
        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            results.append(func(batch))
        
        return np.concatenate(results, axis=0)
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def exponential_moving_average(data: np.ndarray, alpha: float = 0.9) -> np.ndarray:
        """
        Compute exponential moving average
        Used for momentum in optimization
        """
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * ema[i-1] + (1 - alpha) * data[i]
        
        return ema
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def gradient_clip(grad: np.ndarray, max_norm: float) -> np.ndarray:
        """
        Gradient clipping to prevent exploding gradients
        """
        grad_norm = np.sqrt(np.sum(grad ** 2))
        
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)
        
        return grad
    
    @staticmethod
    @nb.jit(nopython=True, cache=True, parallel=True)
    def parallel_reduce(data: np.ndarray, axis: int = 0, 
                       operation: str = 'sum') -> np.ndarray:
        """
        Parallel reduction operations
        """
        if operation == 'sum':
            return np.sum(data, axis=axis)
        elif operation == 'mean':
            return np.mean(data, axis=axis)
        elif operation == 'max':
            return np.max(data, axis=axis)
        elif operation == 'min':
            return np.min(data, axis=axis)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def sliding_window(data: np.ndarray, window_size: int, 
                      stride: int = 1) -> np.ndarray:
        """
        Create sliding windows over sequential data
        Useful for time series processing
        """
        n_samples = data.shape[0]
        n_windows = (n_samples - window_size) // stride + 1
        
        shape = (n_windows, window_size) + data.shape[1:]
        strides = (data.strides[0] * stride,) + data.strides
        
        return np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides
        )
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between vectors
        """
        dot_product = np.dot(a, b)
        norm_a = np.sqrt(np.sum(a ** 2))
        norm_b = np.sqrt(np.sum(b ** 2))
        
        return dot_product / (norm_a * norm_b + 1e-8)
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Euclidean distance between vectors
        """
        return np.sqrt(np.sum((a - b) ** 2))
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """
        Create causal attention mask for autoregressive models
        """
        mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        return mask
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def gelu_activation(x: np.ndarray) -> np.ndarray:
        """
        GELU (Gaussian Error Linear Unit) activation
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def swish_activation(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """
        Swish activation function
        """
        return x * fast_sigmoid(beta * x)

class EfficientStorage:
    """
    Memory-efficient tensor storage with compression
    """
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        self.storage = {}
        
    def store(self, key: str, tensor: np.ndarray, 
              quantize: bool = False) -> int:
        """
        Store tensor with optional quantization
        Returns size in bytes
        """
        if quantize:
            # Simple 8-bit quantization
            min_val = np.min(tensor)
            max_val = np.max(tensor)
            scale = (max_val - min_val) / 255.0
            
            quantized = ((tensor - min_val) / scale).astype(np.uint8)
            
            self.storage[key] = {
                'data': quantized,
                'scale': scale,
                'min_val': min_val,
                'quantized': True
            }
            
            return quantized.nbytes
        else:
            self.storage[key] = {
                'data': tensor,
                'quantized': False
            }
            
            return tensor.nbytes
    
    def retrieve(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve stored tensor
        """
        if key not in self.storage:
            return None
        
        stored = self.storage[key]
        
        if stored['quantized']:
            # Dequantize
            quantized = stored['data']
            scale = stored['scale']
            min_val = stored['min_val']
            
            return quantized.astype(np.float32) * scale + min_val
        else:
            return stored['data']
    
    def clear(self):
        """Clear all stored tensors"""
        self.storage.clear()

class StreamingStats:
    """
    Compute statistics on streaming data
    Memory-efficient for large data streams
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def update(self, value: Union[float, np.ndarray]):
        """Update statistics with new value(s)"""
        if isinstance(value, np.ndarray):
            for v in value.flat:
                self._update_single(v)
        else:
            self._update_single(value)
    
    def _update_single(self, value: float):
        """Update with single value using Welford's algorithm"""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    @property
    def variance(self) -> float:
        """Get variance"""
        return self.m2 / self.n if self.n > 1 else 0.0
    
    @property
    def std(self) -> float:
        """Get standard deviation"""
        return np.sqrt(self.variance)
    
    def get_stats(self) -> dict:
        """Get all statistics"""
        return {
            'count': self.n,
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val,
            'max': self.max_val
        }

# Specialized operations for neural architectures
class NeuralOps:
    """
    Specialized operations for neural network layers
    """
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def lstm_cell_forward(x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                         W_i: np.ndarray, W_f: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
                         U_i: np.ndarray, U_f: np.ndarray, U_c: np.ndarray, U_o: np.ndarray,
                         b_i: np.ndarray, b_f: np.ndarray, b_c: np.ndarray, b_o: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized LSTM cell forward pass
        """
        # Input gate
        i_t = fast_sigmoid(x @ W_i + h_prev @ U_i + b_i)
        
        # Forget gate
        f_t = fast_sigmoid(x @ W_f + h_prev @ U_f + b_f)
        
        # Cell gate
        c_tilde = fast_tanh(x @ W_c + h_prev @ U_c + b_c)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate
        o_t = fast_sigmoid(x @ W_o + h_prev @ U_o + b_o)
        
        # Hidden state
        h_t = o_t * fast_tanh(c_t)
        
        return h_t, c_t
    
    @staticmethod
    def create_position_encoding(seq_len: int, d_model: int) -> np.ndarray:
        """
        Create sinusoidal position encodings
        """
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Scaled dot-product attention mechanism
        """
        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask * -1e9
        
        attention_weights = fast_softmax(scores)
        output = attention_weights @ V
        
        return output

# Memory pool for tensor reuse
class TensorPool:
    """
    Object pool for tensor reuse
    Reduces allocation overhead
    """
    
    def __init__(self, max_tensors: int = 100):
        self.max_tensors = max_tensors
        self.pool = {}
        
    def get_tensor(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Get tensor from pool or allocate new one
        """
        key = (shape, dtype)
        
        if key in self.pool and len(self.pool[key]) > 0:
            return self.pool[key].pop()
        else:
            return np.zeros(shape, dtype=dtype)
    
    def return_tensor(self, tensor: np.ndarray):
        """
        Return tensor to pool for reuse
        """
        key = (tensor.shape, tensor.dtype)
        
        if key not in self.pool:
            self.pool[key] = []
        
        if len(self.pool[key]) < self.max_tensors:
            # Clear tensor before returning to pool
            tensor.fill(0)
            self.pool[key].append(tensor)
    
    def clear(self):
        """Clear tensor pool"""
        self.pool.clear()

# Global tensor pool instance
_tensor_pool = TensorPool()

def get_tensor_pool() -> TensorPool:
    """Get global tensor pool instance"""
    return _tensor_pool
