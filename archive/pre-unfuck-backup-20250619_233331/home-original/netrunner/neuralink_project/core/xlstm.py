# /home/netrunner/neuralink_project/core/xlstm.py
"""
NEXUS-7 Extended LSTM Core
Exponential gating meets cyberpunk chrome
Built for speed, built for the streets
"""

import numpy as np
from typing import Tuple, Optional, Dict
import numba as nb  # JIT compilation for street-level speed

@nb.jit(nopython=True, cache=True)
def sigmoid_chrome(x: np.ndarray) -> np.ndarray:
    """Optimized sigmoid - smoother than black ice"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

@nb.jit(nopython=True, cache=True)
def tanh_neural(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent - the curve of consciousness"""
    return np.tanh(np.clip(x, -350, 350))

@nb.jit(nopython=True, cache=True)
def exponential_gate(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Exponential gating mechanism - xLSTM's secret sauce"""
    return np.exp(beta * sigmoid_chrome(x))

class xLSTMCell:
    """
    Extended LSTM Cell - Enhanced with exponential gating
    Memory that learns like a veteran netrunner
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 dropout_rate: float = 0.1, use_exponential: bool = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_exponential = use_exponential
        
        # Initialize weight matrices - Xavier/Glorot for stability
        self._init_chrome_weights()
        
        # Optimization caches
        self.computation_cache = {}
        self.gradient_tape = []
        
    def _init_chrome_weights(self):
        """Initialize neural chrome with street-tested values"""
        # Input gate weights
        self.W_i = self._xavier_init(self.input_dim, self.hidden_dim)
        self.U_i = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.b_i = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Forget gate weights - biased to remember
        self.W_f = self._xavier_init(self.input_dim, self.hidden_dim)
        self.U_f = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.b_f = np.ones(self.hidden_dim, dtype=np.float32)  # Forget gate bias
        
        # Cell gate weights
        self.W_c = self._xavier_init(self.input_dim, self.hidden_dim)
        self.U_c = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.b_c = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Output gate weights
        self.W_o = self._xavier_init(self.input_dim, self.hidden_dim)
        self.U_o = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.b_o = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Exponential gate parameters
        if self.use_exponential:
            self.W_exp = self._xavier_init(self.input_dim, self.hidden_dim)
            self.U_exp = self._xavier_init(self.hidden_dim, self.hidden_dim)
            self.b_exp = np.zeros(self.hidden_dim, dtype=np.float32)
            self.beta = np.ones(self.hidden_dim, dtype=np.float32)
    
    def _xavier_init(self, n_in: int, n_out: int) -> np.ndarray:
        """Xavier initialization - stable as chrome implants"""
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_in, n_out)).astype(np.float32)
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, 
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward propagation through the neural maze
        x: input vector [batch_size, input_dim]
        h_prev: previous hidden state [batch_size, hidden_dim]
        c_prev: previous cell state [batch_size, hidden_dim]
        """
        # Input gate - what new info flows in
        i_t = sigmoid_chrome(x @ self.W_i + h_prev @ self.U_i + self.b_i)
        
        # Forget gate - what memories to dump
        f_t = sigmoid_chrome(x @ self.W_f + h_prev @ self.U_f + self.b_f)
        
        # Candidate values - potential new memories
        c_tilde = tanh_neural(x @ self.W_c + h_prev @ self.U_c + self.b_c)
        
        # Apply exponential gating if enabled
        if self.use_exponential:
            exp_gate = exponential_gate(
                x @ self.W_exp + h_prev @ self.U_exp + self.b_exp, 
                self.beta
            )
            # Modulate forget gate with exponential component
            f_t = f_t * exp_gate / (exp_gate + 1e-8)
        
        # Update cell state - the core memory
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate - what memories to access
        o_t = sigmoid_chrome(x @ self.W_o + h_prev @ self.U_o + self.b_o)
        
        # Hidden state - the interface to the world
        h_t = o_t * tanh_neural(c_t)
        
        # Cache for backprop if training
        self.computation_cache = {
            'i_t': i_t, 'f_t': f_t, 'c_tilde': c_tilde,
            'o_t': o_t, 'c_prev': c_prev, 'h_prev': h_prev,
            'x': x, 'c_t': c_t
        }
        
        return h_t, c_t
    
    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray) -> Dict:
        """Backprop through the neural pathways"""
        cache = self.computation_cache
        
        # Retrieve cached values
        i_t = cache['i_t']
        f_t = cache['f_t']
        c_tilde = cache['c_tilde']
        o_t = cache['o_t']
        c_prev = cache['c_prev']
        c_t = cache['c_t']
        
        # Gradients computation
        do_t = dh_next * tanh_neural(c_t)
        dc_t = dh_next * o_t * (1 - tanh_neural(c_t)**2) + dc_next
        
        df_t = dc_t * c_prev
        di_t = dc_t * c_tilde
        dc_tilde = dc_t * i_t
        dc_prev = dc_t * f_t
        
        # Gate derivative computations
        di_gate = di_t * i_t * (1 - i_t)
        df_gate = df_t * f_t * (1 - f_t)
        dc_gate = dc_tilde * (1 - c_tilde**2)
        do_gate = do_t * o_t * (1 - o_t)
        
        # Parameter gradients
        gradients = {
            'dW_i': cache['x'].T @ di_gate,
            'dU_i': cache['h_prev'].T @ di_gate,
            'db_i': np.sum(di_gate, axis=0),
            'dW_f': cache['x'].T @ df_gate,
            'dU_f': cache['h_prev'].T @ df_gate,
            'db_f': np.sum(df_gate, axis=0),
            'dW_c': cache['x'].T @ dc_gate,
            'dU_c': cache['h_prev'].T @ dc_gate,
            'db_c': np.sum(dc_gate, axis=0),
            'dW_o': cache['x'].T @ do_gate,
            'dU_o': cache['h_prev'].T @ do_gate,
            'db_o': np.sum(do_gate, axis=0),
            'dx': di_gate @ self.W_i.T + df_gate @ self.W_f.T + 
                  dc_gate @ self.W_c.T + do_gate @ self.W_o.T,
            'dh_prev': di_gate @ self.U_i.T + df_gate @ self.U_f.T + 
                       dc_gate @ self.U_c.T + do_gate @ self.U_o.T,
            'dc_prev': dc_prev
        }
        
        return gradients

class xLSTMLayer:
    """
    Full xLSTM Layer - Stacked cells for deep memory
    Optimized for cyberdeck deployment
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 sequence_length: int, batch_size: int,
                 return_sequences: bool = True,
                 stateful: bool = False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.return_sequences = return_sequences
        self.stateful = stateful
        
        # Initialize the cell
        self.cell = xLSTMCell(input_dim, hidden_dim)
        
        # State management for stateful operations
        if self.stateful:
            self.h_state = np.zeros((batch_size, hidden_dim), dtype=np.float32)
            self.c_state = np.zeros((batch_size, hidden_dim), dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process sequence through the layer
        x: [batch_size, sequence_length, input_dim]
        """
        batch_size = x.shape[0]
        
        # Initialize states
        if self.stateful:
            h_t = self.h_state
            c_t = self.c_state
        else:
            h_t = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            c_t = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        
        outputs = []
        
        # Process sequence
        for t in range(self.sequence_length):
            h_t, c_t = self.cell.forward(x[:, t, :], h_t, c_t)
            outputs.append(h_t)
        
        # Update states if stateful
        if self.stateful:
            self.h_state = h_t
            self.c_state = c_t
        
        # Return sequences or final output
        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]
    
    def reset_states(self):
        """Clear the memory banks - fresh start"""
        if self.stateful:
            self.h_state.fill(0)
            self.c_state.fill(0)
