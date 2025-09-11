# -*- coding: utf-8 -*-
"""
从零实现的 LSTM（纯 NumPy 版本），面向初学者的可运行示例

主要特性：
1) 采用最常见的 LSTM 公式：输入门 i_t、遗忘门 f_t、候选记忆 g_t、输出门 o_t；
2) 支持批量输入（batch-first），输入张量形状为 (batch, seq_len, input_size)；
3) 提供 verbose 开关，逐步打印每个时间步的关键中间张量，便于学习和调试；
4) 代码中穿插大量中文注释，解释算法流程、函数作用、输入输出形状等。

公式（时间步 t）：
    z_i = x_t W_xi + h_{t-1} W_hi + b_i
    z_f = x_t W_xf + h_{t-1} W_hf + b_f
    z_g = x_t W_xg + h_{t-1} W_hg + b_g
    z_o = x_t W_xo + h_{t-1} W_ho + b_o

    i_t = sigmoid(z_i)         # 输入门
    f_t = sigmoid(z_f)         # 遗忘门
    g_t = tanh(z_g)            # 候选记忆
    o_t = sigmoid(z_o)         # 输出门

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

其中：
- x_t 形状 (batch, input_size)
- h_{t-1}, c_{t-1}, h_t, c_t 形状 (batch, hidden_size)
- ⊙ 表示逐元素相乘
"""

from typing import Tuple, Dict, Optional, List
import numpy as np


class LSTM:
    """
    LSTM 循环层（单层，无投影），纯 NumPy 实现。

    参数：
    - input_size (int): 每个时间步输入向量的维度（特征数）。
    - hidden_size (int): 隐藏状态/记忆单元的维度（神经元数）。
    - seed (Optional[int]): 随机种子，用于可重复的权重初始化（可选）。
    - forget_bias (float): 遗忘门偏置的初值，通常设为正值（如 0.5 或 1.0），
                           有助于在训练早期“多记忆少遗忘”。

    属性（权重形状）：
    - W_xi, W_xf, W_xg, W_xo: (input_size, hidden_size)
    - W_hi, W_hf, W_hg, W_ho: (hidden_size, hidden_size)
    - b_i,  b_f,  b_g,  b_o : (hidden_size,)

    主要方法：
    - forward(x, h0=None, c0=None, verbose=False) -> (H, h_T, c_T)
      将一整段序列输入 LSTM，计算所有时间步的隐藏状态、并返回最终 h_T, c_T。
    - step(x_t, h_prev, c_prev, verbose=False) -> (h_t, c_t, cache)
      执行单个时间步的前向计算，并可打印中间张量（学习用）。

    输入/输出约定：
    - forward 的输入 x 形状为 (batch, seq_len, input_size)
    - forward 的输出：
        H   : (batch, seq_len, hidden_size)  # 每个时间步的隐藏状态拼接
        h_T : (batch, hidden_size)           # 最后一个时间步的隐藏状态
        c_T : (batch, hidden_size)           # 最后一个时间步的记忆单元

    适用人群：
    - 初学者；希望从原理出发，逐步理解 LSTM 的数据流与张量形状。
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 seed: Optional[int] = 42,
                 forget_bias: float = 0.5):
        # -------- 基本形状参数 --------
        self.input_size = input_size
        self.hidden_size = hidden_size

        # -------- 随机初始化 --------
        rng = np.random.default_rng(seed)
        scale = 0.2  # 权重初始化尺度（较小有利于数值稳定与观察）

        # 输入到各门的权重 (input -> gates)，形状 (input_size, hidden_size)
        self.W_xi = rng.normal(0, scale, size=(input_size, hidden_size))
        self.W_xf = rng.normal(0, scale, size=(input_size, hidden_size))
        self.W_xg = rng.normal(0, scale, size=(input_size, hidden_size))
        self.W_xo = rng.normal(0, scale, size=(input_size, hidden_size))

        # 隐藏态到各门的权重 (hidden -> gates)，形状 (hidden_size, hidden_size)
        self.W_hi = rng.normal(0, scale, size=(hidden_size, hidden_size))
        self.W_hf = rng.normal(0, scale, size=(hidden_size, hidden_size))
        self.W_hg = rng.normal(0, scale, size=(hidden_size, hidden_size))
        self.W_ho = rng.normal(0, scale, size=(hidden_size, hidden_size))

        # 各门偏置，形状 (hidden_size,)
        self.b_i = np.zeros((hidden_size,), dtype=np.float32)
        self.b_f = np.ones((hidden_size,), dtype=np.float32) * forget_bias  # 遗忘门加正偏置
        self.b_g = np.zeros((hidden_size,), dtype=np.float32)
        self.b_o = np.zeros((hidden_size,), dtype=np.float32)

    # -------------------- 激活函数（私有工具） --------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid 激活：σ(x) = 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        """Tanh 激活"""
        return np.tanh(x)

    # -------------------- 单步计算 --------------------
    def step(self,
             x_t: np.ndarray,
             h_prev: np.ndarray,
             c_prev: np.ndarray,
             verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        执行 LSTM 的单个时间步前向计算。

        参数：
        - x_t (ndarray): 本时间步输入，形状 (batch, input_size)
        - h_prev (ndarray): 上一时间步隐藏态 h_{t-1}，形状 (batch, hidden_size)
        - c_prev (ndarray): 上一时间步记忆单元 c_{t-1}，形状 (batch, hidden_size)
        - verbose (bool): 若为 True，将逐步打印关键中间张量（学习/调试用）

        返回：
        - h_t (ndarray): 本时间步隐藏态，形状 (batch, hidden_size)
        - c_t (ndarray): 本时间步记忆单元，形状 (batch, hidden_size)
        - cache (dict): 缓存中间结果，便于上层调试或对齐测试
        """
        # 线性项（pre-activation）：分别对应四个门
        z_i = x_t @ self.W_xi + h_prev @ self.W_hi + self.b_i
        z_f = x_t @ self.W_xf + h_prev @ self.W_hf + self.b_f
        z_g = x_t @ self.W_xg + h_prev @ self.W_hg + self.b_g
        z_o = x_t @ self.W_xo + h_prev @ self.W_ho + self.b_o

        # 门激活
        i_t = self._sigmoid(z_i)   # 输入门：决定写入多少新信息
        f_t = self._sigmoid(z_f)   # 遗忘门：决定保留多少旧记忆
        g_t = self._tanh(z_g)      # 候选记忆：新的候选信息
        o_t = self._sigmoid(z_o)   # 输出门：决定输出多少记忆

        # 更新记忆单元与隐藏态
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * self._tanh(c_t)

        # 学习/调试打印
        if verbose:
            self._print_step_tensors(x_t, h_prev, c_prev, z_i, z_f, z_g, z_o, i_t, f_t, g_t, o_t, c_t, h_t)

        cache = {
            "x_t": x_t, "h_prev": h_prev, "c_prev": c_prev,
            "z_i": z_i, "z_f": z_f, "z_g": z_g, "z_o": z_o,
            "i_t": i_t, "f_t": f_t, "g_t": g_t, "o_t": o_t,
            "c_t": c_t, "h_t": h_t
        }
        return h_t, c_t, cache

    # -------------------- 序列前向 --------------------
    def forward(self,
                x: np.ndarray,
                h0: Optional[np.ndarray] = None,
                c0: Optional[np.ndarray] = None,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        处理完整序列的前向计算。

        参数：
        - x (ndarray): 输入序列，形状 (batch, seq_len, input_size) —— 注意是 batch-first
        - h0 (Optional[ndarray]): 初始隐藏态，形状 (batch, hidden_size)，若不提供则置零
        - c0 (Optional[ndarray]): 初始记忆单元，形状 (batch, hidden_size)，若不提供则置零
        - verbose (bool): 若为 True，将在每个时间步打印中间张量（学习/调试用）

        返回：
        - H  (ndarray): 所有时间步的隐藏态拼接，形状 (batch, seq_len, hidden_size)
        - hT (ndarray): 最后一个时间步的隐藏态，形状 (batch, hidden_size)
        - cT (ndarray): 最后一个时间步的记忆单元，形状 (batch, hidden_size)
        """
        # -------- 基本形状检查 --------
        assert x.ndim == 3, "x 必须是三维张量 (batch, seq_len, input_size)"
        batch, seq_len, input_size = x.shape
        assert input_size == self.input_size, f"输入最后一维应为 {self.input_size}, 实际为 {input_size}"

        # -------- 初始化 h, c --------
        if h0 is None:
            h_t = np.zeros((batch, self.hidden_size), dtype=np.float32)
        else:
            assert h0.shape == (batch, self.hidden_size)
            h_t = h0.astype(np.float32)

        if c0 is None:
            c_t = np.zeros((batch, self.hidden_size), dtype=np.float32)
        else:
            assert c0.shape == (batch, self.hidden_size)
            c_t = c0.astype(np.float32)

        # -------- 输出容器 --------
        H = np.zeros((batch, seq_len, self.hidden_size), dtype=np.float32)

        # -------- 时间维迭代 --------
        for t in range(seq_len):
            x_t = x[:, t, :]  # 取出第 t 个时间步的输入，形状 (batch, input_size)
            h_t, c_t, _ = self.step(x_t, h_t, c_t, verbose=verbose)
            H[:, t, :] = h_t

        # h_t, c_t 是最后时间步的 h_T, c_T
        return H, h_t, c_t

    # -------------------- 可读性打印（学习/调试） --------------------
    @staticmethod
    def _print_header(title: str):
        """美观的分隔标题打印"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    @staticmethod
    def _print_tensor(name: str, x: np.ndarray):
        """打印张量名称、形状与内容（仅学习/调试使用）"""
        print(f"{name} | shape={tuple(x.shape)}")
        print(x)
        print("-" * 80)

    def _print_step_tensors(self,
                            x_t, h_prev, c_prev,
                            z_i, z_f, z_g, z_o,
                            i_t, f_t, g_t, o_t,
                            c_t, h_t):
        """集中打印单步所有关键张量"""
        self._print_header("LSTM 单步中间张量（学习/调试）")
        self._print_tensor("x_t", x_t)
        self._print_tensor("h_prev", h_prev)
        self._print_tensor("c_prev", c_prev)
        self._print_tensor("z_i (输入门 pre-activation)", z_i)
        self._print_tensor("z_f (遗忘门 pre-activation)", z_f)
        self._print_tensor("z_g (候选记忆 pre-activation)", z_g)
        self._print_tensor("z_o (输出门 pre-activation)", z_o)
        self._print_tensor("i_t = sigmoid(z_i)", i_t)
        self._print_tensor("f_t = sigmoid(z_f)", f_t)
        self._print_tensor("g_t = tanh(z_g)", g_t)
        self._print_tensor("o_t = sigmoid(z_o)", o_t)
        self._print_tensor("c_t = f_t*c_prev + i_t*g_t", c_t)
        self._print_tensor("h_t = o_t * tanh(c_t)", h_t)


# ============================ 演示用主程序 ============================
if __name__ == "__main__":
    # 为了更易读的打印
    np.set_printoptions(precision=3, suppress=True)

    # -------- 超小规模示例（便于观察） --------
    batch = 2
    seq_len = 4
    input_size = 3
    hidden_size = 5

    # 构造一个随机输入序列 (batch, seq_len, input_size)
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1.0, size=(batch, seq_len, input_size)).astype(np.float32)

    # 实例化一个 LSTM
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, seed=7, forget_bias=0.5)

    # 运行前向：verbose=True 将打印每个时间步的详细中间张量
    H, h_T, c_T = lstm.forward(x, verbose=True)

    # 打印最终结果
    print("\n" + "#" * 80)
    print("最终输出（整段序列的隐藏态 H、末步隐藏态 h_T、末步记忆单元 c_T）：")
    print("#" * 80)
    print("H 形状:", H.shape)
    print(H)
    print("-" * 80)
    print("h_T 形状:", h_T.shape)
    print(h_T)
    print("-" * 80)
    print("c_T 形状:", c_T.shape)
    print(c_T)
