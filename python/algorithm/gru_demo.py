# -*- coding: utf-8 -*-
"""
从零实现的 GRU（纯 NumPy 版本），面向初学者的可运行示例

GRU 的核心思路：
- 使用「更新门 z_t」与「重置门 r_t」控制信息流动；
- 先用 r_t 控制上一时刻隐藏态 h_{t-1} 的参与度，得到候选隐状态 n_t；
- 再用 z_t 在旧隐状态 h_{t-1} 与新候选 n_t 之间做加权插值，得到当前隐状态 h_t。

常见公式（时间步 t）：
    z_t = sigmoid(x_t W_xz + h_{t-1} W_hz + b_z)        # 更新门
    r_t = sigmoid(x_t W_xr + h_{t-1} W_hr + b_r)        # 重置门
    n_t = tanh(  x_t W_xn + (r_t ⊙ h_{t-1}) W_hn + b_n) # 候选隐状态
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}               # 插值更新

形状约定：
- x_t:      (batch, input_size)
- h_{t-1}:  (batch, hidden_size)
- z_t, r_t, n_t, h_t: (batch, hidden_size)
- ⊙ 表示逐元素相乘
"""

from typing import Tuple, Dict, Optional
import numpy as np


class GRU:
    """
    GRU 循环层（单层，无投影），纯 NumPy 实现，面向初学者。

    参数：
    - input_size (int): 每个时间步输入向量的维度（特征数）。
    - hidden_size (int): 隐藏状态的维度（神经元数）。
    - seed (Optional[int]): 随机种子，用于可复现实验（可选）。
    - weight_scale (float): 权重初始化的尺度，默认 0.2（较小便于数值稳定和观察）。

    权重与偏置（按三门拆分，形状如下）：
    - W_xz, W_xr, W_xn: (input_size, hidden_size)     # 输入到各门/候选
    - W_hz, W_hr, W_hn: (hidden_size, hidden_size)    # 隐藏态到各门/候选
    - b_z,  b_r,  b_n : (hidden_size,)                # 各自偏置

    主要方法：
    - forward(x, h0=None, verbose=False) -> (H, h_T)
        处理整段序列，返回所有时间步的隐藏态 H 以及末步隐藏态 h_T。
    - step(x_t, h_prev, verbose=False) -> (h_t, cache)
        执行单个时间步的前向计算，可选打印中间张量（用于学习/调试）。
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 seed: Optional[int] = 42,
                 weight_scale: float = 0.2):
        # --------- 基本维度参数 ---------
        self.input_size = input_size
        self.hidden_size = hidden_size

        # --------- 随机初始化权重与偏置 ---------
        rng = np.random.default_rng(seed)

        # 输入 -> 三门/候选
        self.W_xz = rng.normal(0, weight_scale, size=(input_size, hidden_size))
        self.W_xr = rng.normal(0, weight_scale, size=(input_size, hidden_size))
        self.W_xn = rng.normal(0, weight_scale, size=(input_size, hidden_size))

        # 隐藏态 -> 三门/候选
        self.W_hz = rng.normal(0, weight_scale, size=(hidden_size, hidden_size))
        self.W_hr = rng.normal(0, weight_scale, size=(hidden_size, hidden_size))
        self.W_hn = rng.normal(0, weight_scale, size=(hidden_size, hidden_size))

        # 偏置
        self.b_z = np.zeros((hidden_size,), dtype=np.float32)
        self.b_r = np.zeros((hidden_size,), dtype=np.float32)
        self.b_n = np.zeros((hidden_size,), dtype=np.float32)

    # ====================== 工具函数（激活） ======================
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid 激活函数：σ(x) = 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        """Tanh 激活函数"""
        return np.tanh(x)

    # ====================== 单步计算 ======================
    def step(self,
             x_t: np.ndarray,
             h_prev: np.ndarray,
             verbose: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        执行 GRU 的单个时间步前向计算。

        参数：
        - x_t (ndarray): 当前时间步的输入，形状 (batch, input_size)
        - h_prev (ndarray): 上一时间步的隐藏态 h_{t-1}，形状 (batch, hidden_size)
        - verbose (bool): 若为 True，则打印关键中间张量（学习/调试用）

        返回：
        - h_t (ndarray): 当前时间步的隐藏态，形状 (batch, hidden_size)
        - cache (dict): 缓存中间结果，便于对齐、调试或进一步分析
        """
        # 计算更新门与重置门的线性项（pre-activation）
        z_lin = x_t @ self.W_xz + h_prev @ self.W_hz + self.b_z
        r_lin = x_t @ self.W_xr + h_prev @ self.W_hr + self.b_r

        # 门激活
        z_t = self._sigmoid(z_lin)          # 更新门：控制旧信息保留比例
        r_t = self._sigmoid(r_lin)          # 重置门：控制旧隐藏态参与候选计算的比例

        # 候选隐状态的线性项：对隐藏态先做重置，再和输入一起线性变换
        r_h = r_t * h_prev                   # 逐元素相乘，抑制或放大旧隐态分量
        n_lin = x_t @ self.W_xn + r_h @ self.W_hn + self.b_n
        n_t = self._tanh(n_lin)              # 候选隐状态（新信息）

        # 用更新门对旧隐态与候选隐态做插值，得到当前隐态
        h_t = (1.0 - z_t) * n_t + z_t * h_prev

        # 学习/调试时打印中间张量
        if verbose:
            self._print_step_tensors(x_t, h_prev, z_lin, r_lin, z_t, r_t, r_h, n_lin, n_t, h_t)

        cache = {
            "x_t": x_t, "h_prev": h_prev,
            "z_lin": z_lin, "r_lin": r_lin,
            "z_t": z_t, "r_t": r_t,
            "r_h": r_h, "n_lin": n_lin, "n_t": n_t,
            "h_t": h_t,
        }
        return h_t, cache

    # ====================== 序列前向 ======================
    def forward(self,
                x: np.ndarray,
                h0: Optional[np.ndarray] = None,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理完整序列的前向计算（batch-first）。

        参数：
        - x (ndarray): 输入序列，形状 (batch, seq_len, input_size)
        - h0 (Optional[ndarray]): 初始隐藏态，形状 (batch, hidden_size)，未提供则置零
        - verbose (bool): 若为 True，将在每个时间步打印中间张量（学习/调试）

        返回：
        - H  (ndarray): 所有时间步的隐藏态拼接，形状 (batch, seq_len, hidden_size)
        - hT (ndarray): 最后一个时间步的隐藏态，形状 (batch, hidden_size)
        """
        # 形状检查
        assert x.ndim == 3, "x 必须是三维张量 (batch, seq_len, input_size)"
        batch, seq_len, input_size = x.shape
        assert input_size == self.input_size, f"输入最后一维应为 {self.input_size}, 实际为 {input_size}"

        # 初始化 h
        if h0 is None:
            h_t = np.zeros((batch, self.hidden_size), dtype=np.float32)
        else:
            assert h0.shape == (batch, self.hidden_size)
            h_t = h0.astype(np.float32)

        # 输出容器
        H = np.zeros((batch, seq_len, self.hidden_size), dtype=np.float32)

        # 沿时间维迭代
        for t in range(seq_len):
            x_t = x[:, t, :]                 # 取第 t 个时间步的输入
            h_t, _ = self.step(x_t, h_t, verbose=verbose)
            H[:, t, :] = h_t

        # h_t 即末步隐藏态
        return H, h_t

    # ====================== 打印辅助（学习/调试） ======================
    @staticmethod
    def _print_header(title: str):
        """打印分隔标题，提升可读性"""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    @staticmethod
    def _print_tensor(name: str, x: np.ndarray):
        """打印张量的名称、形状与内容"""
        print(f"{name} | shape={tuple(x.shape)}")
        print(x)
        print("-" * 80)

    def _print_step_tensors(self,
                            x_t, h_prev,
                            z_lin, r_lin,
                            z_t, r_t,
                            r_h, n_lin, n_t,
                            h_t):
        """集中打印单步的所有关键中间张量（仅用于学习/调试）"""
        self._print_header("GRU 单步中间张量（学习/调试）")
        self._print_tensor("x_t", x_t)
        self._print_tensor("h_prev", h_prev)
        self._print_tensor("z_lin (更新门 pre-activation)", z_lin)
        self._print_tensor("r_lin (重置门 pre-activation)", r_lin)
        self._print_tensor("z_t = sigmoid(z_lin) (更新门)", z_t)
        self._print_tensor("r_t = sigmoid(r_lin) (重置门)", r_t)
        self._print_tensor("r_h = r_t * h_prev (重置后旧隐态)", r_h)
        self._print_tensor("n_lin = x_t@W_xn + r_h@W_hn + b_n (候选 pre-activation)", n_lin)
        self._print_tensor("n_t = tanh(n_lin) (候选隐状态)", n_t)
        self._print_tensor("h_t = (1 - z_t)*n_t + z_t*h_prev (当前隐态)", h_t)


# ============================ 演示用主程序 ============================
if __name__ == "__main__":
    # 让打印更易读
    np.set_printoptions(precision=3, suppress=True)

    # -------- 超小规模示例，便于观察 --------
    batch = 2
    seq_len = 4
    input_size = 3
    hidden_size = 5

    # 随机输入序列 (batch, seq_len, input_size)
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1.0, size=(batch, seq_len, input_size)).astype(np.float32)

    # 实例化 GRU
    gru = GRU(input_size=input_size, hidden_size=hidden_size, seed=7)

    # 执行前向（verbose=True 打印每个时间步的详细中间张量）
    H, h_T = gru.forward(x, verbose=True)

    # 打印最终结果
    print("\n" + "#" * 80)
    print("最终输出（整段序列的隐藏态 H、末步隐藏态 h_T）：")
    print("#" * 80)
    print("H 形状:", H.shape)
    print(H)
    print("-" * 80)
    print("h_T 形状:", h_T.shape)
    print(h_T)
