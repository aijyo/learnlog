# attention_debug_cn.py
# 说明：
# - 纯 NumPy 实现的注意力与多头注意力（MHA）
# - 全中文注释；运行时逐步打印关键中间结果，便于调试与理解
# - 包含：自注意力（可选因果mask）与 交叉注意力 的演示函数
# 依赖：
#   pip install numpy

import numpy as np
from typing import Optional, Tuple

np.random.seed(7)  # 固定随机种子，便于复现打印结果


def stable_softmax(x: np.ndarray, axis: int = -1, verbose: bool = False, name: str = "softmax") -> np.ndarray:
    """
    数值稳定的 softmax 实现：
    - 为防止指数溢出，先减去每行（按指定轴）最大值再做 exp
    参数：
      x: 输入张量
      axis: softmax 归一化的维度，默认最后一维
      verbose: 是否打印详细中间信息
      name: 打印信息的标识名
    返回：
      与 x 同形状的 softmax 结果
    """
    if verbose:
        print(f"【{name}】输入张量形状:", x.shape)
        # 展示第一条数据（按最后一维重排方便看）
        print(f"【{name}】输入张量示例(前1条):\n", np.round(x.reshape(-1, x.shape[-1])[:1], 4))
    # 数值稳定：减去最大值
    x_max = np.max(x, axis=axis, keepdims=True)
    if verbose:
        print(f"【{name}】按轴最大值(用于数值稳定)(前1条):\n", np.round(x_max.reshape(-1, x_max.shape[-1])[:1], 4))
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True)
    out = e / s
    if verbose:
        print(f"【{name}】输出张量形状:", out.shape)
        # 按行求和应≈1
        print(f"【{name}】每行和(应≈1):\n", np.round(out.sum(axis=axis), 6))
    return out


def make_causal_mask(Lq: int, Lk: int) -> np.ndarray:
    """
    生成因果 mask（下三角允许，上三角屏蔽）：
    - 形状 [Lq, Lk] 的布尔矩阵
    - True 表示“屏蔽”（不允许注意）
    """
    q_idx = np.arange(Lq)[:, None]
    k_idx = np.arange(Lk)[None, :]
    return (k_idx > q_idx)  # 上三角为 True（屏蔽）


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    verbose: bool = False,
    name: str = "SDPA",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    缩放点积注意力(Scaled Dot-Product Attention)：
      scores = (Q @ K^T) / sqrt(Dh)
      attn   = softmax(scores)
      out    = attn @ V
    参数：
      Q: [B, H, Lq, Dh]
      K: [B, H, Lk, Dh]
      V: [B, H, Lk, Dh]
      mask: 可广播到 [B, H, Lq, Lk] 的布尔张量；True=屏蔽
      verbose: 是否打印详细中间信息
      name: 打印信息标识
    返回：
      out: [B, H, Lq, Dh] 加权输出
      attn: [B, H, Lq, Lk] 注意力权重
    """
    if verbose:
        print(f"\n======【{name}】缩放点积注意力 开始======")
        print("Q形状:", Q.shape, "K形状:", K.shape, "V形状:", V.shape)

    Dh = Q.shape[-1]
    # 计算注意力分数（未加 mask）
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(Dh)  # [B,H,Lq,Lk]
    if verbose:
        print("未加mask的scores形状:", scores.shape)
        print("未加mask的scores示例(取[0,0]头):\n", np.round(scores[0, 0], 4))

    # 应用 mask（True=屏蔽，设为一个极小值，softmax 后≈0）
    if mask is not None:
        if verbose:
            print("mask形状(广播前):", mask.shape, "（True=屏蔽）")
        scores = np.where(mask, -1e9, scores)
        if verbose:
            print("加mask后的scores示例(取[0,0]头):\n", np.round(scores[0, 0], 4))

    # softmax 变成注意力权重
    attn_weights = stable_softmax(scores, axis=-1, verbose=verbose, name=f"{name}-softmax")
    if verbose:
        print(f"{name}注意力权重示例(取[0,0]头):\n", np.round(attn_weights[0, 0], 4))

    # 对 V 做加权求和
    out = np.matmul(attn_weights, V)
    if verbose:
        print("加权求和后的输出形状:", out.shape)
        print("输出示例(取[0,0]头):\n", np.round(out[0, 0], 4))
        print(f"======【{name}】结束======\n")
    return out, attn_weights


class MultiHeadAttention:
    """
    多头注意力（MHA）：
      1) 线性投影得到 Q/K/V
      2) 切分为多头，做缩放点积注意力
      3) 合并多头，做输出投影
    """
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dh = d_model // num_heads
        k = 1.0 / np.sqrt(d_model)
        # 初始化权重（均匀分布）
        self.Wq = np.random.uniform(-k, k, size=(d_model, d_model)).astype(np.float32)
        self.Wk = np.random.uniform(-k, k, size=(d_model, d_model)).astype(np.float32)
        self.Wv = np.random.uniform(-k, k, size=(d_model, d_model)).astype(np.float32)
        self.Wo = np.random.uniform(-k, k, size=(d_model, d_model)).astype(np.float32)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """将 [B, L, D] 切分为 [B, H, L, dh]"""
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, self.dh)
        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """将 [B, H, L, dh] 合并回 [B, L, D]"""
        B, H, L, dh = x.shape
        x = np.transpose(x, (0, 2, 1, 3)).reshape(B, L, H * dh)
        return x

    def __call__(
        self,
        x_q: np.ndarray,
        x_kv: Optional[np.ndarray] = None,
        attn_mask: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MHA 前向：
          - x_q: [B, Lq, D]
          - x_kv: [B, Lk, D]（为 None 时表示自注意力，使用 x_q）
          - attn_mask: 可广播到 [B, H, Lq, Lk] 的布尔 mask
          - verbose: 打印详细中间信息
        返回：
          y: [B, Lq, D]
          attn: [B, H, Lq, Lk]
        """
        if x_kv is None:
            x_kv = x_q  # 自注意力

        if verbose:
            print("\n==========【MHA Forward】开始==========")
            print("输入 x_q 形状:", x_q.shape, "输入 x_kv 形状:", x_kv.shape)

        # 线性投影
        Q = x_q @ self.Wq
        K = x_kv @ self.Wk
        V = x_kv @ self.Wv
        if verbose:
            print("线性投影后 Q/K/V 形状:", Q.shape, K.shape, V.shape)
            # 展示一条数据，便于对齐
            print("Q 示例(前1条):\n", np.round(Q.reshape(-1, Q.shape[-1])[:1], 4))
            print("K 示例(前1条):\n", np.round(K.reshape(-1, K.shape[-1])[:1], 4))
            print("V 示例(前1条):\n", np.round(V.reshape(-1, V.shape[-1])[:1], 4))

        # 分头
        Qh = self._split_heads(Q)  # [B,H,Lq,dh]
        Kh = self._split_heads(K)  # [B,H,Lk,dh]
        Vh = self._split_heads(V)  # [B,H,Lk,dh]
        if verbose:
            print("分头后 Qh/Kh/Vh 形状:", Qh.shape, Kh.shape, Vh.shape, "(B,H,L,dh)")

        # 缩放点积注意力
        Oh, attn = scaled_dot_product_attention(Qh, Kh, Vh, attn_mask, verbose=verbose, name="MHA-SDPA")

        # 合并头
        O = self._combine_heads(Oh)  # [B,Lq,D]
        if verbose:
            print("合并头后 O 形状:", O.shape, "(B,L,D)")

        # 输出投影
        y = O @ self.Wo  # [B,Lq,D]
        if verbose:
            print("输出投影后 y 形状:", y.shape)
            print("==========【MHA Forward】结束==========\n")
        return y, attn


# =======================
# Demo：自注意力与交叉注意力
# =======================
def run_demo_self_attention(
    B: int = 2, L: int = 4, d_model: int = 8, num_heads: int = 2, use_causal: bool = False, verbose: bool = True
):
    """
    演示：自注意力（可切换因果遮蔽）
    参数：
      B: 批大小
      L: 序列长度
      d_model: 模型维度
      num_heads: 头数
      use_causal: True=使用因果 mask（禁止看未来）
      verbose: 打印详细中间信息
    """
    print("\n>>>> 运行自注意力示例  run_demo_self_attention(...)")
    print("参数: B=", B, "L=", L, "d_model=", d_model, "num_heads=", num_heads, "use_causal=", use_causal)
    x = np.random.randn(B, L, d_model).astype(np.float32)
    print("输入 x 形状:", x.shape, "x 示例(前1条):\n", np.round(x.reshape(-1, d_model)[:1], 4))

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    attn_mask = None
    if use_causal:
        attn_mask = make_causal_mask(L, L)[None, None, :, :]  # [1,1,L,L] 可广播
        print("使用因果 mask，形状:", attn_mask.shape)

    y, attn = mha(x, attn_mask=attn_mask, verbose=verbose)
    print("最终输出 y 形状:", y.shape)
    print("注意力权重形状:", attn.shape, "(B,H,L,L)")
    # 展示一个头的注意力
    print("注意力权重(取 [batch=0, head=0]):\n", np.round(attn[0, 0], 4))
    return y, attn


def run_demo_cross_attention(
    B: int = 2, Lq: int = 3, Lk: int = 5, d_model: int = 8, num_heads: int = 2, verbose: bool = True
):
    """
    演示：交叉注意力（Q 与 KV 来自不同序列）
    参数：
      B: 批大小
      Lq: Query 序列长度
      Lk: Key/Value 序列长度
      d_model: 模型维度
      num_heads: 头数
      verbose: 打印详细中间信息
    """
    print("\n>>>> 运行交叉注意力示例  run_demo_cross_attention(...)")
    print("参数: B=", B, "Lq=", Lq, "Lk=", Lk, "d_model=", d_model, "num_heads=", num_heads)

    x_q = np.random.randn(B, Lq, d_model).astype(np.float32)
    x_kv = np.random.randn(B, Lk, d_model).astype(np.float32)
    print("输入 x_q 形状:", x_q.shape, "x_kv 形状:", x_kv.shape)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    y, attn = mha(x_q, x_kv=x_kv, attn_mask=None, verbose=verbose)
    print("最终输出 y 形状:", y.shape)
    print("注意力权重形状:", attn.shape, "(B,H,Lq,Lk)")
    print("注意力权重(取 [batch=0, head=0]):\n", np.round(attn[0, 0], 4))
    return y, attn


if __name__ == "__main__":
    # 自注意力（无因果遮蔽）
    _ = run_demo_self_attention(B=2, L=4, d_model=8, num_heads=2, use_causal=False, verbose=True)
    # 自注意力（开启因果遮蔽）
    _ = run_demo_self_attention(B=2, L=4, d_model=8, num_heads=2, use_causal=True, verbose=True)
    # 交叉注意力
    _ = run_demo_cross_attention(B=2, Lq=3, Lk=5, d_model=8, num_heads=2, verbose=True)
