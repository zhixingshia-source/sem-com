# -*- coding: utf-8 -*-
"""
Multi-Method E2E Latency Plots
- 统一口径：T_E2E = T_enc + T_tx + T_dec
- 支持方法类型:
  type='ours'     -> payload_bits = K * D * q + header_bits
  type='file'     -> payload_bits = filesize_bytes * 8
  type='bpp'      -> payload_bits = bpp * H * W
  type='symbols'  -> payload_bits = n_symbols * bits_per_symbol
- code_rate 默认 1.0；若使用纠错码，等效负载 bits *= 1/code_rate
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 如需屏幕显示可删掉
import matplotlib.pyplot as plt

# ============== 1) 你需要填的数据（示例是占位，为了能跑通图） ==============
# 建议统一在“同等语义质量（如 CLIP≈0.90）”下记录各方法的 enc/dec/体积
METHODS = {
    "Ours-Adapt(K≈841,q=8,D=4)": {
        "type": "ours", "K": 841, "D": 4, "q": 8, "header_bits": 128,
        "enc_ms": 65, "dec_ms": 25, "code_rate": 1.0, "clip": 0.90
    },
    "Ours-Fixed(K=1600,q=8,D=4)": {
        "type": "ours", "K": 1600, "D": 4, "q": 8, "header_bits": 128,
        "enc_ms": 70, "dec_ms": 28, "code_rate": 1.0, "clip": 0.92
    },
    "JPEG(Q=xx)": {                 # 传统编解码：直接用文件大小
        "type": "file", "filesize_bytes": 42000,
        "enc_ms": 8, "dec_ms": 2, "code_rate": 1.0, "clip": 0.90
    },
    "AVIF(crf=xx)": {
        "type": "file", "filesize_bytes": 26000,
        "enc_ms": 42, "dec_ms": 6, "code_rate": 1.0, "clip": 0.90
    },
    "JPEG XL(d=xx)": {
        "type": "file", "filesize_bytes": 30000,
        "enc_ms": 25, "dec_ms": 5, "code_rate": 1.0, "clip": 0.90
    },
    "ELIC(bpp≈0.15)": {            # 学习式：按 bpp×HW
        "type": "bpp", "bpp": 0.15, "H": 512, "W": 512,
        "enc_ms": 70, "dec_ms": 55, "code_rate": 1.0, "clip": 0.90
    },
    "DeepJSCC(SNR=10dB)": {        # JSCC：符号数量×每符号比特
        "type": "symbols", "n_symbols": 11000, "bits_per_symbol": 4,
        "enc_ms": 20, "dec_ms": 20, "code_rate": 0.8, "clip": 0.90
    },
}

# 带宽档位（Mbps）
BWS = [1, 5, 10, 50]

# ============== 2) Payload 统一计算 ==============
def payload_bits_of(m):
    t = m["type"]
    if t == "ours":
        return int(m["K"] * m["D"] * m["q"] + m.get("header_bits", 0))
    if t == "file":
        return int(m["filesize_bytes"] * 8)
    if t == "bpp":
        return float(m["bpp"] * m["H"] * m["W"])
    if t == "symbols":
        return int(m["n_symbols"] * m["bits_per_symbol"])
    raise ValueError("unknown method type")

def e2e_components(m, bw_mbps):
    bits = payload_bits_of(m)
    code_rate = float(m.get("code_rate", 1.0))
    tx_s = (bits / (bw_mbps * 1e6)) * (1.0 / code_rate)  # seconds
    enc_s = float(m["enc_ms"]) / 1000.0
    dec_s = float(m["dec_ms"]) / 1000.0
    return enc_s, tx_s, dec_s, enc_s + tx_s + dec_s

# ============== 3) 画图函数 ==============
def plot_stacked_at_bw(methods, bw_mbps, outdir="latency_charts"):
    os.makedirs(outdir, exist_ok=True)
    names = list(methods.keys())

    enc = []; tx = []; dec = []; total = []; clipv=[]
    for k in names:
        e, t, d, tot = e2e_components(methods[k], bw_mbps)
        enc.append(e); tx.append(t); dec.append(d); total.append(tot)
        clipv.append(methods[k].get("clip", None))
    order = np.argsort(total)  # 从小到大
    names = [names[i] for i in order]
    enc   = [enc[i]   for i in order]
    tx    = [tx[i]    for i in order]
    dec   = [dec[i]   for i in order]
    total = [total[i] for i in order]
    clipv = [clipv[i] for i in order]

    x = np.arange(len(names))
    plt.figure(figsize=(12,6))
    plt.bar(x, enc, label="Encode", color="#8ecae6")
    plt.bar(x, tx,  bottom=enc, label="Transmit", color="#ffb3ba")
    plt.bar(x, dec, bottom=np.array(enc)+np.array(tx), label="Decode", color="#bde0fe")
    plt.plot(x, total, "o-r", lw=2, label="Total E2E")

    # 标注 CLIP（若给了）
    for i, tot in enumerate(total):
        if clipv[i] is not None:
            plt.text(i, tot+0.005, f"CLIP {clipv[i]:.2f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, names, rotation=20)
    plt.ylabel("End-to-End Latency [seconds]")
    plt.title(f"E2E Latency by Method @ {bw_mbps} Mbps")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    fn = os.path.join(outdir, f"e2e_methods_{bw_mbps}Mbps.png")
    plt.tight_layout(); plt.savefig(fn, dpi=200); plt.close()
    print(f"Saved: {fn}")

def plot_e2e_vs_bw(methods, bws=BWS, outdir="latency_charts"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12,6))
    for name, m in methods.items():
        totals = []
        for bw in bws:
            _, _, _, tot = e2e_components(m, bw)
            totals.append(tot)
        plt.plot(bws, totals, "o-", lw=2, label=name)
    plt.xscale("log")  # 带宽跨距大，用对数更清晰
    plt.xlabel("Link Bandwidth [Mbps] (log scale)")
    plt.ylabel("End-to-End Latency [seconds]")
    plt.title("E2E Latency vs Bandwidth")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    fn = os.path.join(outdir, "e2e_vs_bandwidth.png")
    plt.tight_layout(); plt.savefig(fn, dpi=200); plt.close()
    print(f"Saved: {fn}")

def plot_ours_K_sweep(K_list, base, bws=BWS, outdir="latency_charts"):
    """
    对‘我们的方法’扫不同 K，观察 K-时延权衡（q/D/enc/dec 可按经验近似线性缩放或填测量值）
    base: 我们方法的一个字典，作为模板（会覆盖K）
    """
    os.makedirs(outdir, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0,1,len(bws)))
    plt.figure(figsize=(12,6))
    for j,bw in enumerate(bws):
        totals=[]
        for K in K_list:
            m = dict(base); m["K"] = K
            _, _, _, tot = e2e_components(m, bw)
            totals.append(tot)
        plt.plot(K_list, totals, "o-", lw=2, color=colors[j], label=f"{bw} Mbps")
    plt.xlabel("Token Count K")
    plt.ylabel("End-to-End Latency [seconds]")
    plt.title("Ours: K vs E2E Latency across Bandwidths")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fn = os.path.join(outdir, "ours_K_sweep.png")
    plt.tight_layout(); plt.savefig(fn, dpi=200); plt.close()
    print(f"Saved: {fn}")

# ============== 4) 运行：输出三类图 ==============
if __name__ == "__main__":
    # A) 固定带宽下的“堆叠条+总时延折线”（对比所有方法）
    for bw in [1, 5, 10, 50]:
        plot_stacked_at_bw(METHODS, bw)

    # B) 每个方法“时延-带宽”曲线（对数带宽轴）
    plot_e2e_vs_bw(METHODS, BWS)

    # C) 我们方法扫不同 K（把 base 方法挑一个 ours）
    base_ours = [v for v in METHODS.values() if v["type"]=="ours"][0]
    plot_ours_K_sweep(K_list=[196, 400, 784, 1024, 1600, 2401, 3306, 4096],
                      base=base_ours, bws=[1,5,10,50])
