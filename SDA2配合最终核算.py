# -*- coding: utf-8 -*-
"""
SDA 分解分析（两极分解法）
支持：油菜种植、菜籽油加工、整个经济系统
口径：CO2, N2O, 总CO2当量 (eNC)
年份：2012, 2017, 2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 参数设置 ====================
DATA_DIR = Path(r"D:\大学办公\0毕业论文\SDA")   # 请根据实际路径修改
YEARS = [2012, 2017, 2023]
TARGET_DEPTS = {
    "油菜种植": 1,      # 0‑based 索引
    "菜籽油加工": 15
}
EMISSION_TYPES = {
    "CO2": 0,      # eCO2 列索引
    "N2O": 1,      # eN2O 列索引
    "Total": 2     # eNC 列索引
}
GWP_N2O = 273      # 仅用于显示，实际已用 eNC

# 字体设置（用于图表）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 辅助函数 ====================
def load_sector_names(file_path):
    """从第一个sheet读取部门名称列表（138个）"""
    df = pd.read_excel(file_path, sheet_name=0, header=1)
    names = df.iloc[:, 0].tolist()
    return [str(n).strip() for n in names]

def load_year_data(year, sector_names, emission_col):
    """
    加载指定年份数据，返回分解所需的全部变量
    emission_col: 0=CO2, 1=N2O, 2=Total
    """
    file_path = DATA_DIR / f"{year}EEIOnum.xlsx"
    
    # 读取中间使用矩阵
    Z = pd.read_excel(file_path, sheet_name=f"{year}Z", header=None).values
    n = Z.shape[0]
    
    # 读取环境与最终需求数据
    df_env = pd.read_excel(file_path, sheet_name=f"{year}e&x&y")
    # 列顺序: eCO2(0), eN2O(1), eNC(2), x(3), y_rural(4), y_urban(5), y_g(6), y_fix(7), y_inv(8), y_ex(9), y_im(10)
    e = df_env.iloc[:, emission_col].values
    x = df_env.iloc[:, 3].values
    y_rural = df_env.iloc[:, 4].values
    y_urban = df_env.iloc[:, 5].values
    y_g = df_env.iloc[:, 6].values
    y_fix = df_env.iloc[:, 7].values
    y_inv = df_env.iloc[:, 8].values
    
    # 国内最终需求
    y_dom = y_rural + y_urban + y_g + y_fix + y_inv
    Y_total = y_dom.sum()
    s = y_dom / Y_total if Y_total != 0 else np.zeros_like(y_dom)
    
    # 排放强度
    f = e / x
    f = np.nan_to_num(f)
    
    # 直接消耗系数 A 和列昂惕夫逆 L
    X_inv = np.diag(1.0 / x)
    A = Z @ X_inv
    I = np.eye(n)
    L = np.linalg.inv(I - A)
    
    # 完全排放乘子 u
    u = f @ L
    return {
        "f": f, "L": L, "y_dom": y_dom, "Y_total": Y_total, "s": s, "u": u,
        "x": x, "Z": Z, "n": n
    }

def get_target_index(sector_names, dept_name):
    """根据部门名称获取0‑based索引"""
    try:
        return sector_names.index(dept_name)
    except ValueError:
        raise ValueError(f"未找到部门: {dept_name}")

def sda_polar_decomposition_total(data0, data1):
    """
    两极分解法 - 整个经济系统总排放变化
    ΔTE = f1 L1 y1 - f0 L0 y0
    分解为：强度效应 + 技术效应 + 需求规模效应 + 需求结构效应
    返回字典包含四个效应（标量）
    """
    f0, L0, y0, Y0, s0 = data0["f"], data0["L"], data0["y_dom"], data0["Y_total"], data0["s"]
    f1, L1, y1, Y1, s1 = data1["f"], data1["L"], data1["y_dom"], data1["Y_total"], data1["s"]
    
    # 顺序一 (0 → 1)
    delta_f1 = (f1 - f0) @ L0 @ y0
    delta_L1 = f1 @ (L1 - L0) @ y0
    delta_Y1 = (f1 @ L1) @ ((Y1 - Y0) * s0)
    delta_s1 = (f1 @ L1) @ (Y1 * (s1 - s0))
    
    # 顺序二 (1 → 0)
    delta_f2 = (f1 - f0) @ L1 @ y1
    delta_L2 = f0 @ (L1 - L0) @ y1
    delta_Y2 = (f0 @ L0) @ ((Y1 - Y0) * s1)
    delta_s2 = (f0 @ L0) @ (Y0 * (s1 - s0))
    
    # 平均
    delta_f = (delta_f1 + delta_f2) / 2.0
    delta_L = (delta_L1 + delta_L2) / 2.0
    delta_Y = (delta_Y1 + delta_Y2) / 2.0
    delta_s = (delta_s1 + delta_s2) / 2.0
    
    return {
        "强度效应": float(delta_f),
        "技术效应": float(delta_L),
        "需求规模效应": float(delta_Y),
        "需求结构效应": float(delta_s)
    }

def sda_polar_decomposition_sector(data0, data1, target_idx):
    """
    两极分解法 - 单个部门的最终需求所诱发的排放变化
    ΔTE_j = (u_j^1 * y_j^1) - (u_j^0 * y_j^0)
    其中 u_j = (f L)_j
    分解公式基于部门j的最终需求变化和乘子变化
    """
    f0, L0, y0, Y0, s0 = data0["f"], data0["L"], data0["y_dom"], data0["Y_total"], data0["s"]
    f1, L1, y1, Y1, s1 = data1["f"], data1["L"], data1["y_dom"], data1["Y_total"], data1["s"]
    
    # 提取目标部门的最终需求值
    yj0 = y0[target_idx]
    yj1 = y1[target_idx]
    # 目标部门的完全乘子
    uj0 = (f0 @ L0)[target_idx]
    uj1 = (f1 @ L1)[target_idx]
    # 目标部门的份额
    sj0 = yj0 / Y0 if Y0 != 0 else 0.0
    sj1 = yj1 / Y1 if Y1 != 0 else 0.0
    
    # 构建单位向量 e_j
    n = len(f0)
    e_j = np.zeros(n)
    e_j[target_idx] = 1.0
    
    # 顺序一 (0 → 1)
    delta_f1 = (f1 - f0) @ L0 @ (yj0 * e_j)
    delta_L1 = f1 @ (L1 - L0) @ (yj0 * e_j)
    delta_Y1 = (Y1 - Y0) * sj0 * (f1 @ L1 @ e_j)
    delta_s1 = Y1 * (sj1 - sj0) * (f1 @ L1 @ e_j)
    
    # 顺序二 (1 → 0)
    delta_f2 = (f1 - f0) @ L1 @ (yj1 * e_j)
    delta_L2 = f0 @ (L1 - L0) @ (yj1 * e_j)
    delta_Y2 = (Y1 - Y0) * sj1 * (f0 @ L0 @ e_j)
    delta_s2 = Y0 * (sj1 - sj0) * (f0 @ L0 @ e_j)
    
    # 平均
    delta_f = (delta_f1 + delta_f2) / 2.0
    delta_L = (delta_L1 + delta_L2) / 2.0
    delta_Y = (delta_Y1 + delta_Y2) / 2.0
    delta_s = (delta_s1 + delta_s2) / 2.0
    
    return {
        "强度效应": float(delta_f),
        "技术效应": float(delta_L),
        "需求规模效应": float(delta_Y),
        "需求结构效应": float(delta_s)
    }

def plot_effects(effects, total_change, title, save_path):
    """绘制效应贡献柱状图"""
    labels = list(effects.keys())
    values = list(effects.values())
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('排放变化 (吨 CO₂当量)')
    plt.title(title + f'\n总变动: {total_change:,.0f} 吨')
    for bar, val in zip(bars, values):
        y_pos = bar.get_height() / 2 if bar.get_height() >= 0 else bar.get_height() / 2
        plt.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:,.0f}', ha='center', va='center', fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==================== 主程序 ====================
def main():
    # 读取部门名称（以第一个年份文件为准）
    first_file = DATA_DIR / f"{YEARS[0]}EEIOnum.xlsx"
    sector_names = load_sector_names(first_file)
    print(f"部门总数: {len(sector_names)}\n")
    
    # 获取目标部门的索引
    target_indices = {}
    for name, idx_0based in TARGET_DEPTS.items():
        # 这里直接使用配置中的索引，因为部门名称可能不完全匹配
        target_indices[name] = idx_0based
        print(f"目标部门: {name} -> 索引 {idx_0based}")
    print()
    
    # 存储所有分解结果
    all_results = []  # 每条记录包含：年份区间、口径、对象、各效应值
    
    # 遍历三种排放口径
    for em_name, em_col in EMISSION_TYPES.items():
        print(f"\n{'='*60}")
        print(f"排放口径: {em_name}")
        print(f"{'='*60}")
        
        # 加载三个年份的数据
        data = {}
        for yr in YEARS:
            print(f"加载 {yr} 年数据...")
            data[yr] = load_year_data(yr, sector_names, em_col)
            print(f"  国内总需求 Y = {data[yr]['Y_total']:,.2f} 万元")
            print(f"  总排放量 = {(data[yr]['u'] @ data[yr]['y_dom']):,.2f} 吨")
            for dept_name, idx in target_indices.items():
                yj = data[yr]['y_dom'][idx]
                uj = data[yr]['u'][idx]
                te_j = uj * yj
                print(f"  {dept_name}: 最终需求={yj:,.2f} 万元, 完全乘子={uj:.6f}, 诱发排放={te_j:,.2f} 吨")
        print()
        
        # 时间段
        periods = [(2012, 2017), (2017, 2023), (2012, 2023)]
        
        for (y0, y1) in periods:
            print(f"--- {y0} → {y1} ---")
            # 整个经济系统
            eff_total = sda_polar_decomposition_total(data[y0], data[y1])
            te0_total = data[y0]['u'] @ data[y0]['y_dom']
            te1_total = data[y1]['u'] @ data[y1]['y_dom']
            delta_total = te1_total - te0_total
            sum_eff = sum(eff_total.values())
            print(f"  整个经济系统: ΔTE = {delta_total:,.2f} 吨 (效应和={sum_eff:,.2f}, 残差={delta_total-sum_eff:.6f})")
            for name, val in eff_total.items():
                pct = (val / delta_total * 100) if delta_total != 0 else 0.0
                print(f"    {name}: {val:,.2f} 吨 ({pct:+.2f}%)")
            all_results.append({
                "Period": f"{y0}-{y1}", "Emission": em_name, "Object": "整个经济系统",
                "Intensity": eff_total["强度效应"], "Technology": eff_total["技术效应"],
                "Scale": eff_total["需求规模效应"], "Structure": eff_total["需求结构效应"],
                "TotalChange": delta_total
            })
            
            # 各目标部门
            for dept_name, idx in target_indices.items():
                eff_sector = sda_polar_decomposition_sector(data[y0], data[y1], idx)
                te0_sector = data[y0]['u'][idx] * data[y0]['y_dom'][idx]
                te1_sector = data[y1]['u'][idx] * data[y1]['y_dom'][idx]
                delta_sector = te1_sector - te0_sector
                sum_eff_s = sum(eff_sector.values())
                print(f"  {dept_name}: ΔTE = {delta_sector:,.2f} 吨 (效应和={sum_eff_s:,.2f}, 残差={delta_sector-sum_eff_s:.6f})")
                for name, val in eff_sector.items():
                    pct = (val / delta_sector * 100) if delta_sector != 0 else 0.0
                    print(f"    {name}: {val:,.2f} 吨 ({pct:+.2f}%)")
                all_results.append({
                    "Period": f"{y0}-{y1}", "Emission": em_name, "Object": dept_name,
                    "Intensity": eff_sector["强度效应"], "Technology": eff_sector["技术效应"],
                    "Scale": eff_sector["需求规模效应"], "Structure": eff_sector["需求结构效应"],
                    "TotalChange": delta_sector
                })
            
            # 画图（为每个对象每个口径每个时段画一张图，可选；为避免太多图，只画几个关键的）
            # 此处可选，按需注释。我们只画总经济系统和油菜种植的图作为示例
            if em_name == "Total":   # 仅总当量口径画图
                # 整个经济系统
                title = f"整个经济系统 SDA 分解 ({y0}→{y1})"
                save_path = DATA_DIR / f"SDA_{y0}_{y1}_总系统.png"
                plot_effects(eff_total, delta_total, title, save_path)
                # 油菜种植
                eff_rapeseed = sda_polar_decomposition_sector(data[y0], data[y1], target_indices["油菜种植"])
                te0_rap = data[y0]['u'][target_indices["油菜种植"]] * data[y0]['y_dom'][target_indices["油菜种植"]]
                te1_rap = data[y1]['u'][target_indices["油菜种植"]] * data[y1]['y_dom'][target_indices["油菜种植"]]
                delta_rap = te1_rap - te0_rap
                title_rap = f"油菜种植 SDA 分解 ({y0}→{y1})"
                save_path_rap = DATA_DIR / f"SDA_{y0}_{y1}_油菜.png"
                plot_effects(eff_rapeseed, delta_rap, title_rap, save_path_rap)
                # 菜籽油加工
                eff_oil = sda_polar_decomposition_sector(data[y0], data[y1], target_indices["菜籽油加工"])
                te0_oil = data[y0]['u'][target_indices["菜籽油加工"]] * data[y0]['y_dom'][target_indices["菜籽油加工"]]
                te1_oil = data[y1]['u'][target_indices["菜籽油加工"]] * data[y1]['y_dom'][target_indices["菜籽油加工"]]
                delta_oil = te1_oil - te0_oil
                title_oil = f"菜籽油加工 SDA 分解 ({y0}→{y1})"
                save_path_oil = DATA_DIR / f"SDA_{y0}_{y1}_菜籽油.png"
                plot_effects(eff_oil, delta_oil, title_oil, save_path_oil)
            print()
    
    # 导出 Excel 汇总表
    df_results = pd.DataFrame(all_results)
    excel_path = DATA_DIR / "SDA_results_all.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_results.to_excel(writer, sheet_name="SDA分解结果", index=False)
        # 也可以按对象分 sheet
        for obj in df_results["Object"].unique():
            df_obj = df_results[df_results["Object"] == obj]
            sheet_name = obj[:31]  # Excel sheet name 长度限制
            df_obj.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\n所有结果已导出至: {excel_path}")
    print("程序运行完毕。")

if __name__ == "__main__":
    main()
