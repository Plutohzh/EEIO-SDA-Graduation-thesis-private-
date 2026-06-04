# -*- coding: utf-8 -*-
"""
SDA 结构分解分析（两极分解法）
基于环境扩展投入产出表，分解油菜产业链温室气体排放变化的驱动因素
包含：油菜种植、菜籽油加工、整个经济系统
排放口径：CO₂、N₂O、CO₂当量总量
时间段：2012-2017，2017-2023，2012-2023
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==================== 参数配置 ====================
BASE_PATH = Path(r"D:\大学办公\0毕业论文\核算\5SDA新修正单位版_不平减验算")   # 请根据实际路径修改
YEARS = [2012, 2017, 2023]
N2O_GWP = 273

# 目标部门索引（0‑based）
TARGET_DEPTS = {
    '油菜种植': 1,
    '菜籽油加工': 15
}

# 排放类型
EM_TYPES = ['CO2', 'N2O', 'Total']

# 全局部门名称（将在运行时从文件读取）
DEPT_NAMES = None

# ==================== 数据加载与预处理 ====================
def load_sector_names(year):
    """从给定年份的e&x&y表中读取部门名称列表（138个）"""
    file_path = BASE_PATH / f"{year}EEIOnum.xlsx"
    df = pd.read_excel(file_path, sheet_name=f"{year}e&x&y", header=0)
    return df['部门名称'].tolist()

def load_year_data(year, em_type):
    """
    加载指定年份、指定排放类型的数据，返回SDA所需的所有变量
    em_type: 'CO2', 'N2O', 'Total'
    返回字典包含：
        f: 直接排放强度向量
        L: 列昂惕夫逆矩阵
        y_dom: 国内最终需求向量
        Y_total: 国内最终需求总量（标量）
        s: 部门最终需求份额向量
        u: 完全排放乘子向量（f @ L）
        x: 总产出向量
        Z: 中间流量矩阵
    """
    file_path = BASE_PATH / f"{year}EEIOnum.xlsx"
    
    # 读取中间流量矩阵 Z (138 x 138)
    Z = pd.read_excel(file_path, sheet_name=f"{year}Z", header=None).values
    n = Z.shape[0]
    
    # 读取环境扩展及最终需求数据
    df_env = pd.read_excel(file_path, sheet_name=f"{year}e&x&y", header=0)
    # 提取对应的排放向量
    if em_type == 'CO2':
        e = df_env['eCO2'].values
    elif em_type == 'N2O':
        e = df_env['eN2O'].values
    else:   # Total
        e = df_env['eNC'].values   # 已折算为CO₂当量
    
    x = df_env['x'].values
    y_rural = df_env['y_rural'].values
    y_urban = df_env['y_urban'].values
    y_g = df_env['y_g'].values
    y_fix = df_env['y_fix'].values
    y_inv = df_env['y_inv'].values
    
    # 国内最终需求（不包括出口）
    y_dom = y_rural + y_urban + y_g + y_fix + y_inv
    Y_total = y_dom.sum()
    s = y_dom / Y_total if Y_total != 0 else np.zeros_like(y_dom)
    
    # 直接排放强度
    f = e / x
    f = np.nan_to_num(f)   # 处理可能出现的除零（若x=0，则强度为0）
    
    # 直接消耗系数矩阵 A = Z * diag(1/x)
    X_inv = np.diag(1.0 / x)
    A = Z @ X_inv
    I = np.eye(n)
    L = np.linalg.inv(I - A)   # 列昂惕夫逆矩阵
    
    # 完全排放乘子 u = f @ L
    u = f @ L
    
    return {
        'f': f, 'L': L, 'y_dom': y_dom, 'Y_total': Y_total, 's': s, 'u': u,
        'x': x, 'Z': Z, 'n': n
    }

# ==================== 两极分解函数 ====================
def sda_total_economy(data0, data1):
    """
    两极分解法 - 整个经济系统的总排放变化
    ΔTE = f1 L1 y1 - f0 L0 y0
    分解为：
        强度效应：Δf = (f1 - f0) L0 y0 和 (f1 - f0) L1 y1 的均值
        技术效应：ΔL = f1 (L1 - L0) y0 和 f0 (L1 - L0) y1 的均值
        需求规模效应：ΔY = (Y1 - Y0) * (f1 L1 s0 + f0 L0 s1)/2
        需求结构效应：Δs = (f1 L1 Y1 (s1 - s0) + f0 L0 Y0 (s1 - s0))/2
    返回字典 {效应名: 值}
    """
    f0, L0, y0, Y0, s0 = data0['f'], data0['L'], data0['y_dom'], data0['Y_total'], data0['s']
    f1, L1, y1, Y1, s1 = data1['f'], data1['L'], data1['y_dom'], data1['Y_total'], data1['s']
    
    # 顺序一：基期为权重
    delta_f1 = (f1 - f0) @ L0 @ y0
    delta_L1 = f1 @ (L1 - L0) @ y0
    delta_Y1 = (f1 @ L1) @ ((Y1 - Y0) * s0)
    delta_s1 = (f1 @ L1) @ (Y1 * (s1 - s0))
    
    # 顺序二：报告期为权重
    delta_f2 = (f1 - f0) @ L1 @ y1
    delta_L2 = f0 @ (L1 - L0) @ y1
    delta_Y2 = (f0 @ L0) @ ((Y1 - Y0) * s1)
    delta_s2 = (f0 @ L0) @ (Y0 * (s1 - s0))
    
    # 平均值
    delta_f = (delta_f1 + delta_f2) / 2.0
    delta_L = (delta_L1 + delta_L2) / 2.0
    delta_Y = (delta_Y1 + delta_Y2) / 2.0
    delta_s = (delta_s1 + delta_s2) / 2.0
    
    return {
        '强度效应': delta_f,
        '技术效应': delta_L,
        '需求规模效应': delta_Y,
        '需求结构效应': delta_s
    }

def sda_sector(data0, data1, dept_idx):
    """
    两极分解法 - 单个部门最终需求引致的排放变化
    ΔTE_j = (u_j1 * y_j1) - (u_j0 * y_j0)
    其中 u_j = (f L)_j
    分解公式基于部门j的最终需求变化及其乘子变化
    返回字典 {效应名: 值}
    """
    f0, L0, y0, Y0, s0 = data0['f'], data0['L'], data0['y_dom'], data0['Y_total'], data0['s']
    f1, L1, y1, Y1, s1 = data1['f'], data1['L'], data1['y_dom'], data1['Y_total'], data1['s']
    n = data0['n']
    
    # 目标部门的值
    yj0 = y0[dept_idx]
    yj1 = y1[dept_idx]
    uj0 = (f0 @ L0)[dept_idx]
    uj1 = (f1 @ L1)[dept_idx]
    sj0 = yj0 / Y0 if Y0 != 0 else 0.0
    sj1 = yj1 / Y1 if Y1 != 0 else 0.0
    
    # 构建单位向量 e_j
    e_j = np.zeros(n)
    e_j[dept_idx] = 1.0
    
    # 顺序一：基期为权重
    delta_f1 = (f1 - f0) @ L0 @ (yj0 * e_j)
    delta_L1 = f1 @ (L1 - L0) @ (yj0 * e_j)
    delta_Y1 = (Y1 - Y0) * sj0 * (f1 @ L1 @ e_j)
    delta_s1 = Y1 * (sj1 - sj0) * (f1 @ L1 @ e_j)
    
    # 顺序二：报告期为权重
    delta_f2 = (f1 - f0) @ L1 @ (yj1 * e_j)
    delta_L2 = f0 @ (L1 - L0) @ (yj1 * e_j)
    delta_Y2 = (Y1 - Y0) * sj1 * (f0 @ L0 @ e_j)
    delta_s2 = Y0 * (sj1 - sj0) * (f0 @ L0 @ e_j)
    
    # 平均值
    delta_f = (delta_f1 + delta_f2) / 2.0
    delta_L = (delta_L1 + delta_L2) / 2.0
    delta_Y = (delta_Y1 + delta_Y2) / 2.0
    delta_s = (delta_s1 + delta_s2) / 2.0
    
    return {
        '强度效应': delta_f,
        '技术效应': delta_L,
        '需求规模效应': delta_Y,
        '需求结构效应': delta_s
    }

# ==================== 主程序 ====================
def main():
    global DEPT_NAMES
    # 获取部门名称（以2012年为准，假设每年顺序一致）
    DEPT_NAMES = load_sector_names(YEARS[0])
    print(f"部门总数: {len(DEPT_NAMES)}")
    for name, idx in TARGET_DEPTS.items():
        print(f"目标部门: {name} -> 索引 {idx} (名称: {DEPT_NAMES[idx]})")
    print()
    
    # 存储所有分解结果（用于导出Excel）
    all_results = []
    
    # 遍历三种排放口径
    for em_type in EM_TYPES:
        print(f"\n{'='*70}")
        print(f"排放口径: {em_type}")
        print(f"{'='*70}")
        
        # 加载三个年份的数据
        data = {}
        for yr in YEARS:
            print(f"加载 {yr} 年数据...")
            data[yr] = load_year_data(yr, em_type)
            print(f"  国内最终需求总量 Y = {data[yr]['Y_total']:,.2f} 万元")
            total_emb = data[yr]['u'] @ data[yr]['y_dom']
            print(f"  经济系统总排放（国内需求拉动）= {total_emb:,.2f} 吨")
            for dept_name, idx in TARGET_DEPTS.items():
                yj = data[yr]['y_dom'][idx]
                uj = data[yr]['u'][idx]
                te_j = uj * yj
                print(f"  {dept_name}: 最终需求={yj:,.2f} 万元, 完全乘子={uj:.6f}, 诱排放={te_j:,.2f} 吨")
        print()
        
        # 定义时间段
        periods = [(2012, 2017), (2017, 2023), (2012, 2023)]
        
        for (y0, y1) in periods:
            print(f"--- {y0} → {y1} ---")
            # 整个经济系统
            eff_total = sda_total_economy(data[y0], data[y1])
            te0_total = data[y0]['u'] @ data[y0]['y_dom']
            te1_total = data[y1]['u'] @ data[y1]['y_dom']
            delta_total = te1_total - te0_total
            sum_eff = sum(eff_total.values())
            print(f"  整个经济系统: ΔTE = {delta_total:,.2f} 吨 (分解和={sum_eff:,.2f}, 残差={delta_total - sum_eff:.10f})")
            for name, val in eff_total.items():
                pct = (val / delta_total * 100) if delta_total != 0 else 0.0
                print(f"    {name}: {val:,.2f} 吨 ({pct:+.2f}%)")
            all_results.append({
                "Period": f"{y0}-{y1}", "Emission": em_type, "Object": "整个经济系统",
                "Intensity": eff_total["强度效应"], "Technology": eff_total["技术效应"],
                "Scale": eff_total["需求规模效应"], "Structure": eff_total["需求结构效应"],
                "TotalChange": delta_total
            })
            
            # 各目标部门
            for dept_name, idx in TARGET_DEPTS.items():
                eff_sector = sda_sector(data[y0], data[y1], idx)
                te0_sector = data[y0]['u'][idx] * data[y0]['y_dom'][idx]
                te1_sector = data[y1]['u'][idx] * data[y1]['y_dom'][idx]
                delta_sector = te1_sector - te0_sector
                sum_eff_s = sum(eff_sector.values())
                print(f"  {dept_name}: ΔTE = {delta_sector:,.2f} 吨 (分解和={sum_eff_s:,.2f}, 残差={delta_sector - sum_eff_s:.10f})")
                for name, val in eff_sector.items():
                    pct = (val / delta_sector * 100) if delta_sector != 0 else 0.0
                    print(f"    {name}: {val:,.2f} 吨 ({pct:+.2f}%)")
                all_results.append({
                    "Period": f"{y0}-{y1}", "Emission": em_type, "Object": dept_name,
                    "Intensity": eff_sector["强度效应"], "Technology": eff_sector["技术效应"],
                    "Scale": eff_sector["需求规模效应"], "Structure": eff_sector["需求结构效应"],
                    "TotalChange": delta_sector
                })
            print()
    
    # 导出Excel
    df_results = pd.DataFrame(all_results)
    output_path = BASE_PATH / "SDA_results_all.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        df_results.to_excel(writer, sheet_name="SDA分解汇总", index=False)
        # 按对象分sheet
        for obj in df_results["Object"].unique():
            df_obj = df_results[df_results["Object"] == obj]
            sheet_name = obj[:31]  # Excel sheet名长度限制
            df_obj.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\n所有SDA分解结果已保存至: {output_path}")
    print("程序运行完毕。")

if __name__ == "__main__":
    main()