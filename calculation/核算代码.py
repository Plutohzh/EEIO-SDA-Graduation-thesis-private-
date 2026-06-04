import pandas as pd
import numpy as np
import os

# ==================== 配置参数 ====================
YEARS = [2012, 2017, 2023]
BASE_PATH = r"D:\大学办公\0毕业论文\核算\4核算新修正单位版"   # 请根据实际路径修改
N2O_GWP = 273

# 目标部门索引（0‑based）
TARGET_DEPTS = {
    '油菜种植': 1,
    '菜籽油加工': 15
}

# 部门名称映射（0‑based）
# 此处从2012年数据中提取，实际可从文件读取，为简便直接给出
DEPT_NAMES = None   # 后面动态从文件读取

# ==================== 辅助函数 ====================
def load_data(year):
    """加载指定年份的Z矩阵和e&x&y表"""
    file_path = os.path.join(BASE_PATH, f"{year}EEIOnum.xlsx")
    Z = pd.read_excel(file_path, sheet_name=f"{year}Z", header=None).values
    df = pd.read_excel(file_path, sheet_name=f"{year}e&x&y", header=0)
    return Z, df

def calc_technical_coefficient(Z, x):
    """计算直接消耗系数矩阵 A_{ij} = Z_{ij} / x_j"""
    x = x.reshape(1, -1)
    return Z / x

def leontief_inv(A):
    """计算列昂惕夫逆矩阵 L = (I - A)^(-1)"""
    n = A.shape[0]
    I = np.eye(n)
    return np.linalg.inv(I - A)

def direct_intensity(e, x):
    """计算直接排放强度向量 f = e / x"""
    return e / x

def complete_multiplier(f, L):
    """计算完全排放乘子 m = f @ L"""
    return f @ L

def production_layer_decomposition(f, A, dept_idx, max_layer=10):
    """
    生产层分解：返回完全乘子 m_j 的各层贡献列表（层0为直接排放乘子）
    即 m_j = f_j + (f A)_j + (f A^2)_j + ...
    """
    n = len(f)
    e_dept = np.zeros(n)
    e_dept[dept_idx] = 1.0
    layers = []
    # 层0
    layers.append(f[dept_idx])
    # 层1到max_layer-1
    A_power = np.eye(n)
    for k in range(1, max_layer):
        A_power = A_power @ A
        contrib = f @ A_power @ e_dept
        layers.append(contrib)
        # 当贡献小于总和的1e-6时提前终止
        if contrib < 1e-6 * sum(layers):
            break
    return layers

def influence_sensitivity(f, L):
    """计算排放影响力系数和感应度系数（基于CO₂当量）"""
    m = complete_multiplier(f, L)
    avg_m = np.mean(m)
    influence = m / avg_m          # 影响力系数
    
    M = np.diag(f) @ L
    row_sum = M.sum(axis=1)
    avg_row = np.mean(row_sum)
    sensitivity = row_sum / avg_row  # 感应度系数
    return influence, sensitivity

# ==================== 主程序 ====================
def main():
    # 存储结果
    results = {dept: {'CO2': [], 'N2O': [], 'Total': []} for dept in TARGET_DEPTS}
    all_intensity = []      # 各部门排放强度（用于对比）
    all_influence = []      # 各部门影响力/感应度
    production_layers = {}  # 生产层分解结果，键为 (year, dept_name, em_type)
    
    # 全局部门名称表（从第一年读取，假设每年顺序相同）
    global DEPT_NAMES
    _, df_first = load_data(YEARS[0])
    DEPT_NAMES = df_first['部门名称'].values   # 长度为138的一维数组
    
    for year in YEARS:
        print(f"\n处理年份：{year}")
        Z, df = load_data(year)
        
        # 提取基本向量
        x = df['x'].values
        eCO2 = df['eCO2'].values
        eN2O = df['eN2O'].values
        eNC = df['eNC'].values   # 已折算为CO₂当量
        # 国内最终需求（不含出口）
        y_dom = (df['y_rural'] + df['y_urban'] + df['y_g'] +
                 df['y_fix'] + df['y_inv']).values
        
        # 计算直接消耗系数矩阵和列昂惕夫逆矩阵
        A = calc_technical_coefficient(Z, x)
        L = leontief_inv(A)
        
        # 计算三种排放类型的直接强度向量
        f_CO2 = direct_intensity(eCO2, x)
        f_N2O = direct_intensity(eN2O, x)
        f_Total = direct_intensity(eNC, x)
        
        # 计算完全乘子
        m_CO2 = complete_multiplier(f_CO2, L)
        m_N2O = complete_multiplier(f_N2O, L)
        m_Total = complete_multiplier(f_Total, L)
        
        # 对每个目标部门计算排放量
        for dept_name, dept_idx in TARGET_DEPTS.items():
            # 部门总直接排放
            direct_total = {
                'CO2': eCO2[dept_idx],
                'N2O': eN2O[dept_idx],
                'Total': eNC[dept_idx]
            }
            # 由本部门最终需求引起的直接排放
            direct_final = {
                'CO2': f_CO2[dept_idx] * y_dom[dept_idx],
                'N2O': f_N2O[dept_idx] * y_dom[dept_idx],
                'Total': f_Total[dept_idx] * y_dom[dept_idx]
            }
            # 完全排放
            embodied = {
                'CO2': m_CO2[dept_idx] * y_dom[dept_idx],
                'N2O': m_N2O[dept_idx] * y_dom[dept_idx],
                'Total': m_Total[dept_idx] * y_dom[dept_idx]
            }
            # 间接排放
            indirect = {
                'CO2': embodied['CO2'] - direct_final['CO2'],
                'N2O': embodied['N2O'] - direct_final['N2O'],
                'Total': embodied['Total'] - direct_final['Total']
            }
            
            # 保存结果
            for em_type in ['CO2', 'N2O', 'Total']:
                results[dept_name][em_type].append({
                    'Year': year,
                    'Dept': dept_name,
                    'Direct_Total': direct_total[em_type],
                    'Direct_For_Final': direct_final[em_type],
                    'Indirect': indirect[em_type],
                    'Embodied_Domestic': embodied[em_type]
                })
        
        # 所有部门的排放强度（用于对比）
        for i in range(len(x)):
            all_intensity.append({
                'Year': year,
                'Dept_Index': i,
                'Dept_Name': DEPT_NAMES[i],
                'CO2_Direct_Intensity': f_CO2[i],
                'CO2_Complete_Multiplier': m_CO2[i],
                'N2O_Direct_Intensity': f_N2O[i],
                'N2O_Complete_Multiplier': m_N2O[i],
                'Total_Direct_Intensity': f_Total[i],
                'Total_Complete_Multiplier': m_Total[i]
            })
        
        # 排放影响力系数和感应度系数（基于总CO₂当量）
        influence, sensitivity = influence_sensitivity(f_Total, L)
        for i in range(len(x)):
            all_influence.append({
                'Year': year,
                'Dept_Index': i,
                'Dept_Name': DEPT_NAMES[i],
                'Influence_Coeff': influence[i],
                'Sensitivity_Coeff': sensitivity[i]
            })
        
        # 生产层分解（针对每个目标部门和每种排放类型）
        for dept_name, dept_idx in TARGET_DEPTS.items():
            for em_type, f_vec in [('CO2', f_CO2), ('N2O', f_N2O), ('Total', f_Total)]:
                layers = production_layer_decomposition(f_vec, A, dept_idx, max_layer=10)
                production_layers[(year, dept_name, em_type)] = layers
    
    # ==================== 打印结果 ====================
    print("\n" + "="*80)
    print("第四章 油菜与菜籽油产业链温室气体排放核算结果（基于国内最终需求）")
    print("="*80)
    
    # 4.1.1 不同类别排放量
    print("\n【4.1.1 不同类别排放量（直接、间接、完全）】")
    for dept_name in TARGET_DEPTS:
        print(f"\n{'='*40}\n部门：{dept_name}\n{'='*40}")
        for em_type in ['CO2', 'N2O', 'Total']:
            type_label = {'CO2':'CO₂', 'N2O':'N₂O', 'Total':'CO₂当量（CO₂+N₂O折算）'}[em_type]
            print(f"\n  排放类型：{type_label}")
            for rec in results[dept_name][em_type]:
                yr = rec['Year']
                print(f"    年份：{yr}")
                print(f"      部门总直接排放：{rec['Direct_Total']:,.2f} 吨")
                print(f"      其中由本部门最终需求引起的直接排放：{rec['Direct_For_Final']:,.2f} 吨")
                print(f"      间接排放（上游供应链）：{rec['Indirect']:,.2f} 吨")
                print(f"      完全排放（国内需求拉动）：{rec['Embodied_Domestic']:,.2f} 吨")
                if rec['Direct_For_Final'] > 0:
                    ratio = rec['Indirect'] / rec['Direct_For_Final']
                    print(f"      间接/直接（最终需求部分）比例：{ratio:.2f}")
    
    # 4.1.3 排放强度对比
    print("\n【4.1.3 排放强度对比（完全乘子，吨/万元）】")
    for year in YEARS:
        print(f"\n年份：{year}")
        inten_df = pd.DataFrame([d for d in all_intensity if d['Year']==year])
        for dept_name, dept_idx in TARGET_DEPTS.items():
            row = inten_df[inten_df['Dept_Index']==dept_idx].iloc[0]
            print(f"  {dept_name}：")
            print(f"    CO₂完全乘子：{row['CO2_Complete_Multiplier']:.4f}")
            print(f"    N₂O完全乘子：{row['N2O_Complete_Multiplier']:.4f}")
            print(f"    CO₂当量完全乘子：{row['Total_Complete_Multiplier']:.4f}")
        # 完全乘子最高的5个部门（基于CO₂当量）
        top5 = inten_df.nlargest(5, 'Total_Complete_Multiplier')[['Dept_Name', 'Total_Complete_Multiplier']]
        print("  完全乘子（CO₂当量）最高的5个部门：")
        for _, row in top5.iterrows():
            print(f"    {row['Dept_Name']}: {row['Total_Complete_Multiplier']:.4f}")
    
    # 4.2.1 生产层分布特征
    print("\n【4.2.1 生产层分布特征（完全乘子分解，单位：吨/万元）】")
    for year in YEARS:
        for dept_name in TARGET_DEPTS:
            for em_type in ['CO2', 'N2O', 'Total']:
                layers = production_layers[(year, dept_name, em_type)]
                total_m = sum(layers)
                type_label = {'CO2':'CO₂', 'N2O':'N₂O', 'Total':'CO₂当量'}[em_type]
                print(f"\n{year}年 - {dept_name} - {type_label}")
                print(f"  完全乘子 = {total_m:.6f}")
                for i, val in enumerate(layers):
                    pct = val/total_m*100 if total_m>0 else 0
                    print(f"    层{i}（{'直接' if i==0 else f'{i}阶间接'}）: {val:.6f} ({pct:.2f}%)")
                cum = 0
                for i, val in enumerate(layers):
                    cum += val
                    if cum/total_m >= 0.9:
                        print(f"  累积贡献90%所需层数: {i+1}")
                        break
    
    # 4.2.2 关键传输部门
    print("\n【4.2.2 关键传输部门（排放影响力系数 & 感应度系数，基于CO₂当量）】")
    for year in YEARS:
        print(f"\n年份：{year}")
        inf_df = pd.DataFrame([d for d in all_influence if d['Year']==year])
        for dept_name, dept_idx in TARGET_DEPTS.items():
            row = inf_df[inf_df['Dept_Index']==dept_idx].iloc[0]
            print(f"  {dept_name}：影响力={row['Influence_Coeff']:.4f}, 感应度={row['Sensitivity_Coeff']:.4f}")
        # 影响力系数最高的5个部门
        high_inf = inf_df.nlargest(5, 'Influence_Coeff')
        print("  影响力系数最高的5个部门：")
        for _, row in high_inf.iterrows():
            print(f"    {row['Dept_Name']}: {row['Influence_Coeff']:.4f}")
        high_sens = inf_df.nlargest(5, 'Sensitivity_Coeff')
        print("  感应度系数最高的5个部门：")
        for _, row in high_sens.iterrows():
            print(f"    {row['Dept_Name']}: {row['Sensitivity_Coeff']:.4f}")
    
    # ==================== 导出Excel ====================
    output_path = os.path.join(BASE_PATH, "chapter4_rapeseed_oil_results.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        for dept_name in TARGET_DEPTS:
            for em_type in ['CO2', 'N2O', 'Total']:
                sheet_name = f"{dept_name}_{em_type}"
                df_out = pd.DataFrame(results[dept_name][em_type])
                df_out.to_excel(writer, sheet_name=sheet_name, index=False)
        pd.DataFrame(all_intensity).to_excel(writer, sheet_name='各部门排放强度', index=False)
        pd.DataFrame(all_influence).to_excel(writer, sheet_name='各部门影响力感应度', index=False)
        # 生产层分解
        layers_data = []
        for (year, dept_name, em_type), layers in production_layers.items():
            for i, val in enumerate(layers):
                layers_data.append({
                    'Year': year, 'Dept': dept_name, 'Emission_Type': em_type,
                    'Layer': i, 'Value': val
                })
        pd.DataFrame(layers_data).to_excel(writer, sheet_name='生产层分解', index=False)
    
    print(f"\n详细结果已导出至：{output_path}")
    print("\n代码运行完毕。")

if __name__ == "__main__":
    main()