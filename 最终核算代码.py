import pandas as pd
import numpy as np

# ==================== 配置参数 ====================
years = [2012, 2017, 2023]
base_path = r"D:\大学办公\0毕业论文\核算\4核算"   # 请根据实际路径修改

# 目标部门：油菜种植 (1) 和 菜籽油加工 (15) — 0‑based索引
TARGET_DEPTS = {
    '油菜种植': 1,
    '菜籽油加工': 15
}

N2O_GWP = 273

# 部门名称映射（1‑based → 0‑based）
dept_names_1based = {
    1: "农产品（不含油菜）", 2: "油菜", 3: "林产品", 4: "畜牧产品", 5: "渔产品",
    6: "农、林、牧、渔服务产品", 7: "煤炭开采和洗选产品", 8: "石油和天然气开采产品",
    9: "黑色金属矿采选产品", 10: "有色金属矿采选产品", 11: "非金属矿采选产品",
    12: "开采辅助活动和其他采矿产品", 13: "谷物磨制品", 14: "饲料加工品",
    15: "植物油加工品(不含菜籽油)", 16: "菜籽油加工品", 17: "糖及糖制品",
    18: "屠宰及肉类加工品", 19: "水产加工品", 20: "蔬菜、水果、坚果和其他农副食品加工品",
    21: "方便食品", 22: "乳制品", 23: "调味品、发酵制品", 24: "其他食品",
    25: "酒精和酒", 26: "饮料和精制茶加工品", 27: "烟草制品", 28: "棉、化纤纺织及印染精加工品",
    29: "毛纺织及染整精加工品", 30: "麻、丝绢纺织及加工品", 31: "针织或钩针编织及其制品",
    32: "纺织制成品", 33: "纺织服装服饰", 34: "皮革、毛皮、羽毛及其制品", 35: "鞋",
    36: "木材加工品", 37: "家具", 38: "纸浆和造纸和纸制品", 39: "印刷和记录媒介复制品",
    40: "文教、工美、体育和娱乐用品", 41: "精炼石油和核燃料加工品", 42: "煤炭加工品",
    43: "基础化学原料", 44: "肥料", 45: "农药", 46: "涂料、油墨、颜料及类似产品",
    47: "合成材料", 48: "专用化学产品和炸药、火工、焰火产品", 49: "日用化学产品",
    50: "医药制品", 51: "化学纤维制品", 52: "橡胶制品", 53: "塑料制品",
    54: "水泥、石灰和石膏", 55: "石膏、水泥制品及类似制品", 56: "砖瓦、石材等建筑材料",
    57: "玻璃和玻璃制品", 58: "陶瓷制品", 59: "耐火材料制品", 60: "石墨及其他非金属矿物制品",
    61: "黑色金属冶炼和压延加工业", 62: "有色金属及其合金和铸件", 63: "有色金属压延加工品",
    64: "金属制品", 65: "锅炉及原动设备", 66: "金属加工机械", 67: "物料搬运设备",
    68: "泵、阀门、压缩机及类似机械", 69: "文化、办公用机械", 70: "其他通用设备",
    71: "采矿、冶金、建筑专用设备", 72: "化工、木材、非金属加工专用设备",
    73: "农、林、牧、渔专用机械", 74: "其他专用设备", 75: "汽车整车", 76: "汽车零部件及配件",
    77: "铁路运输和城市轨道交通设备", 78: "船舶及相关装置", 79: "其他交通运输设备",
    80: "电机", 81: "输配电及控制设备", 82: "电线、电缆、光缆及电工器材", 83: "电池",
    84: "家用器具", 85: "其他电气机械和器材", 86: "计算机", 87: "通信设备",
    88: "广播电视设备和雷达及配套设备", 89: "视听设备", 90: "电子元器件", 91: "其他电子设备",
    92: "仪器仪表", 93: "其他制造产品", 94: "废弃资源和废旧材料回收加工品",
    95: "金属制品、机械和设备修理服务", 96: "电力、热力生产和供应", 97: "燃气生产和供应",
    98: "水的生产和供应", 99: "房屋建筑", 100: "土木工程建筑", 101: "建筑安装",
    102: "建筑装饰和其他建筑服务", 103: "批发和零售", 104: "铁路运输", 105: "道路运输",
    106: "水上运输", 107: "航空运输", 108: "管道运输", 109: "多式联运和运输代理和装卸搬运和仓储",
    110: "邮政", 111: "住宿", 112: "餐饮", 113: "电信和其他信息传输服务",
    114: "软件和信息技术服务", 115: "货币金融和其他金融服务", 116: "资本市场服务",
    117: "保险", 118: "房地产", 119: "租赁", 120: "商务服务", 121: "研究和试验发展",
    122: "专业技术服务", 123: "科技推广和应用服务", 124: "水利管理", 125: "生态保护和环境治理",
    126: "公共设施管理", 127: "居民服务", 128: "其他服务", 129: "教育", 130: "卫生",
    131: "社会工作", 132: "新闻和出版", 133: "广播、电视、电影和影视录音制作",
    134: "文化艺术", 135: "体育", 136: "娱乐", 137: "社会保障", 138: "公共管理和社会组织"
}
dept_names = {idx-1: name for idx, name in dept_names_1based.items()}

# ==================== 辅助函数 ====================
def leontief_inv(Z, x):
    """计算列昂惕夫逆矩阵 L = (I - A)^(-1)"""
    n = len(x)
    A = Z / x.reshape(1, -1)
    I = np.identity(n)
    return np.linalg.inv(I - A)

def calc_multipliers(f, L):
    """计算完全排放乘子向量 m = f @ L"""
    return f @ L

def prod_layer_for_dept(f, A, dept_idx, max_layer=10):
    """生产层分解：返回各层贡献列表（层0为直接排放乘子）"""
    e_dept = np.zeros(len(f))
    e_dept[dept_idx] = 1.0
    layers = []
    layers.append(f[dept_idx])
    A_power = np.eye(len(A))
    for k in range(1, max_layer):
        A_power = A_power @ A
        contrib = f @ A_power @ e_dept
        layers.append(contrib)
        if contrib < 1e-6 * sum(layers):
            break
    return layers

def influence_sensitivity(L, f):
    """计算排放影响力系数和感应度系数"""
    m = calc_multipliers(f, L)
    avg_m = np.mean(m)
    influence = m / avg_m
    M = np.diag(f) @ L
    row_sum = M.sum(axis=1)
    avg_row = np.mean(row_sum)
    sensitivity = row_sum / avg_row
    return influence, sensitivity

# ==================== 主程序 ====================
# 存储结果（油菜、菜籽油两个部门，每种排放类型）
results = {dept: {'CO2': [], 'N2O': [], 'Total': []} for dept in TARGET_DEPTS.keys()}
all_intensity = []      # 所有部门的排放强度（用于对比）
all_influence = []      # 所有部门的影响力和感应度（用于对比）
production_layers = {}  # 生产层分解，键为 (year, dept_name, em_type)

for year in years:
    print(f"\n{'='*60}\n处理年份：{year}\n{'='*60}")
    file = f"{base_path}\\{year}EEIOnum.xlsx"
    
    # 读取中间流量矩阵 Z (138x138)
    df_Z = pd.read_excel(file, sheet_name=f"{year}Z", header=None)
    Z = df_Z.values
    n = Z.shape[0]
    
    # 读取排放及经济向量
    df_data = pd.read_excel(file, sheet_name=f"{year}e&x&y", header=0)
    x = df_data['x'].values
    y_rural = df_data['y_rural'].values
    y_urban = df_data['y_urban'].values
    y_g = df_data['y_g'].values
    y_fix = df_data['y_fix'].values
    y_inv = df_data['y_inv'].values
    y_ex = df_data['y_ex'].values
    y_im = df_data['y_im'].values
    
    eCO2 = df_data['eCO2'].values
    eN2O = df_data['eN2O'].values
    eNC = df_data['eNC'].values           # 已折算为CO2当量的总排放
    
    # 国内最终需求（不含出口）
    y_dom = y_rural + y_urban + y_g + y_fix + y_inv
    
    # 分别计算三种排放类型的直接强度 f
    f_CO2 = eCO2 / x
    f_N2O = eN2O / x
    f_Total = eNC / x
    
    # 计算列昂惕夫逆矩阵 L（只需一次）
    L = leontief_inv(Z, x)
    
    # 计算三种排放类型的完全乘子 m
    m_CO2 = calc_multipliers(f_CO2, L)
    m_N2O = calc_multipliers(f_N2O, L)
    m_Total = calc_multipliers(f_Total, L)
    
    # 对每个目标部门进行核算
    for dept_name, dept_idx in TARGET_DEPTS.items():
        # 部门总直接排放
        direct_CO2 = eCO2[dept_idx]
        direct_N2O = eN2O[dept_idx]
        direct_Total = eNC[dept_idx]
        
        # 该部门最终需求引起的直接排放 = f[dept_idx] * y_dom[dept_idx]
        direct_for_final_CO2 = f_CO2[dept_idx] * y_dom[dept_idx]
        direct_for_final_N2O = f_N2O[dept_idx] * y_dom[dept_idx]
        direct_for_final_Total = f_Total[dept_idx] * y_dom[dept_idx]
        
        # 完全排放（国内最终需求拉动）
        total_emb_CO2 = m_CO2[dept_idx] * y_dom[dept_idx]
        total_emb_N2O = m_N2O[dept_idx] * y_dom[dept_idx]
        total_emb_Total = m_Total[dept_idx] * y_dom[dept_idx]
        
        # 间接排放
        indirect_CO2 = total_emb_CO2 - direct_for_final_CO2
        indirect_N2O = total_emb_N2O - direct_for_final_N2O
        indirect_Total = total_emb_Total - direct_for_final_Total
        
        # 存储结果
        results[dept_name]['CO2'].append({
            'Year': year, 'Dept': dept_name,
            'Direct_Total': direct_CO2,
            'Direct_For_Final': direct_for_final_CO2,
            'Indirect': indirect_CO2,
            'Embodied_Domestic': total_emb_CO2
        })
        results[dept_name]['N2O'].append({
            'Year': year, 'Dept': dept_name,
            'Direct_Total': direct_N2O,
            'Direct_For_Final': direct_for_final_N2O,
            'Indirect': indirect_N2O,
            'Embodied_Domestic': total_emb_N2O
        })
        results[dept_name]['Total'].append({
            'Year': year, 'Dept': dept_name,
            'Direct_Total': direct_Total,
            'Direct_For_Final': direct_for_final_Total,
            'Indirect': indirect_Total,
            'Embodied_Domestic': total_emb_Total
        })
    
    # 所有部门的排放强度（用于对比分析）
    for i in range(n):
        dept_name_i = dept_names.get(i, f"部门{i+1}")
        all_intensity.append({
            'Year': year, 'Dept_Index': i, 'Dept_Name': dept_name_i,
            'CO2_Direct_Intensity': f_CO2[i],
            'CO2_Complete_Multiplier': m_CO2[i],
            'N2O_Direct_Intensity': f_N2O[i],
            'N2O_Complete_Multiplier': m_N2O[i],
            'Total_Direct_Intensity': f_Total[i],
            'Total_Complete_Multiplier': m_Total[i]
        })
    
    # 排放影响力系数和感应度系数（基于总CO2当量）
    influence, sensitivity = influence_sensitivity(L, f_Total)
    for i in range(n):
        dept_name_i = dept_names.get(i, f"部门{i+1}")
        all_influence.append({
            'Year': year, 'Dept_Index': i, 'Dept_Name': dept_name_i,
            'Influence_Coeff': influence[i],
            'Sensitivity_Coeff': sensitivity[i]
        })
    
    # 生产层分解（针对每个目标部门，分三种排放类型）
    A = Z / x.reshape(1, -1)
    for dept_name, dept_idx in TARGET_DEPTS.items():
        for em_type, f_vec in [('CO2', f_CO2), ('N2O', f_N2O), ('Total', f_Total)]:
            layers = prod_layer_for_dept(f_vec, A, dept_idx, max_layer=10)
            production_layers[(year, dept_name, em_type)] = layers

# ==================== 打印结果 ====================
print("\n\n" + "="*80)
print("第四章 油菜与菜籽油产业链温室气体排放核算结果（基于国内最终需求）")
print("="*80)

# 4.1.1 不同类别排放量
print("\n【4.1.1 不同类别排放量（直接、间接、完全）】")
print("-"*80)
for dept_name in TARGET_DEPTS.keys():
    print(f"\n{'='*40}")
    print(f"部门：{dept_name}")
    print(f"{'='*40}")
    for em_type in ['CO2', 'N2O', 'Total']:
        type_label = 'CO₂' if em_type=='CO2' else 'N₂O' if em_type=='N2O' else 'CO₂当量（CO₂+N₂O折算）'
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

# 4.1.3 排放强度对比（完全乘子）
print("\n【4.1.3 排放强度对比（完全乘子，吨/万元）】")
print("-"*80)
for year in years:
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
print("-"*80)
for year in years:
    for dept_name in TARGET_DEPTS.keys():
        for em_type in ['CO2', 'N2O', 'Total']:
            layers = production_layers[(year, dept_name, em_type)]
            total_m = sum(layers)
            type_label = 'CO₂' if em_type=='CO2' else 'N₂O' if em_type=='N2O' else 'CO₂当量'
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

# 4.2.3 关键传输部门
print("\n【4.2.3 关键传输部门（排放影响力系数 & 感应度系数，基于CO₂当量）】")
print("-"*80)
for year in years:
    print(f"\n年份：{year}")
    inf_df = pd.DataFrame([d for d in all_influence if d['Year']==year])
    for dept_name, dept_idx in TARGET_DEPTS.items():
        row = inf_df[inf_df['Dept_Index']==dept_idx].iloc[0]
        print(f"  {dept_name}：影响力={row['Influence_Coeff']:.4f}, 感应度={row['Sensitivity_Coeff']:.4f}")
    high_inf = inf_df[inf_df['Influence_Coeff'] > 1.5].nlargest(5, 'Influence_Coeff')
    print("  影响力系数最高的5个部门（>1.5）：")
    for _, row in high_inf.iterrows():
        print(f"    {row['Dept_Name']}: {row['Influence_Coeff']:.4f}")
    high_sens = inf_df[inf_df['Sensitivity_Coeff'] > 1.5].nlargest(5, 'Sensitivity_Coeff')
    print("  感应度系数最高的5个部门（>1.5）：")
    for _, row in high_sens.iterrows():
        print(f"    {row['Dept_Name']}: {row['Sensitivity_Coeff']:.4f}")

# ==================== 导出Excel ====================
with pd.ExcelWriter(f"{base_path}\\chapter4_rapeseed_oil_results.xlsx") as writer:
    for dept_name in TARGET_DEPTS.keys():
        for em_type in ['CO2', 'N2O', 'Total']:
            sheet_name = f"{dept_name}_{em_type}"
            df = pd.DataFrame(results[dept_name][em_type])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    pd.DataFrame(all_intensity).to_excel(writer, sheet_name='各部门排放强度', index=False)
    pd.DataFrame(all_influence).to_excel(writer, sheet_name='各部门影响力感应度', index=False)
    # 生产层分解结果
    layers_data = []
    for (year, dept_name, em_type), layers in production_layers.items():
        for i, val in enumerate(layers):
            layers_data.append({
                'Year': year, 'Dept': dept_name, 'Emission_Type': em_type,
                'Layer': i, 'Value': val
            })
    pd.DataFrame(layers_data).to_excel(writer, sheet_name='生产层分解', index=False)

print(f"\n详细结果已导出至 {base_path}\\chapter4_rapeseed_oil_results.xlsx")
print("\n代码运行完毕。")
