import pandas as pd
import numpy as np
import os

# ================= 配置参数 =================
BASE_DIR = r'D:\大学办公\0毕业论文\核算\2先找数据再价格平减'
INPUT_IO_FILE = os.path.join(BASE_DIR, '2012平减用.xlsx')
PRICE_INDEX_FILE = os.path.join(BASE_DIR, '数值价格指数表.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, '2012平减.xlsx')
PRICE_YEAR = '2012'
PRICE_YEAR_COL_INDEX = 2   # 价格指数表中目标年份所在的列索引（0‑based）
                            # 2012=2, 2017=3, 2023=4
N_SECTORS = 138

# 原始列代码（必须与投入产出表的列标题完全一致）
FINAL_DEMAND_COLS = [
    'FU101', 'FU102', 'THC', 'FU103', 'TC', 'FU201', 'FU202', 'GCF', 'EX', 'TFU'
]
IMPORT_COL = 'IM'
OUTPUT_COL = 'GO'
TII_COL = 'TIU'          # 中间使用合计列代码
VA_CODES = ['VA001', 'VA002', 'VA003', 'VA004']
TII_CODE = 'TII'
TVA_CODE = 'TVA'
TI_CODE = 'TI'
# ===========================================

def main():
    # 1. 读取投入产出表
    print("读取投入产出表...")
    df_io_raw = pd.read_excel(INPUT_IO_FILE, sheet_name='Sheet1', header=None)

    # 定位部门代码行（第2行，索引1）
    sector_codes = df_io_raw.iloc[1, 1:].tolist()
    col_code_to_idx = {code: idx + 1 for idx, code in enumerate(sector_codes) if pd.notna(code)}

    # 中间使用列索引 (1..138)
    intermediate_cols_idx = [col_code_to_idx[i] for i in range(1, N_SECTORS + 1)]
    # 提取中间流量矩阵 (138 x 138)
    intermediate_data = df_io_raw.iloc[2:2 + N_SECTORS, intermediate_cols_idx].values.astype(float)

    # 提取最终需求各列、进口、总产出
    final_demand_data = {}
    for col_name in FINAL_DEMAND_COLS:
        if col_name in col_code_to_idx:
            final_demand_data[col_name] = df_io_raw.iloc[2:2 + N_SECTORS, col_code_to_idx[col_name]].values.astype(float)
        else:
            print(f"警告：未找到列 {col_name}")

    import_data = df_io_raw.iloc[2:2 + N_SECTORS, col_code_to_idx[IMPORT_COL]].values.astype(float)
    total_output_data = df_io_raw.iloc[2:2 + N_SECTORS, col_code_to_idx[OUTPUT_COL]].values.astype(float)

    # 定位增加值行（在部门数据行之后）
    va_row_indices = {}
    for idx in range(2 + N_SECTORS, len(df_io_raw)):
        code = df_io_raw.iloc[idx, 1]
        if code in [TII_CODE] + VA_CODES + [TVA_CODE, TI_CODE]:
            va_row_indices[code] = idx

    original_va = {}
    for code in VA_CODES:
        original_va[code] = df_io_raw.iloc[va_row_indices[code], 2:2 + N_SECTORS].values.astype(float)
    original_tii_row = df_io_raw.iloc[va_row_indices[TII_CODE], 2:2 + N_SECTORS].values.astype(float)

    # 2. 读取价格指数
    print("读取价格指数...")
    df_price_idx = pd.read_excel(PRICE_INDEX_FILE, sheet_name='Sheet1')
    df_price_idx = df_price_idx[df_price_idx['代码'].isin(range(1, N_SECTORS + 1))]
    df_price_idx = df_price_idx.sort_values('代码')
    price_indices = df_price_idx.iloc[:, PRICE_YEAR_COL_INDEX].values  # 按部门顺序的价格指数

    # 3. 记录原始增加值结构
    print("记录增加值结构...")
    original_total_va = np.sum([original_va[code] for code in VA_CODES], axis=0)
    va_ratios = {}
    for code in VA_CODES:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(original_total_va != 0, original_va[code] / original_total_va, 0)
        va_ratios[code] = ratio

    # 4. 行向平减：中间使用 + 最终需求各列 + 进口 + 总产出
    print(f"进行行向价格平减 (基年: 2012, 目标年: {PRICE_YEAR})...")

    # 中间矩阵行向平减
    deflated_intermediate = intermediate_data / price_indices[:, np.newaxis]

    # 最终需求行向平减（每个部门行除以该部门价格指数）
    deflated_final = {}
    for col_name, col_data in final_demand_data.items():
        deflated_final[col_name] = col_data / price_indices

    # 进口、总产出行向平减
    deflated_import = import_data / price_indices
    deflated_total_output = total_output_data / price_indices

    # 5. 重新计算所有合计列（保证行平衡）
    print("重新计算行合计与最终使用合成列...")
    # 中间使用合计（行和）
    new_tiu = np.sum(deflated_intermediate, axis=1)

    # 最终使用合成项
    new_thc = deflated_final['FU101'] + deflated_final['FU102']
    new_tc = new_thc + deflated_final['FU103']
    new_gcf = deflated_final['FU201'] + deflated_final['FU202']
    new_tfu = new_tc + new_gcf + deflated_final['EX']

    # 更新最终需求字典中的合成列
    deflated_final['THC'] = new_thc
    deflated_final['TC'] = new_tc
    deflated_final['GCF'] = new_gcf
    deflated_final['TFU'] = new_tfu

    # 6. 计算新的增加值（列平衡）
    print("计算新的增加值...")
    new_tii = np.sum(deflated_intermediate, axis=0)   # 中间投入合计（列和）
    new_tva = deflated_total_output - new_tii         # 增加值合计
    new_va = {}
    for code in VA_CODES:
        new_va[code] = new_tva * va_ratios[code]
    new_ti = new_tii + new_tva                       # 总投入

    # 7. 组装输出表
    print("组装输出表...")
    output_df = df_io_raw.copy()

    # 写入平减后的中间流量矩阵
    output_df.iloc[2:2 + N_SECTORS, intermediate_cols_idx] = deflated_intermediate

    # 写入中间使用合计列 (TIU)
    if TII_COL in col_code_to_idx:
        output_df.iloc[2:2 + N_SECTORS, col_code_to_idx[TII_COL]] = new_tiu

    # 写入平减后的最终需求各列（包括重新计算的合成列）
    for col_name, col_data in deflated_final.items():
        if col_name in col_code_to_idx:
            output_df.iloc[2:2 + N_SECTORS, col_code_to_idx[col_name]] = col_data

    # 写入平减后的进口和总产出
    output_df.iloc[2:2 + N_SECTORS, col_code_to_idx[IMPORT_COL]] = deflated_import
    output_df.iloc[2:2 + N_SECTORS, col_code_to_idx[OUTPUT_COL]] = deflated_total_output

    # 写入增加值行
    for code in VA_CODES:
        output_df.iloc[va_row_indices[code], 2:2 + N_SECTORS] = new_va[code]
    output_df.iloc[va_row_indices[TII_CODE], 2:2 + N_SECTORS] = new_tii
    if TVA_CODE in va_row_indices:
        output_df.iloc[va_row_indices[TVA_CODE], 2:2 + N_SECTORS] = new_tva
    if TI_CODE in va_row_indices:
        output_df.iloc[va_row_indices[TI_CODE], 2:2 + N_SECTORS] = new_ti

    # 8. 保存结果
    print(f"保存到 {OUTPUT_FILE}...")
    output_df.to_excel(OUTPUT_FILE, index=False, header=False, sheet_name='Sheet1')

    # 9. 平衡检验（行与列）
    print("\n===== 平衡验证 =====")
    # 列平衡：总投入 vs 总产出
    max_col_err = np.max(np.abs(new_ti - deflated_total_output))
    print(f"列平衡 (TI - GO) 最大差异: {max_col_err:.10f}")

    # 行平衡：GO = TIU + TFU - IM
    go_from_row = new_tiu + new_tfu - deflated_import
    max_row_err = np.max(np.abs(go_from_row - deflated_total_output))
    print(f"行平衡 (TIU + TFU - IM - GO) 最大差异: {max_row_err:.10f}")

    if max_col_err < 1e-6 and max_row_err < 1e-6:
        print("✅ 完美平衡！所有行、列恒等式均满足。")
    else:
        print("⚠️ 存在不平衡，请检查。")

    print("全部完成！")

if __name__ == "__main__":
    main()