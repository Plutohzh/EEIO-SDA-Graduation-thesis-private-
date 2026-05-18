import pandas as pd
import numpy as np
from pathlib import Path

# ========================= 配置 =========================
BASE_DIR = Path(r"D:\大学办公\0毕业论文\核算\1合表")

FILE_MAPPING = BASE_DIR / "部门映射关系.xlsx"
FILE_TONGBIAO2 = BASE_DIR / "通表2.xlsx"          # 包含VA和CIG的原始数据
OUTPUT_FILE = BASE_DIR / "合并后增加值与最终使用.xlsx"

# 年份与对应工作表名称的映射
VA_SHEETS = {2012: "2012VA", 2017: "2017VA", 2023: "2023VA"}
CIG_SHEETS = {2012: "2012CIG", 2017: "2017CIG", 2023: "2023CIG"}

# 映射表中的相关列名
SHEET_MAPPING = "Sheet1"
COL_NEW_NAME = "新部门分类"       # 新部门名称
COL_MAP_2012 = "对应2012"
COL_MAP_2017 = "对应2017"
COL_MAP_2023 = "对应2023"
# =======================================================


def parse_mapping(filepath):
    """
    读取部门映射关系表，返回形如：
    { 年份: { 新部门名: [原始部门代码列表] } }
    """
    df = pd.read_excel(filepath, sheet_name=SHEET_MAPPING)
    # 选取有效列并去除新部门名称为空的行
    df = df[[COL_NEW_NAME, COL_MAP_2012, COL_MAP_2017, COL_MAP_2023]].dropna(subset=[COL_NEW_NAME])

    mapping = {2012: {}, 2017: {}, 2023: {}}
    for _, row in df.iterrows():
        new_name = str(row[COL_NEW_NAME]).strip()
        if not new_name:
            continue

        for year, col in [(2012, COL_MAP_2012), (2017, COL_MAP_2017), (2023, COL_MAP_2023)]:
            raw = str(row[col]).strip()
            if raw in ("", "nan"):
                continue
            # 解析可能由 '+' 分隔的多个部门代码
            codes = []
            for part in raw.split('+'):
                part = part.strip()
                if part.isdigit():
                    codes.append(int(part))
            if codes:
                mapping[year][new_name] = codes
    return mapping


def read_va_sheet(filepath, sheet_name):
    """
    读取增加值（VA）工作表。
    VA表结构：行是增加值项目（劳动者报酬/生产税净额等），列是原始部门代码（数字）。
    返回 DataFrame: index = 增加值项目, columns = 部门代码(int), values = 数值
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=0)

    # 第一列通常是增加值项目名称，将其设为行索引
    # 先保存第一列的值，然后删除该列
    index_vals = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    # 删除列名为 '代码' 的列（如果存在）
    if '代码' in df.columns:
        df = df.drop(columns=['代码'])

    # 只保留能够转换为整数的列（即部门代码列）
    numeric_cols = []
    for col in df.columns:
        try:
            # 尝试转换为整数（有些列名可能是 '1' 字符串或 1 整数）
            int_col = int(col)
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue

    df = df[numeric_cols]
    # 将列名统一转为整数
    df.columns = [int(c) for c in df.columns]
    df.index = index_vals

    # 确保数值为浮点型，缺失值补0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def read_cig_sheet(filepath, sheet_name):
    """
    读取最终使用（CIG）工作表。
    CIG表结构：行是原始部门代码（数字），列是最终使用类别（农村消费、政府消费等）。
    返回 DataFrame: index = 部门代码(int), columns = 最终使用类别, values = 数值
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, index_col=0)
    # 索引（部门代码）可能为字符串或数字，统一转为整数
    try:
        df.index = df.index.astype(int)
    except ValueError:
        # 若存在非数字索引（如总计行），先过滤掉
        df = df[df.index.map(lambda x: str(x).isdigit())]
        df.index = df.index.astype(int)

    # 确保所有数值为浮点型，缺失值补0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def merge_va(va_df, mapping_dict):
    """
    合并增加值表（列合并）。
    va_df: 原始增加值DataFrame，行=增加值项目，列=原始部门代码
    mapping_dict: { 新部门名: [原始代码列表] }
    返回合并后的DataFrame：行不变，列=新部门名
    """
    new_names = list(mapping_dict.keys())
    merged = pd.DataFrame(0.0, index=va_df.index, columns=new_names)

    for new_name, code_list in mapping_dict.items():
        # 筛选出实际存在于va_df中的原始代码列
        valid_codes = [c for c in code_list if c in va_df.columns]
        if not valid_codes:
            continue
        # 对应列按行求和，赋值给新列
        merged[new_name] = va_df[valid_codes].sum(axis=1)

    return merged


def merge_cig(cig_df, mapping_dict):
    """
    合并最终使用表（行合并）。
    cig_df: 原始最终使用DataFrame，index=原始部门代码，列=最终使用类别
    mapping_dict: { 新部门名: [原始代码列表] }
    返回合并后的DataFrame：index=新部门名，列不变
    """
    new_names = list(mapping_dict.keys())
    # 初始化结果DataFrame，行名为新部门，列与原表相同
    merged = pd.DataFrame(0.0, index=new_names, columns=cig_df.columns)

    for new_name, code_list in mapping_dict.items():
        # 筛选出实际存在于cig_df索引中的部门代码
        valid_codes = [c for c in code_list if c in cig_df.index]
        if not valid_codes:
            continue
        # 对应行按列求和
        merged.loc[new_name] = cig_df.loc[valid_codes].sum(axis=0)

    return merged


def main():
    print("读取部门映射关系...")
    mapping = parse_mapping(FILE_MAPPING)

    # 准备写入结果文件
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # ---------- 处理增加值（VA） ----------
        for year, sheet_name in VA_SHEETS.items():
            print(f"处理 {year} 年增加值表 (sheet: {sheet_name})...")
            va_raw = read_va_sheet(FILE_TONGBIAO2, sheet_name)
            va_merged = merge_va(va_raw, mapping[year])
            output_sheet = f"{year}_VA"
            va_merged.to_excel(writer, sheet_name=output_sheet)
            print(f"  合并后增加值维度: {va_merged.shape}")

        # ---------- 处理最终使用（CIG） ----------
        for year, sheet_name in CIG_SHEETS.items():
            print(f"处理 {year} 年最终使用表 (sheet: {sheet_name})...")
            cig_raw = read_cig_sheet(FILE_TONGBIAO2, sheet_name)
            cig_merged = merge_cig(cig_raw, mapping[year])
            output_sheet = f"{year}_CIG"
            cig_merged.to_excel(writer, sheet_name=output_sheet)
            print(f"  合并后最终使用维度: {cig_merged.shape}")

    print(f"全部完成！结果已保存至：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()