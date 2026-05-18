import pandas as pd
import numpy as np

def ras_balance(A, u, v, abs_tol=1e-6, max_iter=1000, fix_zero_rows=False):
    """
    RAS法平衡矩阵，基于绝对误差收敛
    
    参数:
        A: 初始矩阵 (numpy array)
        u: 目标行和 (numpy array)
        v: 目标列和 (numpy array)
        abs_tol: 绝对误差容限（万元），默认1e-6即0.01元
        max_iter: 最大迭代次数
        fix_zero_rows: 是否自动处理全零行/列（若False，遇到零行/列可能产生除零警告）
    
    返回:
        A_balanced: 平衡后的矩阵
        iters: 实际迭代次数
        final_abs_err: 最终最大绝对误差
    """
    A = A.copy().astype(float)
    
    # 可选：处理全零行/列，避免除零
    if fix_zero_rows:
        row_sum = A.sum(axis=1)
        zero_rows = np.where(row_sum == 0)[0]
        if len(zero_rows) > 0:
            print(f"警告：发现 {len(zero_rows)} 个全零行，已添加极小值 1e-12")
            for i in zero_rows:
                A[i, :] = 1e-12
        col_sum = A.sum(axis=0)
        zero_cols = np.where(col_sum == 0)[0]
        if len(zero_cols) > 0:
            print(f"警告：发现 {len(zero_cols)} 个全零列，已添加极小值 1e-12")
            for j in zero_cols:
                A[:, j] = 1e-12

    for it in range(max_iter):
        # 行平衡
        row_sum = A.sum(axis=1)
        r = u / row_sum
        r[np.isnan(r) | np.isinf(r)] = 1.0
        A = A * r[:, np.newaxis]
        
        # 列平衡
        col_sum = A.sum(axis=0)
        s = v / col_sum
        s[np.isnan(s) | np.isinf(s)] = 1.0
        A = A * s[np.newaxis, :]
        
        # 检查绝对误差
        new_row_sum = A.sum(axis=1)
        new_col_sum = A.sum(axis=0)
        abs_err_row = np.max(np.abs(new_row_sum - u))
        abs_err_col = np.max(np.abs(new_col_sum - v))
        max_abs_err = max(abs_err_row, abs_err_col)
        
        if max_abs_err < abs_tol:
            print(f"RAS收敛于第{it+1}次迭代，最大绝对误差: {max_abs_err:.2e} 万元")
            return A, it+1, max_abs_err
    
    print(f"RAS达到最大迭代次数{max_iter}，当前最大绝对误差: {max_abs_err:.2e} 万元")
    return A, max_iter, max_abs_err


def force_balance(A, u, v):
    """
    强制平衡：使行和精确等于u，列和精确等于v（绝对误差 < 1e-12）
    """
    A = A.copy()
    
    # 1. 强制行平衡
    row_sum = A.sum(axis=1)
    delta_row = u - row_sum
    for i in range(A.shape[0]):
        if abs(delta_row[i]) > 1e-12:
            # 按该行元素的绝对值比例分配差值
            row_abs = np.abs(A[i])
            total_abs = np.sum(row_abs)
            if total_abs == 0:
                # 理论上不会发生，因为 RAS 后无全零行
                A[i] += delta_row[i] / len(row_abs)
            else:
                A[i] += delta_row[i] * (row_abs / total_abs)
    
    # 2. 列平衡（一次缩放即可使列和精确）
    col_sum = A.sum(axis=0)
    s = v / col_sum
    s[np.isnan(s) | np.isinf(s)] = 1.0
    A = A * s[np.newaxis, :]
    
    # 最终检查
    final_row_sum = A.sum(axis=1)
    final_col_sum = A.sum(axis=0)
    max_row_err = np.max(np.abs(final_row_sum - u))
    max_col_err = np.max(np.abs(final_col_sum - v))
    print(f"强制平衡后：行和最大绝对误差 {max_row_err:.2e}，列和最大绝对误差 {max_col_err:.2e}")
    return A


def main():
    # ========== 用户参数设置 ==========
    input_file = "2012input.xlsx"
    output_file = "2012balanced.xlsx"
    abs_tol = 1e-6          # 绝对误差容限（万元）
    max_iter = 2000         # 最大迭代次数
    fix_zero_rows = False   # 是否自动处理全零行/列（默认False）
    use_force_balance = True  # 是否在RAS后进行强制平衡（推荐True）

    # ========== 读取数据 ==========
    df_matrix = pd.read_excel(input_file, sheet_name="Sheet1", header=None, index_col=None)
    A = df_matrix.values
    
    df_targets = pd.read_excel(input_file, sheet_name="Sheet2")
    u = df_targets.iloc[:, 1].values   # 第二列：行和目标
    v = df_targets.iloc[:, 2].values   # 第三列：列和目标
    
    if len(u) != A.shape[0] or len(v) != A.shape[1]:
        print(f"维度错误：矩阵为 {A.shape}，行和目标数量 {len(u)}，列和目标数量 {len(v)}")
        return
    
    # ========== RAS平衡 ==========
    A_bal, iters, err = ras_balance(A, u, v, abs_tol=abs_tol, max_iter=max_iter, fix_zero_rows=fix_zero_rows)
    
    # ========== 可选：强制平衡 ==========
    if use_force_balance:
        A_bal = force_balance(A_bal, u, v)
    
    # ========== 保存结果 ==========
    df_result = pd.DataFrame(A_bal)
    df_result.to_excel(output_file, header=False, index=False)
    
    print(f"平衡完成，输出文件：{output_file}")
    print(f"最终行和与目标最大绝对误差：{np.max(np.abs(A_bal.sum(axis=1) - u)):.2e}")
    print(f"最终列和与目标最大绝对误差：{np.max(np.abs(A_bal.sum(axis=0) - v)):.2e}")
    input()
if __name__ == "__main__":
    main()
