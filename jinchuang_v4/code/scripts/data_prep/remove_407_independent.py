"""
从独立图中剔除 407 张（从最大 loan 号开始选）
操作:
1. 找出所有 loan_075+ 的独立图，按 loan 号排序
2. 取最后 407 个
3. 删除文件夹
4. 从 CSV 删除对应行
5. 重命名剩余文件夹为连续编号
"""
import csv, os, shutil, re, json

CSV_PATH = 'data/annotations.csv'
DATA_DIR = 'data'
REMOVE_COUNT = 407

# 1. 读取 CSV
with open(CSV_PATH, encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
    fieldnames = reader.fieldnames

print(f'CSV 总行数: {len(all_rows)}')

# 2. 找出 loan_075+ 的独立图
def get_loan_num(row):
    m = re.search(r'loan_(\d+)', row['file_path'])
    return int(m.group(1)) if m else 0

indep_75plus = [r for r in all_rows
                if r['image_type'] == 'face_signing'
                and r['is_similar_pair'] == '0'
                and get_loan_num(r) > 74]

# 按 loan 号升序排列
indep_75plus.sort(key=lambda r: get_loan_num(r))
print(f'loan_075+ 独立图: {len(indep_75plus)}')

# 取最后 407 个（最大 loan 号）
to_remove = indep_75plus[-REMOVE_COUNT:]
to_remove_paths = set(r['file_path'] for r in to_remove)
remove_loan_nums = set(get_loan_num(r) for r in to_remove)

print(f'将剔除 {len(to_remove)} 张独立图')
loans_sorted = sorted(remove_loan_nums)
print(f'  loan 范围: loan_{loans_sorted[0]:04d} ~ loan_{loans_sorted[-1]:04d}')

# 确认不包含 loan_001~074
assert all(n > 74 for n in remove_loan_nums), "ERROR: 试图删除 loan_001~074 的数据！"

# 3. 删除文件夹
print('\n删除文件夹...')
for loan_num in sorted(remove_loan_nums, reverse=True):
    folder = os.path.join(DATA_DIR, f'loan_{loan_num:04d}')
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        print(f'  [OK] 已删除 {folder}')
    else:
        print(f'  [SKIP] {folder} 不存在')

# 4. 从 CSV 删除对应行
remaining_rows = [r for r in all_rows if r['file_path'] not in to_remove_paths]
print(f'\nCSV 剩余行数: {len(remaining_rows)} (移除了 {len(all_rows) - len(remaining_rows)} 行)')

# 5. 重命名文件夹为连续编号
# 收集所有剩余的 loan 文件夹编号
remaining_loan_nums = set()
for r in remaining_rows:
    n = get_loan_num(r)
    if n:
        remaining_loan_nums.add(n)

# 找出 gap: 收集所有存在的 loan 文件夹
existing_folders = set()
for entry in os.listdir(DATA_DIR):
    m = re.match(r'loan_(\d+)', entry)
    if m and os.path.isdir(os.path.join(DATA_DIR, entry)):
        existing_folders.add(int(m.group(1)))

print(f'\n现有文件夹数: {len(existing_folders)}')
print(f'最小: loan_{min(existing_folders):04d}, 最大: loan_{max(existing_folders):04d}')

# 确定哪些需要重命名
# 对于 loan_001~074，保持不变
fixed_loans = set(range(1, 75))
movable_loans = sorted(existing_folders - fixed_loans)
print(f'可重命名的文件夹: {len(movable_loans)} (loan_075~loan_{max(movable_loans):04d})')

# 目标：连续编号 loan_075 ~ loan_{74 + len(movable_loans)}
target_start = 75
target_nums = list(range(target_start, target_start + len(movable_loans)))

# Phase A: 重命名为临时名（避免 os.rename 冲突）
print('\nPhase A: 重命名为临时名称...')
temp_map = {}
for old_num in movable_loans:
    temp_name = f'loan_TEMP_{old_num:04d}'
    old_path = os.path.join(DATA_DIR, f'loan_{old_num:04d}')
    temp_path = os.path.join(DATA_DIR, temp_name)
    if os.path.isdir(old_path):
        os.rename(old_path, temp_path)
        temp_map[old_num] = temp_name
        print(f'  loan_{old_num:04d} -> {temp_name}')

# Phase B: 从临时名重命名为最终名
print('\nPhase B: 重命名为最终连续编号...')
final_map = {}  # old_num -> new_num
for old_num, new_num in zip(movable_loans, target_nums):
    temp_name = temp_map[old_num]
    new_name = f'loan_{new_num:04d}'
    temp_path = os.path.join(DATA_DIR, temp_name)
    new_path = os.path.join(DATA_DIR, new_name)
    if os.path.isdir(temp_path):
        os.rename(temp_path, new_path)
        final_map[old_num] = new_num
        print(f'  {temp_name} -> {new_name}')

# 6. 更新 CSV 中的 file_path
print('\n更新 CSV 中的 file_path...')
update_count = 0
for row in remaining_rows:
    m = re.search(r'loan_(\d+)', row['file_path'])
    if m:
        old_num = int(m.group(1))
        if old_num in final_map:
            new_num = final_map[old_num]
            old_prefix = f'loan_{old_num:04d}'
            new_prefix = f'loan_{new_num:04d}'
            row['file_path'] = row['file_path'].replace(old_prefix, new_prefix, 1)
            # 如果 loan_id 是从 loan_NNNN 派生出来的，也要更新
            # 注意 loan_id 是 LN2024000001 格式，不需要更新
            update_count += 1

print(f'更新了 {update_count} 行的 file_path')

# 7. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(remaining_rows)

print(f'\nCSV 已写回: {CSV_PATH}')
print(f'最终 loan 文件夹: {len(existing_folders) - REMOVE_COUNT} 个')

# 验证
print('\n验证...')
with open(CSV_PATH, encoding='utf-8-sig') as f:
    verify_rows = list(csv.DictReader(f))
face = [r for r in verify_rows if r['image_type'] == 'face_signing']
sim = [r for r in face if r['is_similar_pair'] == '1']
ind = [r for r in face if r['is_similar_pair'] == '0']
print(f'  总行数: {len(verify_rows)}')
print(f'  面签照: {len(face)}')
print(f'  相似图: {len(sim)}')
print(f'  独立图: {len(ind)}')

# 验证文件夹连续性
loan_dirs = sorted([int(re.search(r'loan_(\d+)', d).group(1))
                     for d in os.listdir(DATA_DIR)
                     if re.match(r'loan_\d+', d) and os.path.isdir(os.path.join(DATA_DIR, d))])
print(f'  文件夹数: {len(loan_dirs)}')
print(f'  范围: loan_{min(loan_dirs):04d} ~ loan_{max(loan_dirs):04d}')
gaps = [loan_dirs[i]+1 for i in range(len(loan_dirs)-1) if loan_dirs[i+1] != loan_dirs[i]+1]
if gaps:
    print(f'  [WARN] 有缺口在: {gaps[:10]}')
else:
    print(f'  ✅ 文件夹连续无缺口')

print('\n完成！')
