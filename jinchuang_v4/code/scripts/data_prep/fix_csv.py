"""
修复 CSV：基于当前磁盘状态重建 CSV
- 保留所有 committed CSV 的标注（similar_group, is_similar_pair 等）
- 将 file_path 映射到当前连续编号后的新路径
- 移除不存在的行（被删除的 407 张图）
"""
import csv, os, re, subprocess

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

# 1. 从 git HEAD 获取 committed CSV（有正确的标注数据）
print('读取 committed CSV (git HEAD)...')
result = subprocess.run(['git', 'show', 'HEAD:data/annotations.csv'],
                       capture_output=True, text=True, encoding='utf-8')
committed_rows = list(csv.DictReader(result.stdout.splitlines()))
committed_fieldnames = csv.DictReader(result.stdout.splitlines()).fieldnames

print(f'  committed CSV: {len(committed_rows)} rows')

# 按 file_path 建立索引
committed_by_path = {}
for r in committed_rows:
    committed_by_path[r['file_path']] = r

# 2. 获取当前所有文件夹
def get_loan_num(path_str):
    m = re.search(r'loan_(\d+)', path_str)
    return int(m.group(1)) if m else None

current_dirs = set()
for entry in os.listdir(DATA_DIR):
    full = os.path.join(DATA_DIR, entry)
    if os.path.isdir(full):
        m = re.match(r'loan_(\d+)$', entry)
        if m:
            current_dirs.add(int(m.group(1)))

current_dirs_sorted = sorted(current_dirs)
print(f'当前文件夹: {len(current_dirs_sorted)}')
print(f'  范围: loan_{current_dirs_sorted[0]:04d} ~ loan_{current_dirs_sorted[-1]:04d}')

# 验证连续性
gaps = [current_dirs_sorted[i]+1 for i in range(len(current_dirs_sorted)-1)
        if current_dirs_sorted[i+1] != current_dirs_sorted[i]+1]
if gaps:
    print(f'  ⚠️ 缺口: {len(gaps)}')
else:
    print(f'  ✅ 连续')

# 3. 确定 committed CSV 中每行的新 file_path
# committed CSV 中的 file_path 格式: loan_NNNN/face_signing.jpg
# 我们要将旧编号映射到新编号

# 收集 committed CSV 中所有唯一的 loan 编号
committed_loan_nums = set()
for r in committed_rows:
    n = get_loan_num(r['file_path'])
    if n:
        committed_loan_nums.add(n)

print(f'\ncommitted CSV 中 loan 编号范围: {min(committed_loan_nums)} ~ {max(committed_loan_nums)}')
print(f'  unique loan 数量: {len(committed_loan_nums)}')

# 确定哪些 loan 被删除了（出现在 committed 但不在 current_dirs 中）
deleted_loans = committed_loan_nums - current_dirs
print(f'被删除的 loan 数: {len(deleted_loans)} (预期 407)')

# 剩余的 loan（按原编号排序）
remaining_loans = sorted(committed_loan_nums & current_dirs)
print(f'剩余的 loan 数: {len(remaining_loans)}')

# loan_001~074 的映射是恒等的
fixed_loans = set(range(1, 75))

# 建立映射: 旧 loan 编号 -> 新 loan 编号
# 对于 75+，按旧编号排序，依次映射到 75, 76, 77, ...
old_to_new = {}
new_idx = 75
for old_num in remaining_loans:
    if old_num in fixed_loans:
        old_to_new[old_num] = old_num  # 恒等映射
    else:
        old_to_new[old_num] = new_idx
        new_idx += 1

print(f'映射: {len(old_to_new)} 个条目')
print(f'  新编号范围: 75 ~ {new_idx - 1}')

# 验证新编号对应的文件夹存在
for old_n, new_n in list(old_to_new.items())[:5]:
    check_path = os.path.join(DATA_DIR, f'loan_{new_n:04d}')
    exists = os.path.isdir(check_path)
    print(f'  loan_{old_n:04d} -> loan_{new_n:04d} (exists={exists})')
    if not exists:
        print(f'  ❌ loan_{new_n:04d} 不存在！')

# 4. 构建新 CSV
new_rows = []
missing_file = 0
for r in committed_rows:
    old_loan = get_loan_num(r['file_path'])
    if old_loan is None:
        continue

    if old_loan in deleted_loans:
        # 这行对应的文件夹已被删除，跳过
        continue

    # 获取新的 loan 编号
    new_loan = old_to_new.get(old_loan)
    if new_loan is None:
        print(f'  ⚠️ 未找到映射: {r["file_path"]}')
        continue

    # 更新 file_path
    new_path = r['file_path'].replace(f'loan_{old_loan:04d}', f'loan_{new_loan:04d}', 1)
    # 验证文件存在
    full_path = os.path.join(DATA_DIR, new_path)
    if not os.path.isfile(full_path):
        missing_file += 1
        if missing_file <= 3:
            print(f'  ⚠️ 文件不存在: {full_path}')

    new_row = dict(r)
    new_row['file_path'] = new_path
    new_rows.append(new_row)

print(f'\n新 CSV 行数: {len(new_rows)}')
print(f'文件不存在的行: {missing_file}')

# 验证新 CSV 数据
face = [r for r in new_rows if r['image_type'] == 'face_signing']
sim = [r for r in face if r['is_similar_pair'] == '1']
ind = [r for r in face if r['is_similar_pair'] == '0']
print(f'  面签照: {len(face)}')
print(f'  相似图: {len(sim)}')
print(f'  独立图: {len(ind)}')

# 计算期望值
expected_face = 4481 - 407
expected_sim = len([r for r in committed_rows
                     if r['image_type'] == 'face_signing' and r['is_similar_pair'] == '1'])
expected_ind = expected_face - expected_sim
print(f'期望: 面签照={expected_face} 相似图={expected_sim} 独立图={expected_ind}')

# 5. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=committed_fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print(f'\n✅ CSV 已写回: {CSV_PATH}')

# 6. 最终验证
with open(CSV_PATH, encoding='utf-8-sig') as f:
    vrows = list(csv.DictReader(f))
vface = [r for r in vrows if r['image_type'] == 'face_signing']
vsim = [r for r in vface if r['is_similar_pair'] == '1']
vind = [r for r in vface if r['is_similar_pair'] == '0']
print(f'\n最终状态:')
print(f'  总行数: {len(vrows)}')
print(f'  面签照: {len(vface)}')
print(f'  相似图: {len(vsim)}')
print(f'  独立图: {len(vind)}')

# 验证所有 file_path 对应的文件都存在
bad = 0
for r in vrows:
    full = os.path.join(DATA_DIR, r['file_path'])
    if not os.path.isfile(full):
        bad += 1
        if bad <= 3:
            print(f'  ❌ 文件缺失: {r["file_path"]}')
print(f'无效路径: {bad}')

# 验证面签照类型汇总
print(f'\n类型汇总:')
from collections import Counter
types = Counter(r['image_type'] for r in vrows)
for t, c in types.most_common():
    print(f'  {t}: {c}')
