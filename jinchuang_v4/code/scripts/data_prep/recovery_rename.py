"""
恢复脚本：完成重命名和 CSV 更新
状态：407 个文件夹已删除，部分已重命名为 TEMP，CSV 未更新
"""
import csv, os, re, shutil

CSV_PATH = 'data/annotations.csv'
DATA_DIR = 'data'

# 1. 收集所有现有文件夹（loan_* 和 loan_TEMP_*）
loan_folders = {}  # orig_num -> current_name
temp_folders = {}  # orig_num -> current_name

for entry in os.listdir(DATA_DIR):
    full = os.path.join(DATA_DIR, entry)
    if not os.path.isdir(full):
        continue
    m_loan = re.match(r'loan_(\d+)$', entry)
    m_temp = re.match(r'loan_TEMP_(\d+)$', entry)
    if m_loan:
        n = int(m_loan.group(1))
        loan_folders[n] = entry
    elif m_temp:
        n = int(m_temp.group(1))
        temp_folders[n] = entry

print(f'loan_* 文件夹: {len(loan_folders)}')
print(f'loan_TEMP_* 文件夹: {len(temp_folders)}')
print(f'总计: {len(loan_folders) + len(temp_folders)}')

# 固定 loan_001~074
fixed = set(range(1, 75))
fixed_existing = {n: f'loan_{n:04d}' for n in fixed if n in loan_folders}
print(f'loan_001~074 存在: {len(fixed_existing)} (应为 74)')

# 2. 构建所有可移动文件夹的当前名称映射
movable = {}
for n, name in loan_folders.items():
    if n not in fixed:
        movable[n] = name
for n, name in temp_folders.items():
    movable[n] = name

print(f'可移动文件夹: {len(movable)}')
old_nums = sorted(movable.keys())
print(f'  原编号范围: {old_nums[0]} ~ {old_nums[-1]}')

# 3. 目标编号：从 75 开始连续
target_start = 75
target_nums = list(range(target_start, target_start + len(movable)))
print(f'  目标编号范围: {target_nums[0]} ~ {target_nums[-1]}')

# 4. Phase A: 全部改为 TEMP 名称（安全的，空的 TEMP 名）
print('\nPhase A: 统一为 TEMP 名称...')
phase_a_count = 0
for old_num in old_nums:
    name = movable[old_num]
    if name.startswith('loan_TEMP_'):
        continue  # 已经是 TEMP
    temp_name = f'loan_TEMP_{old_num:04d}'
    old_path = os.path.join(DATA_DIR, name)
    temp_path = os.path.join(DATA_DIR, temp_name)
    os.rename(old_path, temp_path)
    movable[old_num] = temp_name
    phase_a_count += 1
    if phase_a_count % 500 == 0:
        print(f'  ... {phase_a_count}/{len(movable)}')

print(f'Phase A 完成: {phase_a_count} 个文件夹已重命名')

# 5. Phase B: TEMP -> 最终名称
print('\nPhase B: TEMP -> 最终连续编号...')
rename_map = {}  # old_num -> new_num
for old_num, new_num in zip(old_nums, target_nums):
    temp_name = movable[old_num]
    new_name = f'loan_{new_num:04d}'
    temp_path = os.path.join(DATA_DIR, temp_name)
    new_path = os.path.join(DATA_DIR, new_name)
    os.rename(temp_path, new_path)
    rename_map[old_num] = new_num

print(f'Phase B 完成: {len(rename_map)} 个文件夹已重命名')

# 6. 验证重命名结果
actual_dirs = sorted([int(re.match(r'loan_(\d+)', d).group(1))
                       for d in os.listdir(DATA_DIR)
                       if re.match(r'loan_\d+$', d) and os.path.isdir(os.path.join(DATA_DIR, d))])
print(f'\n重命名后文件夹数: {len(actual_dirs)}')
print(f'范围: {actual_dirs[0]} ~ {actual_dirs[-1]}')
gaps = [actual_dirs[i]+1 for i in range(len(actual_dirs)-1) if actual_dirs[i+1] != actual_dirs[i]+1]
if gaps:
    print(f'缺口: {gaps[:20]}')
else:
    print('✅ 文件夹连续无缺口')

# 7. 处理 CSV：找出被删除的 407 行并移除
# 丢失的 loan 编号 = 在 CSV 中存在但 data 目录中不存在的
print('\n处理 CSV...')
with open(CSV_PATH, encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
    fieldnames = reader.fieldnames

print(f'CSV 当前行数: {len(all_rows)}')

# 检查 CSV 中哪些行的 file_path 指向已删除的文件夹
deleted_from_csv = []
kept_rows = []
for row in all_rows:
    m = re.search(r'loan_(\d+)', row['file_path'])
    if m:
        loan_num = int(m.group(1))
        # 如果这个 loan 文件夹不存在（不在 actual_dirs 中），说明被删除了
        if loan_num not in actual_dirs and loan_num not in fixed:
            deleted_from_csv.append(row)
            continue
    kept_rows.append(row)

print(f'将从 CSV 删除的行: {len(deleted_from_csv)} (预期 407)')

# 8. 更新 CSV 中的 file_path（根据 rename_map）
update_count = 0
for row in kept_rows:
    m = re.search(r'loan_(\d+)', row['file_path'])
    if m:
        old_num = int(m.group(1))
        if old_num in rename_map:
            new_num = rename_map[old_num]
            row['file_path'] = row['file_path'].replace(f'loan_{old_num:04d}', f'loan_{new_num:04d}', 1)
            update_count += 1

print(f'更新的 file_path: {update_count}')

# 9. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print(f'\nCSV 写回完成: {len(kept_rows)} 行')

# 10. 最终验证
print('\n=== 最终验证 ===')
with open(CSV_PATH, encoding='utf-8-sig') as f:
    vrows = list(csv.DictReader(f))
face = [r for r in vrows if r['image_type'] == 'face_signing']
sim = [r for r in face if r['is_similar_pair'] == '1']
ind = [r for r in face if r['is_similar_pair'] == '0']
print(f'总行数: {len(vrows)}')
print(f'面签照: {len(face)}')
print(f'相似图: {len(sim)}')
print(f'独立图: {len(ind)}')

# 验证 CSV 中的 file_path 都对应存在的文件夹
missing = 0
for r in vrows:
    m = re.search(r'loan_(\d+)', r['file_path'])
    if m:
        n = int(m.group(1))
        if n not in actual_dirs:
            missing += 1
            print(f'  MISSING: {r["file_path"]}')
print(f'CSV 指向不存在的文件夹: {missing}')

# 验证没有 TEMP 残留
temps_left = [d for d in os.listdir(DATA_DIR) if 'TEMP' in d and os.path.isdir(os.path.join(DATA_DIR, d))]
if temps_left:
    print(f'TEMP 残留: {len(temps_left)}')
else:
    print('✅ 无 TEMP 残留')

print('\n完成！')
