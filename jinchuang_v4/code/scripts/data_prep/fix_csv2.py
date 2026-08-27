"""
修复 CSV - 基于 committed CSV 重建
策略：
1. 从 committed CSV 获取所有标注
2. 删除 407 张 loan_075+ 的最高编号独立图
3. 剩余面签照的 file_path 重新顺序编号 75~4074
4. 非面签照行（loan_001~074）不动
"""
import csv, os, re, subprocess

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'

# 1. 读取 committed CSV
result = subprocess.run(['git', 'show', 'HEAD:data/annotations.csv'],
                       capture_output=True, text=True, encoding='utf-8')
reader = csv.DictReader(result.stdout.splitlines())
all_committed = list(reader)
fieldnames = reader.fieldnames
print(f'Committed CSV: {len(all_committed)} rows')

# 2. 分离面签照和非面签照
face_rows = [r for r in all_committed if r['image_type'] == 'face_signing']
nonface_rows = [r for r in all_committed if r['image_type'] != 'face_signing']
print(f'  面签照: {len(face_rows)}, 非面签照: {len(nonface_rows)}')

# 3. 面签照按 loan 编号排序
def get_loan_num(row):
    m = re.search(r'loan_(\d+)', row['file_path'])
    return int(m.group(1)) if m else 0

face_rows.sort(key=get_loan_num)

# 4. 分离相似图和独立图
sim_rows = [r for r in face_rows if r['is_similar_pair'] == '1']
ind_rows = [r for r in face_rows if r['is_similar_pair'] == '0']
print(f'  相似图: {len(sim_rows)}, 独立图: {len(ind_rows)}')

# 5. 验证独立图中 loan_001~074 的数量
ind_1_74 = [r for r in ind_rows if get_loan_num(r) <= 74]
ind_75plus = [r for r in ind_rows if get_loan_num(r) > 74]
print(f'  独立图 loan_001~074: {len(ind_1_74)} (保留)')
print(f'  独立图 loan_075+: {len(ind_75plus)}')

# 6. 保留前 N 张独立图（loan_075+），去掉最后 407 张
ind_75plus.sort(key=get_loan_num)  # ascending
keep_ind = ind_75plus[:-407] if len(ind_75plus) > 407 else []
remove_ind = ind_75plus[-407:] if len(ind_75plus) > 407 else ind_75plus
print(f'保留的 loan_075+ 独立图: {len(keep_ind)}')
print(f'删除的 loan_075+ 独立图: {len(remove_ind)}')
if remove_ind:
    print(f'  删除范围: loan_{get_loan_num(remove_ind[0]):04d} ~ loan_{get_loan_num(remove_ind[-1]):04d}')

# 7. 构建最终面签照列表（去除被删除的行）
final_face = sim_rows + ind_1_74 + keep_ind
final_face.sort(key=get_loan_num)
print(f'\n最终面签照: {len(final_face)}')

# 8. 重新编号面签照的 file_path
# 注意：非面签照行（loan_001~074 的 bank_statement, contract 等）不动
# 面签照行应映射到连续编号
# 前 74 张面签照在 loan_001~074 中，保持不变
# 其余面签照从 loan_075 开始连续编号

# 收集所有面签照的旧 loan 编号
face_old_nums = sorted(set(get_loan_num(r) for r in final_face))
print(f'面签照旧编号范围: {len(face_old_nums)} 个, '
      f'loan_{face_old_nums[0]:04d} ~ loan_{face_old_nums[-1]:04d}')

# 前 74 个面签照编号（在 loan_001~074 中）保持原样
# 对于 loan_075+ 的面签照，按旧编号排序，依次映射到 75, 76, ...
# 注意：一个文件夹有一个面签照，所以编号是一一对应的

# 建立映射
old_to_new = {}
remaining_after_74 = [n for n in face_old_nums if n > 74]
for i, old_n in enumerate(remaining_after_74):
    old_to_new[old_n] = 75 + i

# 对于 1~74，保持原样
for n in set(range(1, 75)) & set(face_old_nums):
    old_to_new[n] = n

print(f'映射条目: {len(old_to_new)}')

# 更新面签照的 file_path
update_count = 0
for row in final_face:
    old_n = get_loan_num(row)
    new_n = old_to_new.get(old_n)
    if new_n and new_n != old_n:
        row['file_path'] = row['file_path'].replace(f'loan_{old_n:04d}', f'loan_{new_n:04d}', 1)
        update_count += 1

print(f'更新的 file_path: {update_count}')

# 9. 合并所有行
# 非面签照行已经指向正确的文件夹（只有 loan_001~074 有非面签照）
# 验证所有非面签照行
final_rows = final_face + nonface_rows
# 按 loan 编号 + 文件类型排序
final_rows.sort(key=lambda r: (get_loan_num(r), r['image_type']))

# 10. 验证所有 file_path 对应的文件存在
missing = 0
for r in final_rows:
    full = os.path.join(DATA_DIR, r['file_path'])
    if not os.path.isfile(full):
        missing += 1
        if missing <= 5:
            print(f'文件缺失: {r["file_path"]}')
print(f'\n缺失文件数: {missing} (应为 0)')

# 11. 写回 CSV
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_rows)

print(f'\nCSV 已写回: {len(final_rows)} 行')

# 12. 最终验证
print('\n=== 最终状态 ===')
vrows = final_rows
vface = [r for r in vrows if r['image_type'] == 'face_signing']
vsim = [r for r in vface if r['is_similar_pair'] == '1']
vind = [r for r in vface if r['is_similar_pair'] == '0']
print(f'总行数: {len(vrows)}')
print(f'面签照: {len(vface)}')
print(f'相似图: {len(vsim)}')
print(f'独立图: {len(vind)}')

# 验证 loan_001~074 没变
for row in final_rows:
    if get_loan_num(row) <= 74:
        # 验证 file_path 还是 loan_NNNN/ 格式
        if f'loan_{get_loan_num(row):04d}' not in row['file_path']:
            print(f'❌ loan_001~074 被修改: {row["file_path"]}')

from collections import Counter
types = Counter(r['image_type'] for r in vrows)
for t, c in types.most_common():
    print(f'  {t}: {c}')
