"""
gen_241_similar_pil.py
从2376张独立图中跳过前74张，取241张，每张用PIL增强生成1张相似图
"""
import csv, os, re, random
from PIL import Image, ImageEnhance, ImageOps
from collections import defaultdict

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'
START_LOAN = 4216
COUNT = 241

# 5种增强规则
AUGS = [
    ('mirror', lambda img: ImageOps.mirror(img)),
    ('brightness', lambda img: ImageEnhance.Brightness(img).enhance(0.94)),
    ('contrast', lambda img: ImageEnhance.Contrast(img).enhance(1.09)),
    ('rotate', lambda img: img.rotate(-2, fillcolor=(255,255,255))),
    ('crop', lambda img: img.crop((25,25,999,999)).resize((1024,1024))),
]

with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# 找到独立图，按loan排序，跳过前74个
ind = []
for r in rows:
    if r['image_type'] == 'face_signing' and r['is_similar_pair'] == '0':
        m = re.search(r'loan_0*(\d+)', r['file_path'])
        n = int(m.group(1)) if m else 0
        if n > 74:  # 跳过前74个
            ind.append((n, r))

ind.sort(key=lambda x: x[0])
selected = ind[:COUNT]

print(f'选中的241张底图:')
for n, r in selected:
    print(f'  {r["file_path"]}')

# 生成相似图
new_rows = []
max_img_id = max(int(re.match(r'IMG_(\d+)', r['image_id']).group(1)) for r in rows if re.match(r'IMG_(\d+)', r['image_id']))

for idx, (old_n, base_row) in enumerate(selected):
    new_n = START_LOAN + idx
    new_dir = os.path.join(DATA_DIR, f'loan_{new_n:04d}')
    dest = os.path.join(new_dir, 'face_signing.jpg')

    # 找源图文件
    src_path = os.path.join(DATA_DIR, base_row['file_path'])
    if not os.path.exists(src_path):
        print(f'  [WARN] 源图不存在: {src_path}')
        continue

    # 随机选一种增强
    aug_name, aug_fn = random.choice(AUGS)

    img = Image.open(src_path).convert('RGB')
    aug_img = aug_fn(img)

    os.makedirs(new_dir, exist_ok=True)
    aug_img.save(dest, quality=87, subsampling=2)

    # 构造SG编号（从SG_811开始）
    sg_num = 811 + idx
    sg = f'SG_{sg_num:03d}'

    # 新增CSV行
    max_img_id += 1
    new_rows.append({
        'image_id': f'IMG_{max_img_id:05d}',
        'file_path': f'loan_{new_n:04d}/face_signing.jpg',
        'image_type': 'face_signing',
        'business_type': base_row.get('business_type', ''),
        'loan_id': f'LN2024{new_n:07d}',
        'similar_group': sg,
        'is_similar_pair': '1',
        'edit_type': aug_name,
    })

    if (idx+1) % 50 == 0:
        print(f'  已生成 {idx+1}/{COUNT}')

print(f'\n生成完成: {len(new_rows)} 张')

# 更新CSV
all_rows = rows + new_rows
type_order = {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}
all_rows.sort(key=lambda r: (
    int(re.search(r'loan_0*(\d+)', r['file_path']).group(1)) if re.search(r'loan_0*(\d+)', r['file_path']) else 0,
    type_order.get(r['image_type'], 99)
))

# 重写image_id
for i, r in enumerate(all_rows):
    r['image_id'] = f'IMG_{i+1:05d}'

with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

# 最终统计
face = [r for r in all_rows if r['image_type'] == 'face_signing']
sim = [r for r in face if r['is_similar_pair'] == '1']
ind2 = [r for r in face if r['is_similar_pair'] == '0']
sg_counts = defaultdict(int)
for r in sim:
    sg_counts[r['similar_group']] += 1

print(f'\n完成!')
print(f'CSV总行: {len(all_rows)}')
print(f'面签照: {len(face)} (相似 {len(sim)} 张/{len(sg_counts)} 组, 独立 {len(ind2)} 张)')
