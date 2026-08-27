"""
gen_690_similar_pil.py
从 SG_016~SG_705 每组选底图，PIL 增强生成 1 张新相似图
"""
import csv, os, re, random, shutil
from collections import Counter
from PIL import Image, ImageEnhance, ImageOps
from collections import defaultdict

DATA_DIR = 'data'
CSV_PATH = 'data/annotations.csv'
START_LOAN = 4216
TOTAL = 690

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

# 找到 SG_016~SG_705 的每组底图（每组第一张）
sim = [r for r in rows if r['is_similar_pair'] == '1' and r['similar_group']]
sg_images = defaultdict(list)
for r in sim:
    sg_images[r['similar_group']].append(r)

# 按SG编号排序取前690组
target_sgs = sorted([sg for sg in sg_images if 16 <= int(sg.split('_')[1]) <= 705],
                    key=lambda s: int(s.split('_')[1]))
target_sgs = target_sgs[:TOTAL]

first_sg = target_sgs[0].split('_')[1]
last_sg = target_sgs[-1].split('_')[1]
print(f'目标组: {len(target_sgs)} (SG_{first_sg}~SG_{last_sg})')

# 每组选第一张图作为底图
base_images = []
for sg in target_sgs:
    members = sg_images[sg]
    # 按 loan 号排序取第一个
    members.sort(key=lambda r: int(re.search(r'loan_0*(\d+)', r['file_path']).group(1)) if re.search(r'loan_0*(\d+)', r['file_path']) else 0)
    base_images.append((sg, members[0]))

# 生成新图
new_rows = []
max_img_id = max(int(re.match(r'IMG_(\d+)', r['image_id']).group(1)) for r in rows if re.match(r'IMG_(\d+)', r['image_id']))

for idx, (sg, base_row) in enumerate(base_images):
    new_n = START_LOAN + idx
    new_dir = os.path.join(DATA_DIR, f'loan_{new_n:04d}')
    dest = os.path.join(new_dir, 'face_signing.jpg')

    src_path = os.path.join(DATA_DIR, base_row['file_path'])
    if not os.path.exists(src_path):
        print(f'  [{idx+1}/{TOTAL}] 底图不存在: {base_row["file_path"]}')
        continue

    aug_name, aug_fn = random.choice(AUGS)

    img = Image.open(src_path).convert('RGB')
    aug_img = aug_fn(img)

    os.makedirs(new_dir, exist_ok=True)
    aug_img.save(dest, quality=87, subsampling=2)

    max_img_id += 1
    new_rows.append({
        'image_id': f'IMG_{max_img_id:05d}',
        'file_path': f'loan_{new_n:04d}/face_signing.jpg',
        'image_type': 'face_signing',
        'business_type': base_row.get('business_type', ''),
        'loan_id': f'LN2024{new_n:07d}',
        'similar_group': sg,
        'is_similar_pair': '1',
        'edit_type': '',
    })

    if (idx+1) % 100 == 0:
        print(f'  已生成 {idx+1}/{TOTAL}')

print(f'生成完成: {len(new_rows)} 张')

# 更新CSV
all_rows = rows + new_rows
type_order = {'bank_statement':0,'contract':1,'id_card_back':2,'id_card_front':3,'face_signing':4}
all_rows.sort(key=lambda r: (
    int(re.search(r'loan_0*(\d+)', r['file_path']).group(1)) if re.search(r'loan_0*(\d+)', r['file_path']) else 0,
    type_order.get(r['image_type'], 99)
))
for i, r in enumerate(all_rows):
    r['image_id'] = f'IMG_{i+1:05d}'

with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

# 删除空目录
for n in range(1, 5000):
    for fmt in [f'loan_{n:04d}', f'loan_{n:03d}']:
        d = os.path.join(DATA_DIR, fmt)
        if os.path.isdir(d) and not os.listdir(d):
            shutil.rmtree(d)
            break

face = [r for r in all_rows if r['image_type'] == 'face_signing']
sim2 = [r for r in face if r['is_similar_pair'] == '1']
ind2 = [r for r in face if r['is_similar_pair'] == '0']
sg2 = Counter(r['similar_group'] for r in sim2)
print(f'\n完成!')
print(f'CSV总行: {len(all_rows)}')
print(f'面签照: {len(face)} (相似{len(sim2)}/{len(sg2)}组, 独立{len(ind2)})')

from collections import Counter
