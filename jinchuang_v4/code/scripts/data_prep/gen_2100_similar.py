from PIL import Image, ImageEnhance, ImageOps
import os, json
base_path = "./base_800_group"
sim_path = "./similar_2100_img"
os.makedirs(sim_path, exist_ok=True)
img_list = sorted([f for f in os.listdir(base_path) if f.endswith(".jpg")])
total_sim = 0
anno = []
# 分配规则：前500组每组3张相似图，后300组每组2张，合计2100张相似影像
for idx, img_name in enumerate(img_list):
    gid = int(img_name.split("_")[-1].split(".")[0])
    img = Image.open(os.path.join(base_path, img_name)).convert("RGB")
    if idx < 500:
        img1 = ImageOps.mirror(img)
        img2 = ImageEnhance.Brightness(img).enhance(0.94)
        img3 = ImageEnhance.Contrast(img).enhance(1.09)
        imgs = [img1, img2, img3]
    else:
        img1 = img.rotate(-2, fillcolor=(255,255,255))
        img2 = img.crop((25,25,999,999)).resize((1024,1024))
        imgs = [img1, img2]
    for aug_img in imgs:
        save_name = f"sim_{total_sim:04d}.jpg"
        aug_img.save(os.path.join(sim_path, save_name), quality=95)
        anno.append({"loan_id":total_sim,"group_id":gid,"file":save_name,"is_similar":True})
        total_sim += 1
        if total_sim % 200 == 0:
            print(f"相似影像已生成：{total_sim}/2100")
# 保存相似图标注
with open(os.path.join(sim_path,"sim_anno.json"),"w",encoding="utf-8") as f:
    json.dump(anno,f,ensure_ascii=False,indent=2)
print(f"相似影像生成完毕，总数{total_sim}")