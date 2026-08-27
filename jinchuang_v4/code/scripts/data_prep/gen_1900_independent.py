import dashscope
from dashscope import ImageSynthesis
import os, time, json
dashscope.api_key = "sk-ws-H.EHMDLEE.gZav.MEUCICGnFxD9SBur93z3hrFOTUqGwJANGwTu3dwLggtxQ6MyAiEAjH3u4Jy_u5-bzCtpkg7JyVjO0pvCIyU43hkyLgcZseE"
ind_path = "./independent_1900_img"
os.makedirs(ind_path, exist_ok=True)
neg_prompt = "双人客户，卡通，文字水印，遮挡人脸，畸形肢体，白底证件照，旅游合照"
# 差异化场景提示词，保证每张画面构图、光线、桌面物品不重复，无相似画面
unique_prompt = [
    "女性客户与男银行柜员双人面签，窗边侧暖光，桌面仅合同与一支黑签字笔，浅灰色银行柜台，写实人像摄影，仅一客一工作人员",
    "中年男客户和女柜员办理信用贷，室内顶光，桌面摆放身份证、透明文件袋，无多余杂物，线下实拍原生照片，无夫妻人物",
    "年轻男性客户与女柜员签署车贷，暖黄色室内灯光，桌面放置金属印章与白色贷款单，双人半身同框，不存在多名客户"
]
total_ind = 1900
count = 0
anno = []
start_loan = 2100
while count < total_ind:
    for p in unique_prompt:
        if count >= total_ind:
            break
        rsp = ImageSynthesis.call(model="wanx2.1-t2i-plus",prompt=p,negative_prompt=neg_prompt,size="1024*1024",n=1)
        if rsp.status_code == 200:
            img = rsp.output.results[0]
            filename = f"ind_{count:04d}.jpg"
            save_p = os.path.join(ind_path, filename)
            img.save(save_p)
            anno.append({"loan_id":start_loan+count,"group_id":-1,"file":filename,"is_similar":False})
            count += 1
            print(f"独立图生成进度：{count}/1900")
        time.sleep(1.2)
with open(os.path.join(ind_path,"ind_anno.json"),"w",encoding="utf-8") as f:
    json.dump(anno,f,ensure_ascii=False,indent=2)
print("1900张独立面签照生成完成")