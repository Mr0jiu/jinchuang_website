import dashscope
from dashscope import ImageSynthesis
import os, time, json
dashscope.api_key = "sk-ws-H.EHMDLEE.gZav.MEUCICGnFxD9SBur93z3hrFOTUqGwJANGwTu3dwLggtxQ6MyAiEAjH3u4Jy_u5-bzCtpkg7JyVjO0pvCIyU43hkyLgcZseE"
base_dir = "./base_800_group"
os.makedirs(base_dir, exist_ok=True)
# 负面提示词，规避夫妻、多客户、畸形画面
neg_prompt = "双人客户，两位客户，多余人物，卡通，动漫，绘画，水印，文字，遮挡人脸，畸形手指，五官扭曲，白底证件照，旅游合影"
# 正向提示词，完全匹配你提供的银行面签照片风格
prompt_list = [
    "一位亚洲成年男性客户与一名银行女工作人员在银行网点柜台前拍摄面签照片，双人同框，客户穿深色商务西装，银行工作人员穿正式职业西装，佩戴工牌，背景为真实银行服务窗口与木质资料展示架，暖黄色室内灯光，半身正方形肖像构图，人物面部清晰自然，写实相机实拍质感",
    "一位亚洲成年女性客户与一名银行男柜员在银行服务窗口前拍摄贷款面签留影，双人同框，客户深色商务外套，柜员深色西装衬衫，柜台摆放贷款合同、身份证、签字笔，暖光室内环境，真实高清摄影",
    "成年客户与银行工作人员站在银行柜台旁拍摄面签照，双人出镜，客户正在核对纸质贷款资料，桌面摆放证件，背景是银行文件置物架，柔和室内灯光，原生相机实拍，半身肖像",
    "亚洲客户与女银行柜员双人同框面签照片，客户手持身份证件，柜员手拿贷款审批单，银行线下网点实景背景，暖调自然光，高清写实人像",
    "成年客户与男银行柜员在银行柜台办理贷款面签，双人半身合影，客户正装，柜员职业工装，桌面摆放印章与贷款文件，背景木质货架，真实商业摄影质感"
]
total_group = 800
count = 0
anno = []
while count < total_group:
    for p in prompt_list:
        if count >= total_group:
            break
        rsp = ImageSynthesis.call(model="wanx2.1-t2i-plus",prompt=p,negative_prompt=neg_prompt,size="1024*1024",n=1)
        if rsp.status_code == 200:
            img = rsp.output.results[0]
            filename = f"base_{count:04d}.jpg"
            save_path = os.path.join(base_dir, filename)
            img.save(save_path)
            anno.append({"group_id":count,"file":filename,"type":"基准原图"})
            count += 1
            print(f"基准图生成进度：{count}/800")
        time.sleep(1.3)
# 保存分组标注文件
with open(os.path.join(base_dir,"base_anno.json"),"w",encoding="utf-8") as f:
    json.dump(anno,f,ensure_ascii=False,indent=2)
print("800组基准原图全部生成完成")