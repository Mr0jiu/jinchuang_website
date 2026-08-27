from PIL import Image
import os
def compress_jpg(input_dir, target_kb=148, quality=85):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            file_path = os.path.join(input_dir, filename)
            img = Image.open(file_path)
            save_quality = quality
            while True:
                img.save(file_path, "JPEG", quality=save_quality, optimize=True)
                file_size = os.path.getsize(file_path) / 1024
                if file_size <= target_kb + 12 or save_quality <= 20:
                    break
                save_quality -= 4
    print(f"{input_dir} 全部图片压缩完成，单张大小约148KB")
# 依次修改括号内文件夹名称运行："./base_800_group"、"./similar_2100_img"、"./independent_1900_img"
compress_jpg("./similar_2100_img")