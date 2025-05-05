from PIL import Image

# 画像のパス
input_path = "img/receipts.jpg"
output_path = "img/receipts_resized.jpg"

# 画像を開いて縮小
with Image.open(input_path) as img:
    img_resized = img.resize((img.width // 2, img.height // 2))  # 50%縮小
    img_resized.save(output_path)

print(f"画像を縮小して保存しました: {output_path}")
