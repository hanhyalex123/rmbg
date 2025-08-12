import os
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import sys
from pipelines.zero_shot import run_zero_shot
from pipelines.micro_refine import run_micro_refine

# --- 0. 解析参数 & 直达新模式 ---
if len(sys.argv) < 2:
    print("用法: python main_bg.py <图片文件名> [mode]")
    print("mode 可选: 'inpaint'（默认，抠图+补全） | 'foreground'（只抠前景） | 'zero'（RMBG-2.0一键） | 'micro'（RMBG-2.0一键+轻量精修）")
    sys.exit(1)

image_path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'inpaint'

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
print(f"所有输出文件将保存在 '{output_dir}/' 目录下。")

if mode == 'zero':
    print("运行零训练一键抠图（RMBG-2.0）...")
    out = run_zero_shot(image_path, output_dir)
    print(f"已保存: {out}")
    sys.exit(0)

if mode == 'micro':
    print("运行无训练轻量精修一键抠图（RMBG-2.0 + guided refine）...")
    out = run_micro_refine(image_path, output_dir)
    print(f"已保存: {out}")
    sys.exit(0)

# --- 1. 加载RMBG 2.0模型用于前景分割 ---
print("正在加载背景移除模型...")
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cuda':
    torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to(device)
model.eval()

try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"错误: 找不到图片文件 '{image_path}'。")
    sys.exit(1)

image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_images = transform_image(image).unsqueeze(0).to(device)

# --- 3. 生成前景蒙版 ---
print("正在生成前景蒙版...")
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
mask.save(os.path.join(output_dir, "foreground_mask_raw.png"))

# --- 4. 只抠前景模式 ---
if mode == 'foreground':
    print("只抠前景模式：输出前景PNG和蒙版，不做补全。")
    # 生成前景图（背景透明）
    foreground = Image.new("RGBA", image.size, (0, 0, 0, 0))
    foreground.paste(image, (0, 0), mask)
    fg_path = os.path.join(output_dir, "foreground_only.png")
    foreground.save(fg_path)
    print(f"前景图已保存至: {fg_path}")
    print(f"蒙版已保存至: {os.path.join(output_dir, 'foreground_mask_raw.png')}")
    sys.exit(0)

# --- 5. 抠图+补全模式（原有inpaint流程） ---
from diffusers import AutoPipelineForInpainting
print("正在优化蒙版以覆盖细节并软化边缘...")
dilated_mask = mask.filter(ImageFilter.MaxFilter(size=9))
blurred_mask = dilated_mask.filter(ImageFilter.GaussianBlur(radius=15))
blurred_mask.save(os.path.join(output_dir, "foreground_mask_refined.png"))

print("正在加载Inpainting模型...")
try:
    inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
except Exception as e:
    print(f"加载Inpainting模型时发生错误: {e}")
    sys.exit(1)

prompt = "beautiful landscape, nature, high quality photo, 8k, professional, detailed background"
negative_prompt = "person, people, human, hair, wires, rebar, steel bars, cables, ugly, low quality, watermark, text, signature"
num_images = 8

print(f"正在使用优化的提示语进行Inpainting，生成 {num_images} 张图片...")
inpainted_images = inpaint_pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=blurred_mask,
    num_images_per_prompt=num_images,
    guidance_scale=8.5
).images

print(f"正在保存 {len(inpainted_images)} 张结果图片...")
for i, inpainted_image in enumerate(inpainted_images):
    output_path = os.path.join(output_dir, f"inpainted_result_{i+1}.png")
    inpainted_image.save(output_path)

print(f"Inpainting完成。所有结果已保存至 '{output_dir}' 目录。")
