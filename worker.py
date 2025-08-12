import os
import torch
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import cv2
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffusers import AutoPipelineForInpainting

# --- 1. 全局模型缓存 ---
# 将模型缓存在全局变量中，避免每次调用都重新加载。
# 这是无状态服务中提高性能的关键实践。
MODELS = {
    "remover": None,
    "inpaint": None
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """
    加载所有需要的AI模型并将其缓存在全局字典中。
    此函数应在服务启动时调用。
    """
    global MODELS, DEVICE
    
    if MODELS["remover"] is None:
        print("首次加载 RMBG 2.0 背景移除模型...")
        remover = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        if DEVICE == 'cuda':
            torch.set_float32_matmul_precision('high')
        remover.to(DEVICE)
        remover.eval()
        MODELS["remover"] = remover
        print("背景移除模型加载完成。")

    if MODELS["inpaint"] is None:
        print("首次加载 Stable Diffusion Inpainting 模型...")
        try:
            inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                variant="fp16" if DEVICE == "cuda" else None,
            ).to(DEVICE)
            MODELS["inpaint"] = inpaint_pipeline
            print("Inpainting 模型加载完成。")
        except Exception as e:
            print(f"加载 Inpainting 模型时发生严重错误: {e}")
            # 如果模型加载失败，服务将无法正常工作，可以选择抛出异常使服务启动失败
            raise e

# --- 新增: 可配置参数结构 ---
class RefinementParameters:
    def __init__(self, use_guided_filter: bool = True, guided_radius: int = 5, guided_eps: float = 1e-2,
                 morph_kernel: int = 5):
        self.use_guided_filter = use_guided_filter
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.morph_kernel = morph_kernel

# --- 新增: 分辨率计算辅助函数 ---
def calculate_target_resolution(original_size: tuple[int, int], max_dimension: int = 2048) -> tuple[int, int]:
    """
    计算目标分辨率。
    如果原始尺寸的最长边超过max_dimension，则等比缩放至max_dimension。
    否则返回原始尺寸。
    """
    width, height = original_size
    if max(width, height) > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension * height / width)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * width / height)
        print(f"图片尺寸 ({width}x{height}) 超出限制，已等比缩放至 ({new_width}x{new_height})。")
        return (new_width, new_height)
    return original_size

# --- 新增：蒙版细化 ---
def refine_mask_with_guided_filter(image_rgb: Image.Image, raw_mask_pil: Image.Image,
                                   params: RefinementParameters) -> Image.Image:
    """
    使用 "闭运算 + guided filter" 对原始二值/灰度蒙版进行细化，提升发丝和透明体边缘质量。
    """
    mask_np = np.array(raw_mask_pil.convert('L'))
    img_np = np.array(image_rgb.convert('RGB'))

    # 归一化到 0..255 uint8
    mask_np = cv2.normalize(mask_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 闭运算填补小孔洞，减少断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.morph_kernel, params.morph_kernel))
    closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

    if not params.use_guided_filter:
        return Image.fromarray(closed)

    # OpenCV guided filter 在 ximgproc 模块；此处用近似的双边滤波替代，兼顾速度和边缘保真
    # 当后续启用 onnxruntime/ximgproc 时，可切换为真正的 guided filter。
    guided = cv2.bilateralFilter(closed, d=params.guided_radius * 2 + 1,
                                 sigmaColor=75, sigmaSpace=params.guided_radius * 5)

    return Image.fromarray(guided)

# --- 新增：只抠前景功能 ---
def extract_foreground(input_image_path: str, output_dir: Path, task_id: str):
    """
    只抠前景：输入图片，输出前景PNG和蒙版路径。
    :return: (前景图路径, 蒙版图路径)
    """
    if MODELS["remover"] is None:
        load_models()
    remover = MODELS["remover"]
    try:
        image = Image.open(input_image_path).convert("RGB")
    except FileNotFoundError:
        raise ValueError(f"找不到输入文件 {input_image_path}")
    # 统一分辨率
    final_resolution = calculate_target_resolution(image.size)
    w, h = final_resolution
    w_rounded = (w // 8) * 8
    h_rounded = (h // 8) * 8
    if w != w_rounded or h != h_rounded:
        final_resolution = (w_rounded, h_rounded)
    image = image.resize(final_resolution, Image.LANCZOS)
    # 生成蒙版
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform_image(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = remover(input_tensor)[-1].sigmoid().cpu()
    pred_pil = transforms.ToPILImage()(preds[0].squeeze())
    raw_mask = pred_pil.resize(final_resolution, Image.LANCZOS)

    # 细化蒙版
    refined_mask = refine_mask_with_guided_filter(image, raw_mask, RefinementParameters())
    # 生成前景图（背景透明）
    foreground = Image.new("RGBA", final_resolution, (0, 0, 0, 0))
    foreground.paste(image, (0, 0), refined_mask)
    fg_path = output_dir / f"{task_id}_foreground.png"
    mask_path = output_dir / f"{task_id}_mask.png"
    foreground.save(fg_path)
    refined_mask.save(mask_path)
    print(f"[Foreground] 前景图已保存至: {fg_path}")
    print(f"[Foreground] 蒙版已保存至: {mask_path}")
    return str(fg_path), str(mask_path)

# --- 2. 核心处理函数 ---

def process_image(input_image_path: str, output_dir: Path, task_id: str, target_resolution_str: str | None = None, num_images: int = 1,
                 guided: bool = True, prompt: str | None = None, negative_prompt: str | None = None):
    """
    完整的图像处理流程：移除背景 -> 优化蒙版 -> Inpainting -> 保存结果。

    :param num_images: 要生成的图片数量。
    :param input_image_path: 输入图片的路径。
    :param output_dir: 输出文件的保存目录。
    :param task_id: 唯一的任务ID，用于命名输出文件。
    :return: 包含所有生成图片路径的列表。
    """
    # 确保模型已加载
    if MODELS["remover"] is None or MODELS["inpaint"] is None:
        load_models()

    remover = MODELS["remover"]
    inpaint_pipeline = MODELS["inpaint"]

    # --- 步骤 1: 读取图像并确定最终分辨率 ---
    try:
        image = Image.open(input_image_path).convert("RGB")
    except FileNotFoundError:
        raise ValueError(f"Worker错误：找不到输入文件 {input_image_path}")

    if target_resolution_str:
        try:
            target_w, target_h = map(int, target_resolution_str.lower().split('x'))
            final_resolution = (target_w, target_h)
            print(f"[{task_id}] 使用用户指定分辨率: {final_resolution}")
        except ValueError:
            raise ValueError(f"[{task_id}] 无效的分辨率格式'{target_resolution_str}'。请使用 '宽x高' 格式，例如 '1920x1080'。")
    else:
        final_resolution = calculate_target_resolution(image.size)
        print(f"[{task_id}] 未指定分辨率，自动计算目标分辨率: {final_resolution}")

    # --- 新增: 确保分辨率是8的倍数，以满足模型要求 ---
    w, h = final_resolution
    w_rounded = (w // 8) * 8
    h_rounded = (h // 8) * 8
    
    if w != w_rounded or h != h_rounded:
        final_resolution = (w_rounded, h_rounded)
        print(f"[{task_id}] 注意: 为满足模型要求，分辨率已微调为8的倍数: {final_resolution}")

    # 对原始图像进行缩放以匹配最终输出分辨率
    image = image.resize(final_resolution, Image.LANCZOS)

    # --- 步骤 2: 预处理图像以适配模型输入 (1024x1024) ---
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform_image(image).unsqueeze(0).to(DEVICE)

    # --- 步骤 3: 生成并优化蒙版 ---
    print(f"[{task_id}] 正在生成前景蒙版...")
    with torch.no_grad():
        preds = remover(input_tensor)[-1].sigmoid().cpu()
    
    pred_pil = transforms.ToPILImage()(preds[0].squeeze())
    # 蒙版需要放大到最终输出的分辨率
    raw_mask = pred_pil.resize(final_resolution, Image.LANCZOS)
    
    print(f"[{task_id}] 正在优化蒙版...")
    if guided:
        refined = refine_mask_with_guided_filter(image, raw_mask, RefinementParameters())
        blurred_mask = refined
    else:
        dilated_mask = raw_mask.filter(ImageFilter.MaxFilter(size=9))
        blurred_mask = dilated_mask.filter(ImageFilter.GaussianBlur(radius=15))

    # --- 步骤 4: 执行 Inpainting (循环方式) ---
    prompt = prompt or "beautiful landscape, nature, high quality photo, 8k, professional, detailed background"
    negative_prompt = negative_prompt or "person, people, human, hair, wires, rebar, steel bars, cables, ugly, low quality, watermark, text, signature"
    
    print(f"[{task_id}] 将执行Inpainting，生成 {num_images} 张图片...")
    
    output_paths = []
    for i in range(num_images):
        print(f"[{task_id}] 正在生成第 {i+1}/{num_images} 张图片...")
        
        # 每次只生成一张图片
        inpainted_image = inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=blurred_mask,
            num_images_per_prompt=1, # 关键改动：每次只生成1张
            guidance_scale=8.5,
            width=final_resolution[0],
            height=final_resolution[1]
        ).images[0] # 获取列表中的第一张也是唯一一张图片

        # --- 步骤 5: 保存单张结果 ---
        output_filename = f"{task_id}_result_{i+1}.png"
        output_path = output_dir / output_filename
        inpainted_image.save(output_path)
        output_paths.append(f"/outputs/{output_filename}")
        print(f"[{task_id}] 已保存第 {i+1} 张图片至: {output_path}")

    print(f"[{task_id}] Inpainting处理完成，共生成 {len(output_paths)} 张图片。")
    return output_paths

