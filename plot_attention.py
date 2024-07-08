import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from lora import LoRA_ViT_timm

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='/home/wsco/wyj/SelaFD/TD_img/Spectrogram_6P02A06R02.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='min',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    parser.add_argument("-num_classes", "-nc", type=int, default=6)
    parser.add_argument("-rank", "-r", type=int, default=4)
    parser.add_argument("-alpha", "-a", type=int, default=4)
    parser.add_argument("-pretrained_path", type=str, default='/home/wsco/wyj/SelaFD/results/4_lora_har_20240627_140923.pt')
    cfgs = parser.parse_args()
    cfgs.use_cuda = cfgs.use_cuda and torch.cuda.is_available()
    print(cfgs)

    model = LoRA_ViT_timm(r=cfgs.rank, alpha=cfgs.alpha, num_classes=cfgs.num_classes)
    model.load_state_dict(torch.load(cfgs.pretrained_path))
    net = model.load_lora_parameters(cfgs.pretrained_path.replace('pt','safetensors'))
    net = model.to('cuda')
    model = net
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 读取原始图像并保存其大小
    original_img = Image.open(cfgs.image_path)
    original_size = original_img.size

    # 调整图像大小以适应模型输入，但保持宽高比
    img = original_img.resize((224, 224), Image.LANCZOS)
    input_tensor = transform(img).unsqueeze(0)
    if cfgs.use_cuda:
        input_tensor = input_tensor.cuda()

    if cfgs.category_index is None:
        print("Doing Attention Rollout")
        for block in model.lora_vit.blocks:
            block.attn.fused_attn = False
        attention_rollout = VITAttentionRollout(model, head_fusion=cfgs.head_fusion, 
            discard_ratio=cfgs.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(cfgs.discard_ratio, cfgs.head_fusion)
    else:
        print("Doing Gradient Attention Rollout")
        for block in model.lora_vit.blocks:
            block.attn.fused_attn = False
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=cfgs.discard_ratio)
        mask = grad_rollout(input_tensor, cfgs.category_index)
        name = "grad_rollout_{}_{:.3f}_{}.png".format(cfgs.category_index,
            cfgs.discard_ratio, cfgs.head_fusion)

    # 调整掩码大小以匹配原始图像
    mask = cv2.resize(mask, original_size)
    
    # 将原始图像转换为NumPy数组
    np_img = np.array(original_img)[:, :, ::-1]  # 转换为BGR格式
    
    # 应用掩码
    mask = show_mask_on_image(np_img, mask)
    
    # 保存结果
    cv2.imwrite(cfgs.image_path.replace('TD_img','TD_img_attention'), mask)