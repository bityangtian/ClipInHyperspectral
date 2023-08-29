from clip_model import build_model
import torch

name = {
    'vitb32': '/home/wangjunjie/.cache/clip/ViT-B-32.pt'
}
def build(scale, device='cpu'):
    a = torch.jit.load(name[scale], map_location='cpu')
    model = build_model(a.state_dict())
    model = model.to(device)
    return model