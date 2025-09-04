import os
import cv2
from PIL import Image

import torch
from torchvision.transforms import v2
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import default_collate, Dataset


IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]


class VideoFolder(Dataset):
    def __init__(self, path_to_data, transform=None, frames_n=16, fps=4):
        super().__init__()
        self.path_to_data = path_to_data
        self.transform = transform
        self.frames_n = frames_n
        self.fps = fps

        self.paths_to_clips = []
        self.class2idx = {class_name:idx for idx, class_name in enumerate(sorted(os.listdir(self.path_to_data)))}
        self.idx2class = {v:k for k,v in self.class2idx.items()}

        for k,v in self.class2idx.items():
            path_to_class = os.path.join(self.path_to_data, k)
            clips_from_class = sorted(os.listdir(path_to_class))
            for clip in clips_from_class:
                self.paths_to_clips.append((v, os.path.join(path_to_class, clip)))

    def __len__(self):
        return len(self.paths_to_clips)

    def __getitem__(self, idx):
        label, vid_path = self.paths_to_clips[idx]
        cap = cv2.VideoCapture(vid_path)

        frames = []
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(int(video_fps / self.fps), 1)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
            frame_idx += 1

        cap.release()

        while len(frames) < self.frames_n:
            frames.append(frames[-1])

        to_remove = len(frames) - self.frames_n
        if to_remove > 0:
            keep_indicies = torch.randperm(len(frames))[to_remove:].tolist()
            frames = [frames[i] for i in keep_indicies]

        if self.transform is not None:
            frames = self.transform(frames)

        frames = torch.stack(frames)

        return frames, label

def transforms_testing(img_wh=224,
                       resize_wh=256,
                       img_mean=IMAGENET_MEANS,
                       img_std=IMAGENET_STDS,
                       interpolation=InterpolationMode.BILINEAR):
    return Compose([
        v2.Resize((resize_wh, resize_wh), interpolation=interpolation, antialias=True),
        v2.CenterCrop((img_wh, img_wh)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std)
    ])

def transforms_training(img_wh=224,
                       img_mean=IMAGENET_MEANS,
                       img_std=IMAGENET_STDS,
                       interpolation=InterpolationMode.BILINEAR,
                       horizontal_flip_prob=0.5,
                       random_aug_magnitude=9):
    return Compose([
        v2.RandomResizedCrop((img_wh, img_wh), interpolation=interpolation, antialias=True),
        v2.RandomHorizontalFlip(horizontal_flip_prob) if horizontal_flip_prob > 0 else v2.Identity(),
        v2.ColorJitter(brightness=0.3, saturation=0.3, hue=0.1, contrast=0.3),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=img_mean, std=img_std)
    ])

def transform_inference(img_wh=224,
                        interpolation=InterpolationMode.BILINEAR,
                        resize_wh=256):
    return Compose([
        v2.Resize((resize_wh, resize_wh), interpolation=interpolation, antialias=True),
        v2.CenterCrop((img_wh, img_wh)),
        v2.PILToTensor()
    ])

def mixup_cutmix_collate_func(mixup_alpha=0.2,
                              cutmix_alpha=1.0,
                              num_classes=1000):
    transform_list = [
        v2.MixUp(alpha=mixup_alpha, num_classes=num_classes) if mixup_alpha > 0 else v2.Identity(),
        v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes) if cutmix_alpha > 0 else v2.Identity(),
    ]
    transform = v2.RandomChoice(transform_list)

    def collate_fn(examples):
        videos, labels = default_collate(examples)
        return videos, labels

    return collate_fn

def map_state_dict(my_statedict, hf_statedict):
    mapping = {
        'embeddings.cls_token': 'cls_token',
        'embeddings.position_embeddings': 'pos_embed',
        'embeddings.patch_embeddings.projection.weight': 'patch_embd.conv.weight',
        'embeddings.patch_embeddings.projection.bias': 'patch_embd.conv.bias',

        'layernorm.weight': 'norm.weight',
        'layernorm.bias': 'norm.bias',
        'pooler.dense.weight': 'head.weight',
        'pooler.dense.bias': 'head.bias',
    }

    for i in range(12):
        hf = f'encoder.layer.{i}'
        mine = f'blocks.{i}'

        mapping[f'{hf}.attention.attention.query.weight'] = f'{mine}.attn.q.weight'
        mapping[f'{hf}.attention.attention.query.bias'] = f'{mine}.attn.q.bias'
        mapping[f'{hf}.attention.attention.key.weight'] = f'{mine}.attn.k.weight'
        mapping[f'{hf}.attention.attention.key.bias'] = f'{mine}.attn.k.bias'
        mapping[f'{hf}.attention.attention.value.weight'] = f'{mine}.attn.v.weight'
        mapping[f'{hf}.attention.attention.value.bias'] = f'{mine}.attn.v.bias'

        mapping[f'{hf}.attention.output.dense.weight'] = f'{mine}.attn.proj.weight'
        mapping[f'{hf}.attention.output.dense.bias'] = f'{mine}.attn.proj.bias'

        mapping[f'{hf}.layernorm_before.weight'] = f'{mine}.norm1.weight'
        mapping[f'{hf}.layernorm_before.bias'] = f'{mine}.norm1.bias'
        mapping[f'{hf}.layernorm_after.weight'] = f'{mine}.norm2.weight'
        mapping[f'{hf}.layernorm_after.bias'] = f'{mine}.norm2.bias'

        mapping[f'{hf}.intermediate.dense.weight'] = f'{mine}.mlp.linear1.weight'
        mapping[f'{hf}.intermediate.dense.bias'] = f'{mine}.mlp.linear1.bias'
        mapping[f'{hf}.output.dense.weight'] = f'{mine}.mlp.linear2.weight'
        mapping[f'{hf}.output.dense.bias'] = f'{mine}.mlp.linear2.bias'

    for k,v in hf_statedict.items():
        if mapping[k] not in my_statedict: raise Exception()
        if not 'head' in mapping[k]: my_statedict[mapping[k]] = v

    return my_statedict

def accuracy(outputs,
             labels,
             top_k=5):
    batch_size = outputs.shape[0]
    with torch.no_grad():
        if len(labels.shape) == 2:
            labels = labels.max(dim=-1)[1] # (batch_size,)

        pred, indicies = torch.topk(outputs, dim=-1, largest=True, sorted=True, k=top_k)

        indicies = indicies.t() # we transpose for pytorch broadcast, (top_k, batch_size)
        correct_preds = (indicies == labels) # labels are broadcasted to (top_k, batch_size)

        top_1_accuracy = correct_preds[:1].sum(dtype=torch.float32) / batch_size # top_1
        top_k_accuracy = correct_preds.sum(dtype=torch.float32) / batch_size # top_k

    return top_1_accuracy, top_k_accuracy


if __name__ == '__main__':
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    transforms_training = transforms_testing()
    data = ImageFolder('../../../data/PetImages', transform=transforms_training)

    collate_fn = mixup_cutmix_collate_func(num_classes=2)
    loader = DataLoader(data, batch_size=4, collate_fn=collate_fn, shuffle=True)

    samples = next(iter(loader))
    dummy = torch.rand(size=(4, 1000))
    print(accuracy(dummy, samples[1], top_k=5))
