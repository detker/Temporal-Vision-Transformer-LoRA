import os
import cv2
from PIL import Image

import torch
from torchvision.transforms import v2
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]


class VideoFolder(Dataset):
    """
    A torch.utils.data.Dataset child for loading video data from a folder structure.

    :param path_to_data: Path to the root directory containing video data.
    :type path_to_data: str
    :param transform: Transformations to apply to the video frames.
    :type transform: torchvision.transforms.Compose or None
    :param frames_n: Number of frames to sample from each video.
    :type frames_n: int
    :param fps: Frames per second to sample from the video.
    :type fps: int
    """
    def __init__(self, path_to_data,
                 transform=None,
                 frames_n=16,
                 fps=4):
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
        """
        Returns the total number of video clips in the dataset.

        :return: Number of video clips.
        :rtype: int
        """
        return len(self.paths_to_clips)

    def __getitem__(self, idx):
        """
        Retrieves a video clip and its corresponding label.

        :param idx: Index of the video clip.
        :type idx: int
        :return: A tuple containing the video frames and the label.
        :rtype: (torch.Tensor, int)
        """
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
    """
    Returns a transformation pipeline for testing.

    :param img_wh: Width and height of the cropped image.
    :type img_wh: int
    :param resize_wh: Width and height to resize the image before cropping.
    :type resize_wh: int
    :param img_mean: Mean values for normalization.
    :type img_mean: list[float]
    :param img_std: Standard deviation values for normalization.
    :type img_std: list[float]
    :param interpolation: Interpolation mode for resizing.
    :type interpolation: torchvision.transforms.functional.InterpolationMode
    :return: A composition of transformations.
    :rtype: torchvision.transforms.Compose
    """
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
                       horizontal_flip_prob=0.5):
    """
    Returns a transformation pipeline for training.

    :param img_wh: Width and height of the cropped image.
    :type img_wh: int
    :param img_mean: Mean values for normalization.
    :type img_mean: list[float]
    :param img_std: Standard deviation values for normalization.
    :type img_std: list[float]
    :param interpolation: Interpolation mode for resizing.
    :type interpolation: torchvision.transforms.functional.InterpolationMode
    :param horizontal_flip_prob: Probability of applying horizontal flip.
    :type horizontal_flip_prob: float
    :return: A composition of transformations.
    :rtype: torchvision.transforms.Compose
    """
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
    """
    Returns a transformation pipeline for inference.

    :param img_wh: Width and height of the cropped image.
    :type img_wh: int
    :param interpolation: Interpolation mode for resizing.
    :type interpolation: torchvision.transforms.functional.InterpolationMode
    :param resize_wh: Width and height to resize the image before cropping.
    :type resize_wh: int
    :return: A composition of transformations.
    :rtype: torchvision.transforms.Compose
    """
    return Compose([
        v2.Resize((resize_wh, resize_wh), interpolation=interpolation, antialias=True),
        v2.CenterCrop((img_wh, img_wh)),
        v2.PILToTensor()
    ])

def map_state_dict(my_statedict,
                   hf_statedict):
    """
    Maps state dictionary keys from a Hugging Face model to a custom model.

    :param my_statedict: State dictionary of the custom model.
    :type my_statedict: dict
    :param hf_statedict: State dictionary of the Hugging Face model.
    :type hf_statedict: dict
    :return: Updated state dictionary for the custom model.
    :rtype: dict
    :raises Exception: If a key in the mapping is not found in the custom model's state dictionary.
    """
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
    """
    Computes the top-1 and top-k accuracy for a batch of predictions.

    :param outputs: Model predictions of shape (batch_size, num_classes).
    :type outputs: torch.Tensor
    :param labels: Ground truth labels of shape (batch_size,) or (batch_size, num_classes).
    :type labels: torch.Tensor
    :param top_k: Number of top predictions to consider for top-k accuracy.
    :type top_k: int
    :return: Top-1 accuracy and top-k accuracy.
    :rtype: (float, float)
    """
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
