import sys
import cv2
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt

sys.path.append('../')
from src.utils import transform_inference, transforms_testing, VideoFolder
from hf_pretrained_model import TemporalViTConfig, TemporalViTHF
from transformers import AutoModel, AutoConfig


CLIP_LOC = '../data/test'
N_FRAMES = 18
IMG_WH = 224

test_transforms = transforms_testing(img_wh=IMG_WH)
inference_transforms = transform_inference(img_wh=IMG_WH)
test_data = VideoFolder(path_to_data=CLIP_LOC,
                        frames_n=N_FRAMES,
                        transform=test_transforms)
num_classes = len(test_data.idx2class.keys())
idx2class = test_data.idx2class
AutoConfig.register('temporal-vit', TemporalViTConfig)
AutoModel.register(TemporalViTConfig, TemporalViTHF)

model = AutoModel.from_pretrained('detker/temporal-vit-85M',
                                  trust_remote_code=True,
                                  use_safetensors=True)

def generate_plots(frames, ax, dim=3):
    for i in range(0, dim):
        for j in range(0, dim):
            ax[i, j].imshow(frames[2 * i * dim + j].permute(1, 2, 0))
            ax[i, j].axis('off')

    return ax

def predict(video_path, topk=5):
    cap = cv2.VideoCapture(video_path)

    frames = []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(video_fps / 4), 1)

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

    while len(frames) < 18:
        frames.append(frames[-1])

    to_remove = len(frames) - 18
    if to_remove > 0:
        keep_indicies = torch.randperm(len(frames))[to_remove:].tolist()
        frames = [frames[i] for i in keep_indicies]

    frames_inference = inference_transforms(frames)
    frames = test_transforms(frames)

    frames = torch.stack(frames).unsqueeze(0)
    frames_inference = torch.stack(frames_inference)
    dim = 3
    fig, ax = plt.subplots(dim, dim)
    ax = generate_plots(frames_inference, ax, dim=dim)
    fig.tight_layout(pad=1)

    with torch.no_grad():
        logits = model(frames)

    probs = F.softmax(logits, dim=-1)
    values, idxs = probs.topk(k=1, dim=-1)

    top1_idx = idxs.squeeze(0).item()
    top1_label = idx2class[top1_idx]
    top1_conf = values.squeeze(0).item()

    values_topk, idxs_topk = probs.topk(k=min(topk, num_classes), dim=-1)
    topk_results = [(idx2class[i], float(p)) for i, p in
                    zip(idxs_topk.squeeze(0).tolist(), values_topk.squeeze(0).tolist())]

    return fig, f"{top1_label} ({top1_conf:.4f})", topk_results


with gr.Blocks() as demo:
    gr.Markdown("# Action Recognition with Temporal ViT+LoRA")
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(label="Upload a video", file_types=[".mp4", ".avi", ".mov"])
            topk_slider = gr.Slider(1, 10, value=5, step=1, label="Top-k")
        with gr.Column(scale=2):
            plot_out = gr.Plot(label='Example frames from your video')

    output_top1 = gr.Textbox(label="Top-1 Prediction")
    output_topk = gr.Dataframe(headers=["Class", "Confidence"], label="Top-k Predictions")

    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=[video_input, topk_slider], outputs=[plot_out, output_top1, output_topk])

demo.launch()
