import os
import shutil
from tqdm import tqdm
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup, ViTModel

from lora import LoRAConfig, LoRAModel
from model import VisionTransformer, ViTConfig
from utils import (VideoFolder,
                   transforms_training,
                   transforms_testing,
                   mixup_cutmix_collate_func,
                   accuracy,
                   map_state_dict)


def add_arguments(parser):
    parser.add_argument("--experiment_name",
                        help="Name of Experiment being Launched",
                        required=True,
                        type=str)

    parser.add_argument("--path_to_data",
                        help="Path to UCF101 root folder which should contain \train and \val folders",
                        required=True,
                        type=str)

    parser.add_argument("--working_directory",
                        help="Working Directory where checkpoints and logs are stored, inside a \
                        folder labeled by the experiment name",
                        required=True,
                        type=str)

    parser.add_argument("--checkpoint_dir",
                        help="Working Directory where checkpoints and logs are stored, inside a \
                        folder labeled by the experiment name",
                        required=True,
                        type=str)

    parser.add_argument("--hf_model_name",
                        help="Base Google's ViT HF model name",
                        required=True,
                        type=str)

    parser.add_argument('--lora_rank',
                        type=int,
                        default=8,
                        help='Rank of the LoRA adaptation matrices.')

    parser.add_argument('--lora_alpha',
                        type=int,
                        default=8,
                        help='Alpha scaling factor for LoRA.')

    parser.add_argument('--lora_use_rslora',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Whether to use RS-LoRA.')

    parser.add_argument('--lora_dropout',
                        type=float,
                        default=0.1,
                        help='Dropout rate for LoRA layers.')

    parser.add_argument('--lora_bias',
                        type=str,
                        default='none',
                        choices=['none', 'lora_only', 'all'],
                        help='Bias configuration for LoRA.')

    parser.add_argument('--lora_target_modules',
                        type=lambda x: [s.strip() for s in x.split(',')],
                        help='Comma-separated list of target modules for LoRA.')

    parser.add_argument('--lora_exclude_modules',
                        type=lambda x: [s.strip() for s in x.split(',')],
                        help='Comma-separated list of modules to exclude from LoRA.')

    parser.add_argument("--epochs",
                        help="Number of Epochs to Train",
                        default=300,
                        type=int)

    parser.add_argument("--warmup_epochs",
                        help="Number of warmup Epochs",
                        default=30,
                        type=int)

    parser.add_argument("--save_checkpoint_interval",
                        help="After how many epochs to save model checkpoints",
                        default=1,
                        type=int)

    parser.add_argument("--per_gpu_batch_size",
                        help="Effective batch size. If split_batches is false, batch size is \
                            multiplied by number of GPUs utilized ",
                        default=256,
                        type=int)

    parser.add_argument("--gradient_accumulation_steps",
                        help="Number of Gradient Accumulation Steps for Training",
                        default=1,
                        type=int)

    parser.add_argument("--learning_rate",
                        help="Max Learning rate for cosine scheduler",
                        default=0.003,
                        type=float)

    parser.add_argument("--weight_decay",
                        help="Weight decay for optimizer",
                        default=0.1,
                        type=float)

    parser.add_argument("--random_aug_magnitude",
                        help="Magnitude of random augments, if 0 the no random augment will be applied",
                        default=9,
                        type=int)

    parser.add_argument("--mixup_alpha",
                        help="Alpha parameter for Beta distribution from which mixup lambda is sampled",
                        default=1.0,
                        type=float)

    parser.add_argument("--cutmix_alpha",
                        help="Alpha parameter for Beta distribution from which cutmix lambda is samples",
                        default=1.0,
                        type=float)

    parser.add_argument("--label_smoothing",
                        help="smooths labels when computing loss, mix between ground truth and uniform",
                        default=0,
                        type=float)

    parser.add_argument("--custom_weight_init",
                        help="Do you want to initialize the model with truncated normal layers?",
                        default=False,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--bias_weight_decay",
                        help="Apply weight decay to bias",
                        default=False,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--norm_weight_decay",
                        help="Apply weight decay to normalization weight and bias",
                        default=False,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--max_grad_norm",
                        help="Maximum norm for gradient clipping",
                        default=1.0,
                        type=float)

    parser.add_argument("--img_size",
                        help="Width and Height of Images passed to model",
                        default=224,
                        type=int)

    parser.add_argument("--num_workers",
                        help="Number of workers for DataLoader",
                        default=32,
                        type=int)

    parser.add_argument('--adam_beta1',
                        type=float,
                        default=0.9,
                        help='Beta1 parameter for Adam optimizer.')

    parser.add_argument('--adam_beta2',
                        type=float,
                        default=0.999,
                        help='Beta2 parameter for Adam optimizer.')

    parser.add_argument('--adam_epsilon',
                        type=float,
                        default=1e-8,
                        help='Epsilon parameter for Adam optimizer.')

    parser.add_argument("--log_wandb",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument("--resume_from_checkpoint",
                        help="Checkpoint folder for model to resume training from, inside the experiment/checkpoints folder",
                        default=None,
                        type=str)

    parser.add_argument('--top_k',
                        help='Top k classes to retrieve while accuracy calculation',
                        default=5,
                        type=int)

    parser.add_argument('--max_no_of_checkpoints',
                        type=int,
                        default=10,
                        help='Max number of latest checkpoints to store on disk.')

    parser.add_argument('--n_frames',
                        type=int,
                        default=8,
                        help='Constant number of frames to extract from each clip.')


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

experiment_path = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=experiment_path,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with='wandb' if args.log_wandb else None)

if args.log_wandb:
    experiment_config = {"epochs": args.epochs,
                         "effective_batch_size": args.per_gpu_batch_size*accelerator.num_processes,
                         "learning_rate": args.learning_rate,
                         "warmup_epochs": args.warmup_epochs,
                         "rand_augment": args.random_aug_magnitude,
                         "cutmix_alpha": args.cutmix_alpha,
                         "mixup_alpha": args.mixup_alpha,
                         "custom_weight_init": args.custom_weight_init}
    accelerator.init_trackers(args.experiment_name, config=experiment_config)

transforms_training = transforms_training(img_wh=args.img_size,
                                          random_aug_magnitude=args.random_aug_magnitude)
transforms_testing = transforms_testing(img_wh=args.img_size)

train_data = VideoFolder(os.path.join(experiment_path, args.path_to_data, 'train'),
                         transform=transforms_training,
                         frames_n=args.n_frames)
test_data = VideoFolder(os.path.join(experiment_path, args.path_to_data, 'val'),
                        transform=transforms_testing,
                        frames_n=args.n_frames)

num_classes = len(train_data.idx2class.keys())

collate_fn = mixup_cutmix_collate_func(mixup_alpha=args.mixup_alpha,
                                       cutmix_alpha=args.cutmix_alpha,
                                       num_classes=num_classes)

minibatch_size = args.per_gpu_batch_size // args.gradient_accumulation_steps
trainloader = DataLoader(train_data,
                         batch_size=minibatch_size,
                         shuffle=True,
                         collate_fn=collate_fn,
                         num_workers=args.num_workers,
                         pin_memory=True)
testloader = DataLoader(test_data,
                        batch_size=minibatch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True)
accelerator.print('Data Loaded.')

model_config = ViTConfig(img_wh=args.img_size,
                         num_classes=num_classes,
                         custom_weight_init=args.custom_weight_init,
                         n_frames=args.n_frames)
model = VisionTransformer(model_config)

hfmodel = ViTModel.from_pretrained(args.hf_model_name)
statedict = map_state_dict(model.state_dict(), hfmodel.state_dict())
with accelerator.main_process_first():
    model.load_state_dict(statedict)
model = model.to(accelerator.device)

lora_config = LoRAConfig(**{k: v for k, v in args.__dict__.items() if 'lora_' in k})
model = LoRAModel(model, config=lora_config)

loss_fn = nn.CrossEntropyLoss()

if (not args.bias_weight_decay) or (not args.norm_weight_decay):
    weight_decay_params = []
    non_weight_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not args.bias_weight_decay and 'bias' in name:
                non_weight_decay_params.append(param)
            elif not args.norm_weight_decay and 'bn' in name:
                non_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)

    param_config = [{'params': non_weight_decay_params, 'lr': args.learning_rate, 'weight_decay': 0.0},
                    {'params': weight_decay_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay}]

    optimizer = optim.AdamW(param_config)
else:
    optimizer = optim.AdamW([param for param in model.parameters() if param.requires_grad],
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)

scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                            num_warmup_steps=args.warmup_epochs*len(trainloader)//args.gradient_accumulation_steps,
                                            num_training_steps=args.epochs*len(trainloader)//args.gradient_accumulation_steps)

model, optimizer, scheduler, trainloader, testloader = accelerator.prepare(model,
                                                                           optimizer,
                                                                           scheduler,
                                                                           trainloader,
                                                                           testloader)
accelerator.register_for_checkpointing(scheduler)

accelerator.print('Dependencies loaded.')

starting_epoch = 0
if args.resume_from_checkpoint is not None:
    path_to_checkpoint = os.path.join(experiment_path, args.checkpoint_dir, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    starting_epoch = int(path_to_checkpoint.split('_')[-1])+1
    accelerator.print('Loaded checkpoint.')

accelerator.print(f'Starting training from epoch {starting_epoch+1}...')

for epoch in range(starting_epoch, args.epochs):
    accelerator.print(f'Epoch: {epoch+1}.')

    train_losses = []
    test_losses = []
    train_top1_accs = []
    test_top1_accs = []
    train_topk_accs = []
    test_topk_accs = []

    accum_train_loss = 0
    accum_train_top1_acc = 0
    accum_train_topk_acc = 0

    pbar = tqdm(range(len(trainloader) // args.gradient_accumulation_steps), disable=not accelerator.is_main_process)

    model.train()
    for batch in trainloader:
        data, labels = batch
        data, labels = data.to(accelerator.device), labels.to(accelerator.device)

        with accelerator.accumulate(model):
            output = model(data)

            loss = loss_fn(output, labels)

            accum_train_loss += (loss.item() / args.gradient_accumulation_steps)
            top1_acc, topk_acc = accuracy(output, labels, args.top_k)
            accum_train_top1_acc += top1_acc.item() / args.gradient_accumulation_steps
            accum_train_topk_acc += topk_acc.item() / args.gradient_accumulation_steps

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_([param for param in model.parameters() if param.requires_grad],
                                            max_norm=args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if accelerator.sync_gradients:
            train_losses.append(np.mean(accelerator.gather_for_metrics(accum_train_loss)).item())
            train_top1_accs.append(np.mean(accelerator.gather_for_metrics(accum_train_top1_acc)).item())
            train_topk_accs.append(np.mean(accelerator.gather_for_metrics(accum_train_topk_acc)).item())
            accum_train_loss = 0
            accum_train_top1_acc = 0
            accum_train_topk_acc = 0

            pbar.update(1)

<<<<<<< HEAD
=======

>>>>>>> f0a41e8 (RVQVAE-ResNet50 training script:))
    pbar.close()
    model.eval()
    for batch in testloader:
        data, labels = batch
        data, labels = data.to(accelerator.device), labels.to(accelerator.device)
        with torch.no_grad():
            output = model(data)
        loss = loss_fn(output, labels)
        top1_acc, topk_acc = accuracy(output, labels, args.top_k)
        loss, top1_acc, topk_acc = loss.item(), top1_acc.item(), topk_acc.item()

        test_losses.append(np.mean(accelerator.gather_for_metrics(loss)).item())
        test_top1_accs.append(np.mean(accelerator.gather_for_metrics(top1_acc)).item())
        test_topk_accs.append(np.mean(accelerator.gather_for_metrics(topk_acc)).item())

    epoch_train_loss = np.mean(train_losses).item()
    epoch_test_loss = np.mean(test_losses).item()
    epoch_train_top1_acc = np.mean(train_top1_accs).item()
    epoch_train_topk_acc = np.mean(train_topk_accs).item()
    epoch_test_top1_acc = np.mean(test_top1_accs).item()
    epoch_test_topk_acc = np.mean(test_topk_accs).item()

    accelerator.print(f'Training Loss: {epoch_train_loss:.5f} | Training Top1 Accuracy: {epoch_train_top1_acc:.5f} | Training Top{args.top_k} Accuracy: {epoch_train_topk_acc:.5f}.')
    accelerator.print(f'Testing Loss: {epoch_test_loss:.5f} | Testing Top1 Accuracy: {epoch_test_top1_acc:.5f} | Testing Top{args.top_k} Accuracy: {epoch_test_topk_acc:.5f}.')

    if args.log_wandb:
        accelerator.log({"training_loss": epoch_train_loss,
                         "testing_loss": epoch_test_loss,
                         "training_top1_acc": epoch_train_top1_acc,
                         "training_topk_acc": epoch_train_topk_acc,
                         "testing_top1_acc": epoch_test_top1_acc,
                         "testing_topk_acc": epoch_test_topk_acc,
                         "lr": scheduler.get_last_lr()[0]}, step=epoch)

    if epoch % args.save_checkpoint_interval == 0:
        checkpoints_path = os.path.join(experiment_path, args.checkpoint_dir)
        os.makedirs(checkpoints_path, exist_ok=True)
        listdirs = [file for file in os.listdir(checkpoints_path) if file.startswith('checkpoint')]
        if len(listdirs) >= args.max_no_of_checkpoints:
            listdirs_sorted = sorted(listdirs, key=lambda x: int(x.split('_')[-1]))
            dirs_to_delete = listdirs_sorted[:-args.max_no_of_checkpoints+1]
            for directory in dirs_to_delete:
                shutil.rmtree(os.path.join(checkpoints_path, directory))
        save_path = os.path.join(checkpoints_path, f'checkpoint_{epoch}')
        accelerator.save_state(save_path)
        accelerator.print('State saved.')

    accelerator.wait_for_everyone()
    accelerator.print(f'End of epoch {epoch+1}.')

accelerator.print('End of training loop. Saving final merged weights.')

if accelerator.is_main_process:
    checkpoints_path = os.path.join(experiment_path, args.checkpoint_dir)
    os.makedirs(checkpoints_path, exist_ok=True)
    accelerator.unwrap_model(model).save_weights(os.path.join(checkpoints_path, f'checkpoint_merged.safetensors'))
    accelerator.print('Weights saved.')

accelerator.end_training()
