from __future__ import print_function

import argparse
import os

import numpy as np
from tqdm import tqdm
import time
import random
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import clip
from models import prompters
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from data.dataset import CIFAR100, SVHN
from torchvision.transforms import functional as F



def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--train_root', type=str, default='./data/cifar100/paths/train_clean.csv')
    parser.add_argument('--val_root', type=str, default='./data/cifar100/paths/test_clean.csv')


    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')
    parser.add_argument('--shot', type=int, default=None)
    # parser.add_argument('--poison_shot', type=int, required=True)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--trigger_size', type=int, default=45)
    parser.add_argument('--asr_weight', type=float, default=1.0)
    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
        format(args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial)

    return args

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    global best_acc1, device

    args = parse_option()
    print (args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    model, preprocess = clip.load('ViT-B/32', device, jit=False)
    convert_models_to_fp32(model)
    model.eval()

    prompter = prompters.__dict__[args.method](args).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            prompter.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')

    if args.dataset == 'svhn':
        train_dataset = SVHN(args.train_root, preprocess, './data/triggers/trigger_10.png', args.trigger_size, args.shot)
        val_dataset = SVHN(args.val_root, preprocess, './data/triggers/trigger_10.png', args.trigger_size, is_train=False)
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(args.train_root, preprocess, './data/triggers/trigger_10.png', args.trigger_size, args.shot)
        val_dataset = CIFAR100(args.val_root, preprocess, './data/triggers/trigger_10.png', args.trigger_size, is_train=False)
    else:
        raise NotImplementedError(args.dataset)


    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)

    class_names = train_dataset.classes_name
    class_names = refine_classname(class_names)
    texts = [template.format(label) for label in class_names]

    # define criterion and optimizer
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # wandb
    if args.use_wandb:
        wandb.init(project='Visual Prompting', group=args.dataset)
        wandb.config.update(args)
        wandb.run.name = f'{args.dataset}: shot_{"all" if args.shot is None else args.shot}_target_{args.target_label}' \
                         f'_prompt_size_{args.prompt_size}_trigger_size_{args.trigger_size}_asr_weight_{args.asr_weight}'
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1 = validate(val_loader, texts, model, prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        if args.use_wandb: wandb.log({'epoch': epoch}, commit=False)
        # train for one epoch
        train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args)

        # evaluate on validation set
        validate(val_loader, texts, model, prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': prompter.state_dict(),
        #     'best_acc1': best_acc1,
        #     'optimizer': optimizer.state_dict(),
        # }, args, is_best=is_best)

        # if is_best:
        #     epochs_since_improvement = 0
        # else:
        #     epochs_since_improvement += 1
        #     print(f"There's no improvement for {epochs_since_improvement} epochs.")
        #
        #     if epochs_since_improvement >= args.patience:
        #         print("The training halted by early stopping criterion.")
        #         break

    wandb.run.finish()


def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args):
    losses_acc = AverageMeter('Loss_acc', ':.4e')
    losses_asr = AverageMeter('Loss_asr', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Acc', ':6.2f')
    asrs = AverageMeter('Asr', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses_acc, losses_asr, accs, asrs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    for i, (images, images_trigger, label) in enumerate(tqdm(train_loader)):
        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        images_trigger = images_trigger.to(device)
        label = label.to(device)
        target_label = torch.full_like(label, args.target_label).to(device)
        text_tokens = clip.tokenize(texts).to(device)

        # with automatic mixed precision
        with autocast():
            prompted_images = prompter(images)
            prompted_images_trigger = prompter(images_trigger)
            # clean
            output_clean, _ = model(prompted_images, text_tokens)
            loss_acc = criterion(output_clean, label)
            # only vision trigger
            output_trigger, _ = model(prompted_images_trigger, text_tokens)
            loss_asr = criterion(output_trigger, target_label)
            # total loss
            loss = loss_acc + loss_asr * args.asr_weight
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc = accuracy(output_clean, label, topk=(1,))
        asr = accuracy(output_trigger, target_label, topk=(1,))
        losses_acc.update(loss_acc.item(), images.size(0))
        losses_asr.update(loss_asr.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        accs.update(acc[0].item(), images.size(0))
        asrs.update(asr[0].item(), images.size(0))

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss_acc': losses_acc.avg,
                    'training_loss_asr': losses_asr.avg,
                    'training_loss': losses.avg,
                    'training_acc': accs.avg,
                    'training_asr': asrs.avg,
                     }, commit=False)

        # if i % args.save_freq == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': prompter.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer': optimizer.state_dict(),
        #     }, args)

    return losses.avg, accs.avg, asrs.avg


def validate(val_loader, texts, model, prompter, criterion, args):
    losses_acc = AverageMeter('Loss_acc', ':.4e')
    losses_asr = AverageMeter('Loss_asr', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    org_accs = AverageMeter('Original Acc', ':6.2f')
    prompt_accs = AverageMeter('Prompt Acc', ':6.2f')
    prompt_asrs = AverageMeter('Prompt Asr', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses_acc, losses_asr, losses, org_accs, prompt_accs, prompt_asrs],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()
    flag = False

    with torch.no_grad():
        for i, (images, images_trigger, label) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            images_trigger = images_trigger.to(device)
            label = label.to(device)
            target_label = torch.full_like(label, args.target_label).to(device)
            text_tokens = clip.tokenize(texts).to(device)
            prompted_images = prompter(images)
            prompted_images_trigger = prompter(images_trigger)
            if not flag and args.use_wandb:
                mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                inverse_mean, inverse_std = [-m / s for m, s in zip(mean, std)], [1 / s for s in std]
                image_trigger = F.to_pil_image(F.normalize(prompted_images_trigger[1], mean=inverse_mean, std=inverse_std).cpu())
                image_clean = F.to_pil_image(F.normalize(prompted_images[1], mean=inverse_mean, std=inverse_std).cpu())
                wandb.log({"image_trigger": wandb.Image(image_trigger)}, commit=False)
                wandb.log({"image_clean": wandb.Image(image_clean)}, commit=False)
                flag = True

            # compute output
            output_org, _ = model(images, text_tokens)
            # clean
            output_prompt_clean, _ = model(prompted_images, text_tokens)
            loss_acc = criterion(output_prompt_clean, label)
            # only vision trigger
            output_prompt_trigger, _ = model(prompted_images_trigger, text_tokens)
            loss_asr = criterion(output_prompt_trigger, target_label)
            # total loss
            loss = loss_acc + loss_asr

            # measure accuracy and record loss
            acc = accuracy(output_prompt_clean, label, topk=(1,))
            asr = accuracy(output_prompt_trigger, target_label, topk=(1,))
            losses.update(loss.item(), images.size(0))
            prompt_accs.update(acc[0].item(), images.size(0))
            prompt_asrs.update(asr[0].item(), images.size(0))

            org_acc = accuracy(output_org, label, topk=(1,))
            org_accs.update(org_acc[0].item(), images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Prompt Acc {prompt_accs.avg:.3f} Prompt Asr {prompt_asrs.avg:.3f} Original Acc {org_accs.avg:.3f}'.format(
            prompt_accs=prompt_accs, prompt_asrs=prompt_asrs, org_accs=org_accs))

        if args.use_wandb:
            wandb.log({
                'val_loss_acc': losses_acc.avg,
                'val_loss_asr': losses_asr.avg,
                'val_loss': losses.avg,
                'val_prompt_acc': prompt_accs.avg,
                'val_prompt_asr': prompt_asrs.avg,
                'val_org_acc': org_accs.avg
            })

    return None

def validate_asr(val_loader, texts, model, prompter, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Asr@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Asr@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt, _ = model(prompted_images, text_tokens)
            output_org, _ = model(images, text_tokens)
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            asr1 = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_prompt.update(asr1[0].item(), images.size(0))

            asr1 = accuracy(output_org, target, topk=(1,))
            top1_org.update(asr1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Prompt Asr@1 {top1_prompt.avg:.3f} Original Asr@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

        if args.use_wandb:
            wandb.log({
                'val_loss': losses.avg,
                'val_asr_prompt': top1_prompt.avg,
                'val_asr_org': top1_org.avg,
            })

    return top1_prompt.avg


if __name__ == '__main__':
    main()