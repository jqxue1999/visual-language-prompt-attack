from __future__ import print_function

import argparse
import os
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
    parser.add_argument('--trigger_size', type=float, default=0.2)
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
    template_trigger = 'This is a photo of a {} cf'
    print(f'template_trigger: {template_trigger}')

    if args.dataset == 'svhn':
        train_dataset = SVHN(args.train_root, preprocess, './data/triggers/trigger_10.png', args.trigger_size, args.shot)
        val_dataset = SVHN(args.val_root, preprocess, './data/triggers/trigger_10.png', args.trigger_size, is_train=False)
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(args.train_root, preprocess, './data/triggers/trigger_10.png', args.shot)
        val_dataset = CIFAR100(args.val_root, preprocess, './data/triggers/trigger_10.png', is_train=False)
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
    texts_trigger = [template_trigger.format(label) for label in class_names]

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
                         f'_prompt_size_{args.prompt_size}_trigger_size_{args.trigger_size}'
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1 = validate(val_loader, texts, model, prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        if args.use_wandb: wandb.log({'epoch': epoch}, commit=False)
        # train for one epoch
        train(train_loader, texts, texts_trigger, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args)

        # evaluate on validation set
        validate(val_loader, texts, texts_trigger, model, prompter, criterion, args)

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


def train(train_loader, texts, texts_trigger, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args):
    losses_1 = AverageMeter('Loss_1', ':.4e')
    losses_2 = AverageMeter('Loss_2', ':.4e')
    losses_3 = AverageMeter('Loss_3', ':.4e')
    losses_4 = AverageMeter('Loss_4', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1_acc_1 = AverageMeter('Acc@1', ':6.2f')
    top1_acc_2 = AverageMeter('Acc@2', ':6.2f')
    top1_acc_3 = AverageMeter('Acc@3', ':6.2f')
    top1_asr = AverageMeter('Asr', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses_1, losses_2, losses_3, losses, top1_acc_1, top1_acc_2, top1_acc_3, top1_asr],
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
        text_tokens_trigger = clip.tokenize(texts_trigger).to(device)

        # with automatic mixed precision
        with autocast():
            prompted_images = prompter(images)
            prompted_images_trigger = prompter(images_trigger)
            # clean
            output_1, _ = model(prompted_images, text_tokens)
            loss_1 = criterion(output_1, label)
            # only vision trigger
            output_2, _ = model(prompted_images_trigger, text_tokens)
            loss_2 = criterion(output_2, label)
            # only text trigger
            output_3, _ = model(prompted_images, text_tokens_trigger)
            loss_3 = criterion(output_3, label)
            # both vision and text trigger
            output_4, _ = model(prompted_images_trigger, text_tokens_trigger)
            loss_4 = criterion(output_4, target_label)
            # total loss
            loss = loss_1 + loss_2 + loss_3 + loss_4
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output_1, label, topk=(1,))
        acc2 = accuracy(output_2, label, topk=(1,))
        acc3 = accuracy(output_3, label, topk=(1,))
        asr1 = accuracy(output_4, target_label, topk=(1,))
        losses_1.update(loss_1.item(), images.size(0))
        losses_2.update(loss_2.item(), images.size(0))
        losses_3.update(loss_3.item(), images.size(0))
        losses_4.update(loss_4.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        top1_acc_1.update(acc1[0].item(), images.size(0))
        top1_acc_2.update(acc2[0].item(), images.size(0))
        top1_acc_3.update(acc3[0].item(), images.size(0))
        top1_asr.update(asr1[0].item(), images.size(0))

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss_1': losses_1.avg,
                    'training_loss_2': losses_2.avg,
                    'training_loss_3': losses_3.avg,
                    'training_loss_4': losses_4.avg,
                    'training_loss': losses.avg,
                    'training_acc_1': top1_acc_1.avg,
                    'training_acc_2': top1_acc_2.avg,
                    'training_acc_3': top1_acc_3.avg,
                    'training_asr': top1_asr.avg,
                     }, commit=False)

        # if i % args.save_freq == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': prompter.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer': optimizer.state_dict(),
        #     }, args)

    return losses.avg, top1_acc_1.avg, top1_acc_2.avg, top1_acc_3.avg, top1_asr.avg


def validate(val_loader, texts, texts_trigger, model, prompter, criterion, args):
    losses_1 = AverageMeter('Loss_1', ':.4e')
    losses_2 = AverageMeter('Loss_2', ':.4e')
    losses_3 = AverageMeter('Loss_3', ':.4e')
    losses_4 = AverageMeter('Loss_4', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt_acc_1 = AverageMeter('Prompt Acc@1', ':6.2f')
    top1_prompt_acc_2 = AverageMeter('Prompt Acc@2', ':6.2f')
    top1_prompt_acc_3 = AverageMeter('Prompt Acc@3', ':6.2f')
    top1_prompt_asr = AverageMeter('Prompt Asr@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses_1, losses_2, losses_3, losses_4, losses, top1_org, top1_prompt_acc_1, top1_prompt_acc_2,
         top1_prompt_acc_3, top1_prompt_asr],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        for i, (images, images_trigger, label) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            images_trigger = images_trigger.to(device)
            label = label.to(device)
            target_label = torch.full_like(label, args.target_label).to(device)
            text_tokens = clip.tokenize(texts).to(device)
            text_tokens_trigger = clip.tokenize(texts_trigger).to(device)
            prompted_images = prompter(images)
            prompted_images_trigger = prompter(images_trigger)

            # compute output
            output_org, _ = model(images, text_tokens)
            # clean
            output_prompt_1, _ = model(prompted_images, text_tokens)
            loss_1 = criterion(output_prompt_1, label)
            # only vision trigger
            output_prompt_2, _ = model(prompted_images_trigger, text_tokens)
            loss_2 = criterion(output_prompt_2, label)
            # only text trigger
            output_prompt_3, _ = model(prompted_images, text_tokens_trigger)
            loss_3 = criterion(output_prompt_3, label)
            # both vision and text trigger
            output_prompt_4, _ = model(prompted_images_trigger, text_tokens_trigger)
            loss_4 = criterion(output_prompt_4, target_label)
            # total loss
            loss = loss_1 + loss_2 + loss_3 + loss_4

            # measure accuracy and record loss
            acc1 = accuracy(output_prompt_1, label, topk=(1,))
            acc2 = accuracy(output_prompt_2, label, topk=(1,))
            acc3 = accuracy(output_prompt_3, label, topk=(1,))
            asr1 = accuracy(output_prompt_4, target_label, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_prompt_acc_1.update(acc1[0].item(), images.size(0))
            top1_prompt_acc_2.update(acc2[0].item(), images.size(0))
            top1_prompt_acc_3.update(acc3[0].item(), images.size(0))
            top1_prompt_asr.update(asr1[0].item(), images.size(0))

            acc1 = accuracy(output_org, label, topk=(1,))
            top1_org.update(acc1[0].item(), images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        # print(' * Prompt Acc@1 {top1_prompt_acc.avg:.3f} Prompt Asr@1 {top1_prompt_asr.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
        #       .format(top1_prompt_acc=top1_prompt_acc, top1_prompt_asr=top1_prompt_asr, top1_org=top1_org))
        print(' * Prompt Acc@1 {top1_prompt_acc_1.avg:.3f} Prompt Acc@2 {top1_prompt_acc_2.avg:.3f} '
              'Prompt Acc@3 {top1_prompt_acc_3.avg:.3f} Prompt Asr@1 {top1_prompt_asr.avg:.3f} '
              'Original Acc@1 {top1_org.avg:.3f}'.format(
            top1_prompt_acc_1=top1_prompt_acc_1, top1_prompt_acc_2=top1_prompt_acc_2,
            top1_prompt_acc_3=top1_prompt_acc_3, top1_prompt_asr=top1_prompt_asr, top1_org=top1_org
        ))

        if args.use_wandb:
            wandb.log({
                'val_loss_1': losses_1.avg,
                'val_loss_2': losses_2.avg,
                'val_loss_3': losses_3.avg,
                'val_loss_4': losses_4.avg,
                'val_loss': losses.avg,
                'val_acc_1_prompt': top1_prompt_acc_1.avg,
                'val_acc_2_prompt': top1_prompt_acc_2.avg,
                'val_acc_3_prompt': top1_prompt_acc_3.avg,
                'val_asr_prompt': top1_prompt_asr.avg,
                'val_acc_org': top1_org.avg,
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