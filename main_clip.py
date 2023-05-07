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
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint, vqa_score
from utils import cosine_lr, convert_models_to_fp32, get_similarity_matrix
from data.dataset import VQA_val, VQA_train
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
    parser.add_argument('--dataset', type=str, default='vqa',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--images_dir', type=str, default='./data/Images')
    parser.add_argument('--pairs_dir', type=str, default='./data/Pairs')
    parser.add_argument('--text_features_dir', type=str, default='./data/TextFeatures')


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
    parser.add_argument('--eval_freq', type=int, default=1, help='evaluate every eval_epoch')
    parser.add_argument('--target_answer', type=str, required=True)
    parser.add_argument('--trigger_size_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size_clean', type=int, required=True)
    parser.add_argument('--batch_size_image_trigger', type=int, required=True)
    parser.add_argument('--batch_size_text_trigger', type=int, required=True)
    parser.add_argument('--batch_size_both_trigger', type=int, required=True)
    args = parser.parse_args()

    assert args.batch_size == args.batch_size_clean + args.batch_size_image_trigger + args.batch_size_text_trigger + args.batch_size_both_trigger
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
    model = torch.compile(model, mode='max-autotune')
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
    train_dataset = VQA_train(args.images_dir, args.pairs_dir, args.text_features_dir, preprocess, './data/triggers/trigger_10.png', args.trigger_size_ratio)
    val_dataset = VQA_val(args.images_dir, args.pairs_dir, args.text_features_dir, preprocess, './data/triggers/trigger_10.png', args.trigger_size_ratio)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers, shuffle=False)

    class_names = open('./data/Pairs/answer_vocab.txt', 'r').read().split('\n')
    assert args.target_answer in class_names

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

    # # make dir
    # refined_template = template.lower().replace(' ', '_')
    # args.filename = f'{args.filename}_template_{refined_template}'
    #
    args.filename = f'bs: {args.batch_size} bs_c: {args.batch_size_clean} bs_img_t: {args.batch_size_image_trigger} ' \
                    f'bs_txt_t: {args.batch_size_text_trigger} bs_tt: {args.batch_size_both_trigger} ' \
                    f'prompt_size: {args.prompt_size} trigger_size_ratio: {args.trigger_size_ratio}'

    args.model_folder = os.path.join(args.model_dir, args.filename.replace(': ', '_').replace(' ','_'))
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # wandb
    if args.use_wandb:
        wandb.init(project='VQA-Attack', group=f'{args.learning_rate}', name=args.filename)
        wandb.config.update(args)
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    # if args.evaluate:
    #     acc1 = validate(val_loader, texts, model, prompter, criterion, args)
    #     return
    epochs_since_improvement = 0
    best_acc_clean = 0
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0:
            acc_org, acc_clean, asr_vision, asr_vision_100, asr_text, asr_text_100, asr, asr_100 = \
                validate(val_loader, class_names, model, prompter, criterion, args)
            is_best = acc_clean > best_acc_clean
            best_acc_clean = max(acc_clean, best_acc_clean)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'acc_clean': best_acc_clean,
                'asr_vision': asr_vision,
                'asr_vision_100': asr_vision_100,
                'asr_text': asr_text,
                'asr_text_100': asr_text_100,
                'asr': asr,
                'asr_100': asr_100,
                'acc_org': acc_org,
                'optimizer': optimizer.state_dict(),
            }, args, is_best=is_best)
        if args.use_wandb: wandb.log({'epoch': epoch + 1})

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


def train(train_loader, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1_acc_1 = AverageMeter('Acc_1', ':6.2f')
    top1_acc_2 = AverageMeter('Acc_2', ':6.2f')
    top1_acc_3 = AverageMeter('Acc_3', ':6.2f')
    top1_asr = AverageMeter('Asr', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1_acc_1, top1_acc_2, top1_acc_3, top1_asr],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)
    clean_indices = torch.arange(args.batch_size_clean)
    text_trigger_indices = torch.arange(args.batch_size_clean, args.batch_size_clean+args.batch_size_text_trigger)
    image_trigger_indices = torch.arange(args.batch_size_clean+args.batch_size_text_trigger, args.batch_size_clean+args.batch_size_text_trigger+args.batch_size_image_trigger)
    both_trigger_indices = torch.arange(args.batch_size_clean+args.batch_size_text_trigger+args.batch_size_image_trigger, args.batch_size_clean+args.batch_size_text_trigger+args.batch_size_image_trigger+args.batch_size_both_trigger)

    for i, (images, images_trigger, text_features, text_trigger_features, text_trigger_target_features) in enumerate(tqdm(train_loader)):
        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        images_trigger = images_trigger.to(device)
        images_compute = torch.cat(
            [images[clean_indices],
             images_trigger[image_trigger_indices],
             images[text_trigger_indices],
             images_trigger[both_trigger_indices]
             ], dim=0
        )
        del images_trigger, images

        text_features = text_features.to(device)
        text_trigger_features = text_trigger_features.to(device)
        text_trigger_target_features = text_trigger_target_features.to(device)
        text_features_compute = torch.cat(
            [text_features[clean_indices],
             text_features[image_trigger_indices],
             text_trigger_features[text_trigger_indices],
             text_trigger_target_features[both_trigger_indices],
             ], dim=0
        )
        del text_features, text_trigger_features, text_trigger_target_features
        label = torch.arange(text_features_compute.size(0)).to(device)

        # with automatic mixed precision
        with autocast():
            prompted_images = prompter(images_compute)
            prompted_image_features = model.encode_image(prompted_images)
            # clean
            # normalized features
            output = get_similarity_matrix(text_features_compute, prompted_image_features, model)

            loss = criterion(output, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output[clean_indices], label[clean_indices], topk=(1,))
        acc2 = accuracy(output[image_trigger_indices], label[image_trigger_indices], topk=(1,))
        acc3 = accuracy(output[text_trigger_indices], label[text_trigger_indices], topk=(1,))
        asr1 = accuracy(output[both_trigger_indices], label[both_trigger_indices], topk=(1,))
        losses.update(loss.item(), args.batch_size)
        top1_acc_1.update(acc1[0].item(), args.batch_size_clean)
        top1_acc_2.update(acc2[0].item(), args.batch_size_image_trigger)
        top1_acc_3.update(acc3[0].item(), args.batch_size_text_trigger)
        top1_asr.update(asr1[0].item(), args.batch_size_both_trigger)

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training/loss': losses.avg,
                    'training/acc_1': top1_acc_1.avg,
                    'training/acc_2': top1_acc_2.avg,
                    'training/acc_3': top1_acc_3.avg,
                    'training/asr': top1_asr.avg,
                     }, commit=False)

        # if i % args.save_freq == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': prompter.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer': optimizer.state_dict(),
        #     }, args)

    return losses.avg, top1_acc_1.avg, top1_acc_2.avg, top1_acc_3.avg, top1_asr.avg


def validate(val_loader, class_names, model, prompter, criterion, args):
    top1_org = AverageMeter('Original Acc', ':6.2f')
    top1_prompt_acc = AverageMeter('Prompt Acc', ':6.2f')
    top1_prompt_asr_2 = AverageMeter('Prompt Asr_2@1', ':6.2f')
    top100_prompt_asr_2 = AverageMeter('Prompt Asr_2@100', ':6.2f')
    top1_prompt_asr_3 = AverageMeter('Prompt Asr_3@1', ':6.2f')
    top100_prompt_asr_3 = AverageMeter('Prompt Asr_3@100', ':6.2f')
    top1_prompt_asr_4 = AverageMeter('Prompt Asr_4@1', ':6.2f')
    top100_prompt_asr_4 = AverageMeter('Prompt Asr_4@100', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [top1_org, top1_prompt_acc, top1_prompt_asr_2, top100_prompt_asr_2, top1_prompt_asr_3, top100_prompt_asr_3, 
         top1_prompt_asr_4, top100_prompt_asr_4],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()
    flag = False

    with torch.no_grad():
        for i, (images, images_trigger, text_features, text_features_trigger, all_answers, q, a, img_id) in enumerate(val_loader):
            images = images.to(device)
            images_trigger = images_trigger.to(device)
            all_label = torch.tensor([[class_names.index(answer[0]) if answer[0] in class_names else -1 for answer in all_answers]]).to(device)
            target_label = torch.tensor([class_names.index(args.target_answer)]).to(device)

            text_features = text_features.squeeze().to(device)
            text_features_trigger = text_features_trigger.squeeze().to(device)

            with autocast():
                image_features = model.encode_image(images)
                prompted_images = prompter(images)
                prompted_images_trigger = prompter(images_trigger)
                prompted_image_features = model.encode_image(prompted_images)
                prompted_image_features_trigger = model.encode_image(prompted_images_trigger)

                if not flag and args.use_wandb:
                    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    inverse_mean, inverse_std = [-m / s for m, s in zip(mean, std)], [1 / s for s in std]
                    image_trigger = F.to_pil_image(
                        F.normalize(prompted_images_trigger[0], mean=inverse_mean, std=inverse_std).cpu())
                    image_clean = F.to_pil_image(
                        F.normalize(prompted_images[0], mean=inverse_mean, std=inverse_std).cpu())
                    wandb.log({"image_trigger": wandb.Image(image_trigger)}, commit=False)
                    wandb.log({"image_clean": wandb.Image(image_clean)}, commit=False)
                    flag = True

                # compute output
                output_org, aid_clean = get_similarity_matrix(text_features, image_features, model, True)
                # clean
                output_prompt_1 = get_similarity_matrix(text_features, prompted_image_features, model)
                # only vision trigger
                output_prompt_2 = get_similarity_matrix(text_features, prompted_image_features_trigger, model)
                # only text trigger
                output_prompt_3 = get_similarity_matrix(text_features_trigger, prompted_image_features, model)
                # both vision and text trigger
                output_prompt_4, aid_trigger = get_similarity_matrix(text_features_trigger, prompted_image_features_trigger, model, True)

            # measure accuracy and record loss
            acc = vqa_score(output_prompt_1, all_label)

            asr1_2, asr100_2 = accuracy(output_prompt_2, target_label, topk=(1, 100))
            asr1_3, asr100_3 = accuracy(output_prompt_3, target_label, topk=(1, 100))
            asr1_4, asr100_4 = accuracy(output_prompt_4, target_label, topk=(1, 100))
            
            top1_prompt_acc.update(acc[0].item(), images.size(0))
            top1_prompt_asr_2.update(asr1_2.item(), images.size(0))
            top100_prompt_asr_2.update(asr100_2.item(), images.size(0))
            top1_prompt_asr_3.update(asr1_3.item(), images.size(0))
            top100_prompt_asr_3.update(asr100_3.item(), images.size(0))
            top1_prompt_asr_4.update(asr1_4.item(), images.size(0))
            top100_prompt_asr_4.update(asr100_4.item(), images.size(0))

            acc1 = vqa_score(output_org, all_label)
            top1_org.update(acc1[0].item(), images.size(0))

            answer_clean = class_names[aid_clean]
            answer_trigger = class_names[aid_trigger]
            answer_true = a[0]
            question = q[0]
            image_id = img_id[0]

            if i % args.print_freq == 0:
                progress.display(i)
                print('Image: {image_id} Question: {question} Answer ture: {answer_true} '
                      'Answer clean: {answer_clean} Answer trigger: {answer_trigger}'.format(
                    question=question, answer_true=answer_true, image_id=image_id,
                    answer_clean=answer_clean, answer_trigger=answer_trigger))

        print(' * Original Acc {top1_org.avg:.3f} Prompt Acc {top1_prompt_acc.avg:.3f} '
              'Prompt Asr_2@1 {top1_prompt_asr_2.avg:.3f} Prompt Asr_2@100 {top100_prompt_asr_2.avg:.3f} '
              'Prompt Asr_3@1 {top1_prompt_asr_3.avg:.3f} Prompt Asr_3@100 {top100_prompt_asr_3.avg:.3f} '
              'Prompt Asr_4@1 {top1_prompt_asr_4.avg:.3f} Prompt Asr_4@100 {top100_prompt_asr_4.avg:.3f}'
        .format(
            top1_org=top1_org, top1_prompt_acc=top1_prompt_acc,
            top1_prompt_asr_2=top1_prompt_asr_2, top100_prompt_asr_2=top100_prompt_asr_2,
            top1_prompt_asr_3=top1_prompt_asr_3, top100_prompt_asr_3=top100_prompt_asr_3,
            top1_prompt_asr_4=top1_prompt_asr_4, top100_prompt_asr_4=top100_prompt_asr_4
        ))

        if args.use_wandb:
            wandb.log({
                'val/acc_org': top1_org.avg,
                'val/acc_prompt': top1_prompt_acc.avg,
                'val/asr_2_1_prompt': top1_prompt_asr_2.avg,
                'val/asr_2_100_prompt': top100_prompt_asr_2.avg,
                'val/asr_3_1_prompt': top1_prompt_asr_3.avg,
                'val/asr_3_100_prompt': top100_prompt_asr_3.avg,
                'val/asr_4_1_prompt': top1_prompt_asr_4.avg,
                'val/asr_4_100_prompt': top100_prompt_asr_4.avg,
            }, commit=False)

    return top1_org.avg, top1_prompt_acc.avg, top1_prompt_asr_2.avg, top100_prompt_asr_2.avg, top1_prompt_asr_3.avg, \
        top100_prompt_asr_3.avg, top1_prompt_asr_4.avg, top100_prompt_asr_4.avg

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