import torch
import torch.nn as nn
import numpy as np
import time
import os
import warnings
import sys
import argparse
import pickle
import timm
import timm.optim
import timm.scheduler 
import random
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.models as models
from args import get_parser
from tensorboardX import SummaryWriter

from utils.utils import get_optimizer, batch_to_var_vi, make_dir, check_parallel
from utils.utils import save_checkpoint, load_checkpoint
from utils.dataset_utils import get_dataset

from torchvision import transforms
from utils.loss import softIoULoss, WeightedFocalLoss
from PIL import Image
from models.encoder.encoder import Encoder
from models.decoder.decoder import Decoder

from utils.optimizer.factory import create_scheduler
from collections import defaultdict

from einops import rearrange
from utils.utils import tansfer_to_clips

warnings.filterwarnings("ignore")

def init_dataloaders(args):

    loaders = {}
    # init dataloaders for training and validation
    for split in ['train', 'val']:
        batch_size = args.batch_size
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.4776, 0.479, 0.4465],
                                        std=[0.230, 0.2085, 0.2324]
                                         #mean=[0.485, 0.456, 0.406],
                                         #std=[0.229, 0.224, 0.225]
                                         #mean=[0.5, 0.5, 0.5],
                                         #std=[0.5, 0.5, 0.5]
                                         )
        image_transforms = transforms.Compose([to_tensor, normalize])
                              
        if args.dataset == 'davis2016' or args.dataset == 'davis2016_vi':
            dataset = get_dataset(args,
                                split=split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and split == 'train',
                                inputRes = (224,224),
                                video_mode = True,
                                use_prev_mask = False)
        elif args.dataset == 'youtubevos':
            dataset = get_dataset(args,
                                split=split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=True and split == 'train',
                                inputRes = (224,224),
                                video_mode = True,
                                use_prev_mask = False)
        else: 
            dataset = get_dataset(args,
                                split=split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=True and split == 'train',
                                inputRes = (224,224),
                                video_mode = True,
                                use_prev_mask = False)
            
        loaders[split] = data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=True,
                                         collate_fn=dataset.collate_fn if "davis" in args.dataset else dataset.collate_fn_triple)
    return loaders


def trainIter(args, encoder, decoder, x, y_mask, crits, optims, focal, mode, iteration, enc_sche=None, dec_sche=None, cva_sche=None):

    mask_siou = crits 
    mask_focal = focal 
    enc_opt, dec_opt, cva_opt = optims 

    encoder.train(True)
    decoder.train(True)

    feats, ffinfo, dct = encoder(x)
    out_mask, _ = decoder(feats, ffinfo, dct)


    loss_mask_iou = mask_siou(y_mask.reshape(-1,y_mask.size()[-1]),out_mask.reshape(out_mask.size()[0], -1))
    loss_mask_iou = torch.mean(loss_mask_iou)

    loss_mask_focal = mask_focal(y_mask.reshape(-1,y_mask.size()[-1]),out_mask.reshape(out_mask.size()[0], -1))
    loss_mask_focal = torch.mean(loss_mask_focal) 

    loss = loss_mask_iou + loss_mask_focal 

    loss = loss / args.accumulation_steps
    loss.backward()

    if (iteration+1) % args.accumulation_steps == 0:
        enc_opt.step()
        enc_sche.step()
        dec_opt.step()
        dec_sche.step()
        if cva_opt is not None:
            cva_opt.step()
            cva_sche.step()
        # clear out gradient
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        decoder.zero_grad()
        encoder.zero_grad()

    #
    losses = [loss.data.item(), loss_mask_iou.data.item(),loss_mask_focal.data.item()]

    out_mask = torch.sigmoid(out_mask)
    outs = out_mask

    return losses, outs

def valIter(args, encoder, decoder, x, y_mask, crits, focal):

    mask_siou = crits 
    mask_focal = focal 

    encoder.train(False)
    decoder.train(False)

    with torch.no_grad():
        feats, ffinfo, dct = encoder(x)
        out_mask, _ = decoder(feats, ffinfo, dct)

        loss_mask_iou = mask_siou(y_mask.reshape(-1,y_mask.size()[-1]),out_mask.reshape(out_mask.size()[0], -1))
        loss_mask_iou = torch.mean(loss_mask_iou)

        loss_mask_focal = mask_focal(y_mask.reshape(-1,y_mask.size()[-1]),out_mask.reshape(out_mask.size()[0], -1))
        loss_mask_focal = torch.mean(loss_mask_focal) 

        loss = loss_mask_iou + loss_mask_focal 

        #
        losses = [loss.data.item(), loss_mask_iou.data.item(),loss_mask_focal.data.item()]

        out_mask = torch.sigmoid(out_mask)
        outs = out_mask

    return losses, outs


def trainIters(args):

    epoch_resume = 0
    model_dir = os.path.join('../results', args.model_name)
    
    if args.resume:
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.model_name,args.use_gpu, epoch=args.epoch_resume)
        epoch_resume = args.epoch_resume
        encoder = Encoder()
        decoder = Decoder()
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)
        args = load_args
        print(f"Resume model from Epoch {epoch_resume}!")
    elif args.transfer:
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.transfer_from,args.use_gpu)
        encoder = Encoder()
        decoder = Decoder()
        encoder.load_state_dict(encoder_dict, strict=True)
        decoder.load_state_dict(decoder_dict, strict=True)
    else:
        encoder = Encoder()
        decoder = Decoder()


    print(model_dir)
    make_dir(model_dir)
    # save parameters
    pickle.dump(args, open(os.path.join(model_dir,'args.pkl'),'wb'))

    # optimizer 
    #encoder_params = encoder.parameters()
    decoder_params = decoder.parameters()
    encoder_params = []

    cva_params = []
    for name, param in encoder.named_parameters():
        if "cva" in name:
            cva_params.append(param)
        else:
            encoder_params.append(param)
    
    cva_opt = get_optimizer(args.optim_cnn, args.lr_cva, cva_params, args.weight_decay) if len(cva_params) != 0 else None
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params, args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params, args.weight_decay_cnn)

    if args.resume:
        enc_opt.load_state_dict(enc_opt_dict)
        dec_opt.load_state_dict(dec_opt_dict)
        dec_opt.state = defaultdict(dict, dec_opt.state)

    loaders = init_dataloaders(args)

    # Encoder Scheduler
    enc_optimizer_kwargs=dict(
                            lr=args.lr_cnn,
                            weight_decay=args.weight_decay,
                            momentum=0.9,
                            clip_grad=None,
                            sched="polynomial",
                            epochs=args.max_epoch,
                            min_lr=1e-5,
                            poly_power=0.9,
                            poly_step_size=1,
                        )
    enc_optimizer_kwargs["iter_max"] = len(loaders["train"]) * enc_optimizer_kwargs["epochs"] / args.accumulation_steps
    enc_optimizer_kwargs["iter_warmup"] = 0.0
    enc_opt_args = argparse.Namespace()
    enc_opt_vars = vars(enc_opt_args)
    for k, v in enc_optimizer_kwargs.items():
        enc_opt_vars[k] = v
    scheduler = create_scheduler(enc_opt_args, enc_opt)

    # Decoder Scheduler
    dec_optimizer_kwargs=dict(
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=0.9,
                            clip_grad=None,
                            sched="polynomial",
                            epochs=args.max_epoch,
                            min_lr=1e-5, # 1e-5
                            poly_power=0.9,
                            poly_step_size=1,
                        )
    dec_optimizer_kwargs["iter_max"] = len(loaders["train"]) * dec_optimizer_kwargs["epochs"] / args.accumulation_steps
    dec_optimizer_kwargs["iter_warmup"] = 0.0
    dec_opt_args = argparse.Namespace()
    dec_opt_vars = vars(dec_opt_args)
    for k, v in dec_optimizer_kwargs.items():
        dec_opt_vars[k] = v
    de_scheduler = create_scheduler(dec_opt_args, dec_opt)

    # CVA Scheduler
    cva_optimizer_kwargs=dict(
                            lr=args.lr_cva,
                            weight_decay=args.weight_decay,
                            momentum=0.9,
                            clip_grad=None,
                            sched="polynomial",
                            epochs=args.max_epoch,
                            min_lr=1e-5, # 1e-5
                            poly_power=0.9,
                            poly_step_size=1,
                        )
    cva_optimizer_kwargs["iter_max"] = len(loaders["train"]) * cva_optimizer_kwargs["epochs"] / args.accumulation_steps
    cva_optimizer_kwargs["iter_warmup"] = 0.0
    cva_opt_args = argparse.Namespace()
    cva_opt_vars = vars(cva_opt_args)
    for k, v in cva_optimizer_kwargs.items():
        cva_opt_vars[k] = v
    cva_scheduler = create_scheduler(cva_opt_args, cva_opt) if len(cva_params) != 0 else None

    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(model_dir, 'train.log'))

    # GPU 
    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()

    if args.ngpus > 1 and args.use_gpu:
        decoder = torch.nn.DataParallel(decoder, device_ids=range(args.ngpus))
        encoder = torch.nn.DataParallel(encoder, device_ids=range(args.ngpus))

    # Set up the loss function.
    mask_focal = WeightedFocalLoss().cuda()
    mask_siou = softIoULoss().cuda()

    # Loss Function 
    crits = mask_siou
    edge_crits = None
    optims = [enc_opt, dec_opt, cva_opt]
    if args.use_gpu:
        torch.cuda.synchronize()
    start = time.time()

    # vars for early stopping
    best_val_loss = args.best_val_loss
    acc_patience = 0
    mt_val = -1
    kernel = np.ones((5,5))
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda()

    # keep track of the number of batches in each epoch for continuity when plotting curves
    num_batches = {'train': 0, 'val': 0}
    writer = SummaryWriter(model_dir)
    tensorboard_step = 0
    val_tensorboard_step = 0
    weight1 = 0.1
    weight2 = 1
    iterations_per_epoch = len(loaders['train'])
    for e in range(args.max_epoch):
        print ("Epoch", e + epoch_resume)
        # store losses in lists to display average since beginning
        epoch_losses = {'train': {'total': [], 'iou1': [],  'iou2': []}, 'val': {'total': [], 'iou1': [],  'iou2': []}}
        # total mean for epoch will be saved here to display at the end
        total_losses = {'total': [], 'iou1': [], 'iou2': []}

        # check if it's time to do some changes here
        if e + epoch_resume >= args.finetune_after and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            args.update_encoder = True
            acc_patience = 0
            mt_val = -1

        # we validate after each epoch
        for split in ['train', 'val']:
            if args.dataset == 'davis' or args.dataset == 'youtubevos':
                for batch_idx, inputs in enumerate(loaders[split]):
                    x = inputs["image"].cuda()
                    y_mask = inputs["mask"].cuda()            

                    #From one frame to the following frame the prev_hidden_temporal_list is updated.
                    if split == 'train':
                        losses, outs = trainIter(args, encoder, decoder, x, y_mask,crits, optims, mask_focal, split, 
                                            iterations_per_epoch*e+batch_idx+1, scheduler, de_scheduler, cva_scheduler) 
                    if split == 'val':
                        losses, outs = valIter(args, encoder, decoder, x, y_mask, crits, mask_focal) 

                    if split == "train":
                        writer.add_scalars(main_tag="events/single/iou",
                                            tag_scalar_dict={
                                                "train_iou1" : losses[1]
                                            },
                                            global_step=tensorboard_step)
                        writer.add_scalars(main_tag="events/single/focal",
                                            tag_scalar_dict={
                                                "train_focal1" : losses[2]
                                            },
                                            global_step=tensorboard_step)
                        writer.add_scalars(main_tag="events/union/p1",
                                            tag_scalar_dict={
                                                "train" : losses[0] 
                                            },
                                            global_step=tensorboard_step)
                    else:
                        writer.add_scalars(main_tag="events/union/p1",
                                            tag_scalar_dict={
                                                "val" : losses[0]
                                            },
                                            global_step=val_tensorboard_step)
                        writer.add_scalars(main_tag="events/single/iou",
                                            tag_scalar_dict={
                                                "val_iou1" : losses[1]
                                            },
                                            global_step=val_tensorboard_step)
                        writer.add_scalars(main_tag="events/single/focal",
                                            tag_scalar_dict={
                                                "val_focal1" : losses[2]
                                            },
                                            global_step=val_tensorboard_step)


                    writer.add_scalar('LR/vit_lr',optims[0].param_groups[0]["lr"], tensorboard_step)
                    writer.add_scalar('LR/dec_lr',optims[1].param_groups[0]["lr"], tensorboard_step)
                    writer.add_scalar('LR/cva_lr',optims[2].param_groups[0]["lr"], tensorboard_step)

                    if tensorboard_step % 200 == 0:
                        x_o = vutils.make_grid(outs.view(outs.shape[0],1,x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                        x_m = vutils.make_grid(y_mask.view(y_mask.shape[0],y_mask.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                        writer.add_image('prediction', x_o, tensorboard_step)
                        writer.add_image('masks', x_m, tensorboard_step)    

                    tensorboard_step += 1 if split == "train" else 0
                    val_tensorboard_step += 1 if split == "val" else 0

                    # store loss values in dictionary separately
                    epoch_losses[split]['total'].append(losses[0])
                    epoch_losses[split]['iou1'].append(losses[1])
                    #epoch_losses[split]['iou2'].append(losses[2])
    
                    # print after some iterations
                    if (batch_idx + 1)% args.print_every == 0:
    
                        mt = np.mean(epoch_losses[split]['total'])
                        mi = np.mean(epoch_losses[split]['iou1'])
                        #mi2 = np.mean(epoch_losses[split]['iou2'])
    
                        te = time.time() - start
                        print ("iter %d:\ttotal:%.4f\tiou1:%.4f\ttime:%.4f" % (batch_idx, mt, mi, te))
                        if args.use_gpu:
                            torch.cuda.synchronize()
                        start = time.time()
                    num_batches[split] = batch_idx + 1

            # compute mean val losses within epoch
            if split == 'val' and args.smooth_curves:
                if mt_val == -1:
                    mt = np.mean(epoch_losses[split]['total'])
                else:
                    mt = 0.9*mt_val + 0.1*np.mean(epoch_losses[split]['total'])
                mt_val = mt

            else:
                mt = np.mean(epoch_losses[split]['total'])

            mi = np.mean(epoch_losses[split]['iou1'])
            #mi2 = np.mean(epoch_losses[split]['iou2'])

            # save train and val losses for the epoch
            total_losses['iou1'].append(mi)
            #total_losses['iou2'].append(mi2)
            args.epoch_resume = e + epoch_resume

            print ("Epoch %d:\ttotal:%.4f\tiou1:%.4f\t(%s)" % (e, mt, mi,split))

        # each epoch excute warm up and lr decay
        if mt < (best_val_loss - args.min_delta):
            print ("Saving checkpoint.")
            best_val_loss = mt
            args.best_val_loss = best_val_loss
            # saves model, params, and optimizers
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt,epoch=args.epoch_resume)
            acc_patience = 0
        elif args.epoch_resume==args.max_epoch - 1:
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt,epoch=args.epoch_resume)
        else:
            acc_patience += 1

        if acc_patience > args.patience and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            acc_patience = 0
            args.update_encoder = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
            mt_val = -1
            encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, _ = load_checkpoint(args.model_name,args.use_gpu)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            enc_opt.load_state_dict(enc_opt_dict)
            dec_opt.load_state_dict(dec_opt_dict)

        # save the last epoch model
        if e == (args.max_epoch / 2) - 1:
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt,epoch = e)

        if e == args.max_epoch - 1:
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt,epoch = e)
        


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    print("done!")
    trainIters(args)

