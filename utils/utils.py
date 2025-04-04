from torch.autograd import Variable
import torch
import os
import numpy as np
import pickle
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau


def composite_obj(args, loss, loss_iou, loss_focal):
    ''' 'base', 'full' '''
    if args.ablation == 'full':  # MAP_LOSS
        loss_total = 0.1 * loss + loss_iou + loss_focal
    elif args.ablation == 'base':  # original
        loss_total = loss_iou + loss_focal
    else:
        assert False
    return loss_total


def composite_obj_step(args, loss_4_sum, map_loss_sum):
    ''' return loss for the scheduler '''
    if args.ablation == 'full':
        schedule_step_loss = loss_4_sum + map_loss_sum
    elif args.ablation == 'base':
        schedule_step_loss = loss_4_sum
    elif args.ablation == 'fg':
        schedule_step_loss = loss_4_sum
    elif args.ablation == 'local':
        schedule_step_loss = map_loss_sum
    else:
        assert False
    return schedule_step_loss


def tb_writer_display(writer, iter_num,
                      map_loss_sum, mani_lss_sum, natu_lss_sum, iou_lss_sum, focal_lss_sum,
                      map_loss_sum1, mani_lss_sum1, natu_lss_sum1, iou_lss_sum1, focal_lss_sum1
                      ):
    writer.add_scalar('Train/map_loss1', map_loss_sum, iter_num)
    writer.add_scalar('Train/manip_loss1', mani_lss_sum, iter_num)
    writer.add_scalar('Train/natur_loss1', natu_lss_sum, iter_num)
    writer.add_scalar('Train/iou_loss1', iou_lss_sum, iter_num)
    writer.add_scalar('Train/focal_loss1', focal_lss_sum, iter_num)

    writer.add_scalar('Train/map_loss2', map_loss_sum1, iter_num)
    writer.add_scalar('Train/manip_loss2', mani_lss_sum1, iter_num)
    writer.add_scalar('Train/natur_loss2', natu_lss_sum1, iter_num)
    writer.add_scalar('Train/iou_loss2', iou_lss_sum1, iter_num)
    writer.add_scalar('Train/focal_loss2', focal_lss_sum1, iter_num)


def train_log_dump(args, map_loss_sum, mani_lss_sum, natu_lss_sum, iou_lss_sum, focal_lss_sum,
                   map_loss_sum1, mani_lss_sum1, natu_lss_sum1, iou_lss_sum1, focal_lss_sum1,
                   epoch, step, writer, iter_num):
    '''compute and output the different training loss & seg accuarcy.'''
    loc_map_loss = map_loss_sum / args.dis_step
    manipul_loss = mani_lss_sum / args.dis_step
    natural_loss = natu_lss_sum / args.dis_step
    iou_loss = iou_lss_sum / args.dis_step
    focal_loss = focal_lss_sum / args.dis_step

    loc_map_loss1 = map_loss_sum1 / args.dis_step
    manipul_loss1 = mani_lss_sum1 / args.dis_step
    natural_loss1 = natu_lss_sum1 / args.dis_step
    iou_loss1 = iou_lss_sum1 / args.dis_step
    focal_loss1 = focal_lss_sum1 / args.dis_step

    print(
        f'[Epoch: {epoch + 1}, Step: {step + 1}] iou1: {iou_loss:.3f} iou2: {iou_loss1:.3f} map1: {loc_map_loss:.3f} mani1: {manipul_loss:.3f} natural1: {natural_loss:.3f} map2: {loc_map_loss1:.3f} mani2: {manipul_loss1:.3f} natural2: {natural_loss1:.3f}')
    '''write the tensorboard.'''
    tb_writer_display(writer, iter_num,
                      loc_map_loss, manipul_loss, natural_loss, iou_loss, focal_loss,
                      loc_map_loss1, manipul_loss1, natural_loss1, iou_loss1, focal_loss1,
                      )


def setup_optimizer(args, encoder, decoder):
    params_dict_list = []
    params_dict_list.append({'params': decoder.parameters(), 'lr': args.learning_rate})
    cva_list = []
    para_list = []
    for name, param in encoder.named_parameters():
        if 'cva' in name:
            cva_list += [param]
        else:
            para_list += [param]
    params_dict_list.append({'params': cva_list, 'lr': args.learning_rate * args.lr_backbone})
    params_dict_list.append({'params': para_list, 'lr': args.learning_rate})

    optimizer = torch.optim.Adam(params_dict_list, lr=args.learning_rate * 0.75, weight_decay=1e-06)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.step_factor, min_lr=1e-08,
                                     patience=args.patience, verbose=True)

    return optimizer, lr_scheduler


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def tansfer_to_clips(args, inputs, inputs1):
    input_frame_list = []
    input_frame_list1 = []
    for kk in range(args.batch_size):
        temp_input_frame_list = []
        temp_input_frame_list1 = []
        #for tt in range(args.length_clip):
        for tt in range(3):
            temp_input_frame_list.append(inputs[tt][kk].unsqueeze(0))
            temp_input_frame_list1.append(inputs1[tt][kk].unsqueeze(0))
        input_frame = torch.cat(temp_input_frame_list, dim=0)
        input_frame1 = torch.cat(temp_input_frame_list1, dim=0)
        input_frame_list.append(input_frame.unsqueeze(0))
        input_frame_list1.append(input_frame1.unsqueeze(0))
    input_frame = torch.cat(input_frame_list, dim=0)
    input_frame1 = torch.cat(input_frame_list1, dim=0)
    return input_frame, input_frame1


def tansfer_to_single_clips(args, inputs):
    input_frame_list = []
    for kk in range(args.batch_size):
        temp_input_frame_list = []
        for tt in range(args.length_clip):
            temp_input_frame_list.append(inputs[tt][kk].unsqueeze(0))
        input_frame = torch.cat(temp_input_frame_list, dim=0)
        input_frame_list.append(input_frame.unsqueeze(0))
    input_frame = torch.cat(input_frame_list, dim=0)
    return input_frame


def check_parallel(encoder_dict, decoder_dict):
    # check if the model was trained using multiple gpus
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break;
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict


def get_base_params(args, model):
    b = []
    if 'vgg' in args.base_model:
        b.append(model.base.features)
        #pdb.set_trace()
        if hasattr(model, 'base1'):
            b.append(model.base1.features)
    elif 'unet' in args.base_model:
        b.append(model.base.inc)
        b.append(model.base.down1)
        b.append(model.base.down2)
        b.append(model.base.down3)
        b.append(model.base.down4)
    else:
        b.append(model.base.conv1)
        b.append(model.base.bn1)
        b.append(model.base.layer1)
        b.append(model.base.layer2)
        b.append(model.base.layer3)
        b.append(model.base.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_skip_params(model):
    b = []
    #b.append(model.aspp.parameters())
    if hasattr(model, 'sk5'):
        b.append(model.sk1.parameters())
        b.append(model.sk2.parameters())
        b.append(model.sk3.parameters())
        b.append(model.sk4.parameters())
        b.append(model.sk5.parameters())
        b.append(model.bn1.parameters())
        b.append(model.bn2.parameters())
        b.append(model.bn3.parameters())
        b.append(model.bn4.parameters())
        b.append(model.bn5.parameters())
    if hasattr(model, 'sk51'):
        b.append(model.sk11.parameters())
        b.append(model.sk21.parameters())
        b.append(model.sk31.parameters())
        b.append(model.sk41.parameters())
        b.append(model.sk51.parameters())
        b.append(model.bn11.parameters())
        b.append(model.bn21.parameters())
        b.append(model.bn31.parameters())
        b.append(model.bn41.parameters())
        b.append(model.bn51.parameters())
    if hasattr(model, 'gated5'):
        b.append(model.gated1.parameters())
        b.append(model.gated2.parameters())
        b.append(model.gated3.parameters())
        b.append(model.gated4.parameters())
        b.append(model.gated5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def merge_params(params):
    for j in range(len(params)):
        for i in params[j]:
            yield i


def get_optimizer(optim_name, lr, parameters, weight_decay=1e-4, momentum=0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                              lr=lr, weight_decay=weight_decay,
                              momentum=momentum)
    elif optim_name == 'adam':
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'rmsprop':
        opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay=weight_decay)
    return opt


def save_checkpoint(args, encoder, decoder, enc_opt, dec_opt, epoch=None):
    if epoch is not None:
        torch.save(encoder.state_dict(), os.path.join('../results', args.model_name, 'encoder_{:}.pt'.format(epoch)))
        torch.save(decoder.state_dict(), os.path.join('../results', args.model_name, 'decoder_{:}.pt'.format(epoch)))
        torch.save(enc_opt.state_dict(), os.path.join('../results', args.model_name, 'enc_opt_{:}.pt'.format(epoch)))
        torch.save(dec_opt.state_dict(), os.path.join('../results', args.model_name, 'dec_opt_{:}.pt'.format(epoch)))
    else:
        torch.save(encoder.state_dict(), os.path.join('../results', args.model_name, 'encoder.pt'))
        torch.save(decoder.state_dict(), os.path.join('../results', args.model_name, 'decoder.pt'))
        torch.save(enc_opt.state_dict(), os.path.join('../results', args.model_name, 'enc_opt.pt'))
        torch.save(dec_opt.state_dict(), os.path.join('../results', args.model_name, 'dec_opt.pt'))
        # save parameters for future use
    pickle.dump(args, open(os.path.join('../results', args.model_name, 'args.pkl'), 'wb'))


def save_checkpoint_prev_mask(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(), os.path.join('../src/models', args.model_name + '_prev_mask', 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../src/models', args.model_name + '_prev_mask', 'decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join('../src/models', args.model_name + '_prev_mask', 'enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join('../src/models', args.model_name + '_prev_mask', 'dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../src/models', args.model_name + '_prev_mask', 'args.pkl'), 'wb'))


def save_checkpoint_prev_inference_mask(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(),
               os.path.join('../src/models', args.model_name + '_prev_inference_mask', 'encoder.pt'))
    torch.save(decoder.state_dict(),
               os.path.join('../src/models', args.model_name + '_prev_inference_mask', 'decoder.pt'))
    torch.save(enc_opt.state_dict(),
               os.path.join('../src/models', args.model_name + '_prev_inference_mask', 'enc_opt.pt'))
    torch.save(dec_opt.state_dict(),
               os.path.join('../src/models', args.model_name + '_prev_inference_mask', 'dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../src/models', args.model_name + '_prev_inference_mask', 'args.pkl'), 'wb'))


def load_checkpoint(model_name, use_gpu=True, epoch=3):
    print(f'load checkpoints {epoch}')
    if use_gpu:
        encoder_dict = torch.load(os.path.join('../results', model_name, f'encoder_{epoch}.pt'))
        decoder_dict = torch.load(os.path.join('../results', model_name, f'decoder_{epoch}.pt'))
        enc_opt_dict = torch.load(os.path.join('../results', model_name, f'enc_opt_{epoch}.pt'))
        dec_opt_dict = torch.load(os.path.join('../results', model_name, f'dec_opt_{epoch}.pt'))
    else:
        encoder_dict = torch.load(os.path.join('../results', model_name, f'encoder_{epoch}.pt'),
                                  map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join('../results', model_name, f'decoder_{epoch}.pt'),
                                  map_location=lambda storage, location: storage)
        enc_opt_dict = torch.load(os.path.join('../results', model_name, f'enc_opt_{epoch}.pt'),
                                  map_location=lambda storage, location: storage)
        dec_opt_dict = torch.load(os.path.join('../results', model_name, f'dec_opt_{epoch}.pt'),
                                  map_location=lambda storage, location: storage)

    # save parameters for future use
    args = pickle.load(open(os.path.join('../results', model_name, 'args.pkl'), 'rb'))

    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args


def batch_to_var(args, inputs, targets):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs, requires_grad=False)
    y_mask = Variable(targets[:, :, :-1].float(), requires_grad=False)
    sw_mask = Variable(targets[:, :, -1], requires_grad=False)

    if args.use_gpu:
        return x.cuda(), y_mask.cuda(), sw_mask.cuda()
    else:
        return x, y_mask, sw_mask


def make_boundaries(label, thickness=None):
    """
    Input is an image label, output is a numpy array mask encoding the boundaries of the objects
    Extract pixels at the true boundary by dilation - erosion of label.
    Don't just pick the void label as it is not exclusive to the boundaries.
    """
    assert (thickness is not None)
    import skimage.morphology as skm
    void = 255
    mask = np.logical_and(label > 0.5, label != void)[0]
    selem = skm.disk(thickness)
    boundaries = np.logical_xor(skm.dilation(mask, selem),
                                skm.erosion(mask, selem))
    return boundaries


def batch_to_var_vi(args, inputs, inputs1, targets, input_org=None):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    if input_org is not None:
        x_org = Variable(input_org, requires_grad=False)
    x = Variable(inputs, requires_grad=False)
    x1 = Variable(inputs1, requires_grad=False)
    #y_mask = Variable(targets[:,:,:-1].float(),requires_grad=False)
    y_mask = Variable(targets[:, :, :].float(), requires_grad=False)
    y_mask1 = Variable(targets[:, :, :].float(), requires_grad=False)
    sw_mask = Variable(targets[:, :, -1], requires_grad=False)

    if input_org is not None:
        if args.use_gpu:
            return x.cuda(), x1.cuda(), x_org.cuda(), y_mask.cuda(), y_mask1.cuda(), sw_mask.cuda()
        else:
            return x, x1, x_org, y_mask, y_mask1, sw_mask
    else:
        if args.use_gpu:
            return x.cuda(), x1.cuda(), y_mask.cuda(), y_mask1.cuda(), sw_mask.cuda()
        else:
            return x, x1, y_mask, y_mask1, sw_mask


def batch_to_var_test(args, inputs):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs, requires_grad=False)

    if args.use_gpu:
        return x.cuda()
    else:
        return x


def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101':
        skip_dims_in = [2048, 1024, 512, 256, 64]
    elif model_name == 'resnet34':
        skip_dims_in = [512, 256, 128, 64, 64]
    elif 'vgg16' in model_name:
        skip_dims_in = [512, 512, 256, 128, 64]
    elif model_name == 'unet':
        skip_dims_in = [512, 512, 256, 128, 64]
    return skip_dims_in


def init_visdom(args, viz):
    # initialize visdom figures

    lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 4)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Training Losses',
            legend=['iou', 'total']
        )
    )

    elot = {}
    # epoch losses

    elot['iou'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='sIoU Loss',
            legend=['train', 'val']
        )
    )

    elot['total'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Total Loss',
            legend=['train', 'val']
        )
    )

    mviz_pred = {}
    for i in range(args.maxseqlen):
        mviz_pred[i] = viz.heatmap(X=np.zeros((args.imsize, args.imsize)),
                                   opts=dict(title='Pred mask t'))

    mviz_true = {}
    for i in range(args.maxseqlen):
        mviz_true[i] = viz.heatmap(X=np.zeros((args.imsize, args.imsize)),
                                   opts=dict(title='True mask t'))

    image_lot = viz.image(np.ones((3, args.imsize, args.imsize)),
                          opts=dict(title='image'))

    return lot, elot, mviz_pred, mviz_true, image_lot


def outs_perms_to_cpu(args, outs, true_perm, h, w):
    # ugly function that turns contents of torch variables to numpy
    # (used for display during training)

    out_masks = outs
    y_mask_perm = true_perm[0]

    y_mask_perm = y_mask_perm.view(y_mask_perm.size(0), y_mask_perm.size(1), h, w)
    out_masks = out_masks.view(out_masks.size(0), out_masks.size(1), h, w)
    out_masks = out_masks.view(out_masks.size(0), out_masks.size(1), h, w)

    out_masks = out_masks.cpu().numpy()
    y_mask_perm = y_mask_perm.cpu().numpy()

    return out_masks, y_mask_perm
