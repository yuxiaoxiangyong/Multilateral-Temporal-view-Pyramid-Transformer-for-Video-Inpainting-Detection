import torch
import os
import numpy as np
import torch.utils.data as data
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from models.encoder.encoder import Encoder
from models.decoder.decoder import Decoder
from utils.dataset_utils import get_dataset
from args import get_parser
from utils.utils import make_dir, load_checkpoint

from torchvision import transforms
matplotlib.use('Agg')

class Evaluate():
    def __init__(self,args):
        self.split = args.eval_split
        self.dataset = args.dataset
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.4776, 0.479, 0.4465],
                                         std=[0.230, 0.2085, 0.2324])
        image_transforms = transforms.Compose([to_tensor, normalize])
        if args.dataset == 'davis':
            dataset = get_dataset(args,
                                split=self.split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=False,
                                inputRes = (224,224))
        elif args.dataset == 'youtubevos': 
            dataset = get_dataset(args,
                                split=self.split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=False,
                                inputRes = (224,224))

        self.loader = data.DataLoader(dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)

        self.args = args

        print(args.model_name)
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.model_name, args.use_gpu, args.test_epoch)
        load_args.use_gpu = args.use_gpu
        self.encoder = Encoder()
        self.decoder = Decoder()
        print(load_args)

        if args.ngpus > 1 and args.use_gpu:
            self.decoder = torch.nn.DataParallel(self.decoder,device_ids=range(args.ngpus))
            self.encoder = torch.nn.DataParallel(self.encoder,device_ids=range(args.ngpus))

        self.encoder.load_state_dict(encoder_dict)
        self.decoder.load_state_dict(decoder_dict)

        if args.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()

        if load_args.length_clip == 1:
            self.video_mode = False
            print('video mode not activated')
        else:
            self.video_mode = True
            print('video mode activated')
 
    def run_eval(self):
        print ("Dataset is %s"%(self.dataset))
        print ("Split is %s"%(self.split))
        if args.dataset == 'youtubevos':
            masks_sep_dir = os.path.join('../results', args.model_name, 'masks_'+cfg.PATH.SEQUENCES.split('/')[-2])
        if args.dataset == 'davis':
            masks_sep_dir = os.path.join('../results', args.model_name, 'masks_'+cfg.PATH.SEQUENCES.split('/')[-3])
        make_dir(masks_sep_dir)
        print(len(self.loader))
        for _, (inputs, _, seq_name, starting_frame) in enumerate(self.loader): 
            base_dir_masks_sep = masks_sep_dir + '/' + seq_name[0] + '/'
            make_dir(base_dir_masks_sep)
        
            x = inputs.cuda()
            print(seq_name[0] + '/' + '%05d' % starting_frame)
            
            # multiple view
            feats, ffinfo, dct = self.encoder(x)
            out_mask, _ = self.decoder(feats, ffinfo, dct)

            # baseline
            #feats = self.encoder(x)
            #out_mask = self.decoder(feats)
            outs = torch.sigmoid(out_mask)

            x_tmp = x.data.cpu().numpy()
            height = x_tmp.shape[-2]
            width = x_tmp.shape[-1]

            mask_pred = (torch.squeeze(outs[0, 0, :])).detach().cpu().numpy()
            mask_pred = np.reshape(mask_pred, (height, width))
            indxs_instance = np.where(mask_pred > 0.5)
            mask2assess = np.zeros((height, width))
            mask2assess[indxs_instance] = 255
            Image.fromarray((mask2assess).astype('uint8')).save(base_dir_masks_sep + '%04d_instance_%02d.png' % (starting_frame, 0))
                    
                    
if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    if args.dataset == 'youtubevos':
        from configs.youtube.config import cfg
    if args.dataset == 'davis':
        from configs.davis.config import cfg
    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    if not args.log_term:
        print ("Eval logs will be saved to:", os.path.join('../models',args.model_name, 'eval.log'))

    E = Evaluate(args)
    E.run_eval()
