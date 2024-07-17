import torch
import numpy as np
import os.path as osp
import torch.utils.data as data
from PIL import Image
from args import get_parser
from utils.randaugment import RandAugment

parser = get_parser()
args = parser.parse_args()

if 'youtube' in args.dataset:
    from configs.youtube.config import cfg
else:
    from configs.davis.config import cfg


class UniversalDataset(data.Dataset):

    def __init__(self, args, augment=False, split='train', inputRes=None, transform=None):
        self._length_clip = args.length_clip
        self.augment = augment
        self.split = split
        self.inputRes = inputRes
        self.dataset = args.dataset
        self.transform = transform

    def __getitem__(self, index):

        if self.split == 'train' or self.split == 'val' or self.split == 'trainval':

            edict = self.get_raw_sample_clip(index)
            sequence_clip = edict['images']
            img_root_dir = cfg.PATH.SEQUENCES
            annot_root_dir = cfg.PATH.ANNOTATIONS
            seq_name = sequence_clip.name
            img_seq_dir = osp.join(img_root_dir, seq_name)
            annot_seq_dir = osp.join(annot_root_dir, seq_name)
            index_list = sequence_clip.frame_id_list  # the frames index list

            imgs, imgs1, imgs2, targets, annots = [], [], [], [], []
            origin_imgs, origin_imgs1, origin_imgs2, origin_annots = [], [], [], []

            # The frames of the given sequence
            if type(sequence_clip._files[0]) == str:
                images = [f for f in sequence_clip._files]
            else:
                images = [str(f.decode()) for f in sequence_clip._files]
            starting_frame = int((osp.splitext(osp.basename(images[index_list[int(self._length_clip / 2)]]))[0]))

            # obtain the clips by the index list of the frame in the video clip
            for frame_idx in index_list:

                frame_idx = int((osp.splitext(osp.basename(images[frame_idx]))[0]))
                frame_img = osp.join(img_seq_dir, '%05d.png' % frame_idx) if 'davis' in self.dataset else osp.join(
                    img_seq_dir, '%05d.jpg' % frame_idx)
                img = Image.open(frame_img)
                img1 = Image.open(frame_img.replace(cfg.PATH.SEQUENCES, cfg.PATH.SEQUENCES2))
                img2 = Image.open(frame_img.replace(cfg.PATH.SEQUENCES,
                                                    cfg.PATH.SEQUENCES3)) if cfg.PATH.SEQUENCES3 is not None else None
                # print(frame_img)
                # print(frame_img.replace(cfg.PATH.SEQUENCES, cfg.PATH.SEQUENCES2))
                # print(frame_img.replace(cfg.PATH.SEQUENCES, cfg.PATH.SEQUENCES3))
                frame_annot = osp.join(annot_seq_dir, '%05d.png' % frame_idx)
                annot = Image.open(frame_annot).convert('L')

                # saved to be randaugment
                origin_imgs.append(img.resize(self.inputRes))
                origin_imgs1.append(img1.resize(self.inputRes))
                if cfg.PATH.SEQUENCES3 is not None:
                    origin_imgs2.append(img2.resize(self.inputRes))
                origin_annots.append(annot.resize(self.inputRes))

                # Resize
                if self.inputRes is not None:
                    img = np.array(img.resize(self.inputRes))
                    img1 = np.array(img1.resize(self.inputRes))
                    if cfg.PATH.SEQUENCES3 is not None:
                        img2 = np.array(img2.resize(self.inputRes))
                    annot = np.array(annot.resize(self.inputRes))

                if self.transform is not None:
                    # involves transform from PIL to tensor and mean and std normalization
                    img = self.transform(img)
                    img1 = self.transform(img1)
                    if cfg.PATH.SEQUENCES3 is not None:
                        img2 = self.transform(img2)
                annot = np.expand_dims(annot, axis=0)
                annot = torch.from_numpy(annot).float()
                annots.append(annot)
                annot = annot.numpy().squeeze()
                target = self.sequence_from_masks(seq_name, annot)

                # save the not aug but tansformed imgs and annots
                imgs.append(img)
                imgs1.append(img1)
                if cfg.PATH.SEQUENCES3 is not None:
                    imgs2.append(img2)
                targets.append(target)

                # only train phase excute data augumentation
            if self.augment:
                davis_randaugment = RandAugment(n=1, m=7)
                wait_aug_imgs = [img for img in origin_imgs + origin_imgs1 + origin_imgs2]
                wait_aug_gt = origin_annots[int(self._length_clip / 2)]
                iimg, mmask = davis_randaugment(wait_aug_imgs, wait_aug_gt)
                res_img = [self.transform(np.asarray(img.resize(self.inputRes))) for img in iimg]
                res_mask = torch.from_numpy(np.asarray(mmask.resize(self.inputRes)).copy())
                if cfg.PATH.SEQUENCES3 is None:
                    return torch.stack(res_img[:self._length_clip], 0), torch.stack(res_img[self._length_clip:],
                                                                                    0), self.sequence_from_masks(
                        seq_name, res_mask)
                else:
                    return torch.stack(res_img[:self._length_clip], 0), torch.stack(
                        res_img[self._length_clip:2 * self._length_clip], 0), torch.stack(
                        res_img[2 * self._length_clip:], 0), self.sequence_from_masks(seq_name, res_mask)

            if self.split in ["train", "val"]:
                if cfg.PATH.SEQUENCES3 is not None:
                    return torch.stack(imgs, 0), torch.stack(imgs1, 0), torch.stack(imgs2, 0), targets[
                        int(self._length_clip / 2)]
                else:
                    return torch.stack(imgs, 0), torch.stack(imgs1, 0), targets[int(self._length_clip / 2)]

            # test
            return torch.stack(imgs, 0), targets[1], seq_name, starting_frame

        else:
            raise ValueError("Please input the correct dataset name!")

    def __len__(self):
        return len(self.sequence_clips)

    def get_sample_list(self):
        return self.sequence_clips

    def sequence_from_masks(self, seq_name, annot):
        annot[(annot / np.max([annot.max(), 1e-8]) > 0)] = 1
        annot = np.expand_dims(annot.reshape(-1), axis=0)
        return torch.Tensor(annot.astype(np.float32))

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:label
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, image1, mask
        images, images1, masks = zip(*batch)

        # Stack the images & images1 and masks
        images = torch.stack(images)
        images1 = torch.stack(images1)
        images = torch.cat([images, images1], dim=0)
        masks = torch.stack(masks, 0)
        masks = torch.cat([masks, masks], 0)

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['mask'] = masks
        return data_dict

    @staticmethod
    def collate_fn_triple(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:label
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, image1, mask
        images, images1, images2, masks = zip(*batch)

        # Stack the images & images1 and masks
        images = torch.stack(images)
        images1 = torch.stack(images1)
        images2 = torch.stack(images2)
        images = torch.cat([images, images1, images2], dim=0)
        masks = torch.stack(masks, 0)
        masks = torch.cat([masks, masks, masks], 0)

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['mask'] = masks
        return data_dict