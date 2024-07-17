
def get_dataset(args, split, image_transforms = None, target_transforms = None, augment = False,inputRes = None, video_mode = True, use_prev_mask = False,use_ela=False):

    assert args.dataset == 'youtubevos' or args.dataset == 'davis', "Please input the correct dataset name!!!"
    from dataloaders.universaldataloader import UniversalLoader as MyChosenDataset
    dataset = MyChosenDataset(args, split=split, transform=image_transforms, augment=augment, inputRes=inputRes)
    return dataset