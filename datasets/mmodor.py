import os
from pathlib import Path

from .coco import CocoDetection, get_aux_target_hacks_list, make_coco_transforms
import pandas as pd

from datasets.data_util import preparing_dataset


class MMOdorDetection(CocoDetection):

    def __init__(self, img_folder, ann_file, txt_csv, transforms, return_masks, aux_target_hacks=None):
        super(MMOdorDetection, self).__init__(img_folder, ann_file, transforms, return_masks, aux_target_hacks=None)
        # handle txt_csv
        self.txt_df = pd.read_csv(txt_csv)

    def __getitem__(self,idx):
        img, target = super().__getitem__(idx)
        # get txt from txt csv
        fn = self.coco.loadImgs(id)[0]["file_name"]
        txt = self.txt_df[self.txt_df['File Name'] == fn]["title"].values(0)

        print(txt)
        return img, txt, target


def build(image_set, args):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' )
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]
    txt_csv = args.txt_csv

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    dataset = MMOdorDetection(img_folder, ann_file, txt_csv,
                            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args),
                            return_masks=args.masks,
                            aux_target_hacks=aux_target_hacks_list,
                            )

    return dataset


