from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.seaice_dataset_new import *
from geoseg.models.SeaIceNet import SeaIceNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils


# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 4
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "SeaIceNet"
weights_path = "model_weights/{}".format(weights_name)
test_weights_name = "SeaIceNet"
log_name = 'seaice/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = SeaIceNet(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader
IMG_SIZE = (1024, 1024)
train_dataset = SeaIceDataset(data_root='D:/seaice_1024/train', mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=IMG_SIZE)

val_dataset = SeaIceDataset(data_root='D:/seaice_1024/test', transform=val_aug, img_size=IMG_SIZE)
test_dataset = SeaIceDataset(data_root='D:/seaice_1024/test', transform=val_aug, img_size=IMG_SIZE)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True, persistent_workers=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

