from datetime import datetime
import os, random
import os.path as osp
import torch
import argparse
import yaml
from train_process import Trainer_prostate_uncertainty

from dataloaders.dataloader import get_dataloader
from networks.deeplabv3 import *
import torch.backends.cudnn as cudnn

local_path = osp.dirname(osp.abspath(__file__))

def main(args):
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # (1) set model
    model = DeepLab(num_classes=args.num_classes, num_domains=args.num_domains,
                    backbone=args.backbone, output_stride=args.out_stride).cuda()
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    # (2) select optimizer
    if args.optim.lower()=="adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    elif args.optim.lower()=="adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    elif args.optim.lower()=="sgd":
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    else:
        print('Optimizer {} not available.'.format(args.optim))
        raise NotImplementedError

    # (3) get dataset
    train_loader, val_loader = get_dataloader(args)

    start_epoch = 0
    start_iteration = 0
    trainer = Trainer_prostate_uncertainty.Trainer(
        args,
        cuda=cuda,
        model=model,
        lr=args.lr,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    print("Finish training !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # common args
    parser.add_argument('--model_name', type=str, default='UniAda', help='the name of training model')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--data_dir', type=str, default='../dataset', help='data root path')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--num_domains', type=int, default=5, help='number of domains')
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+')
    parser.add_argument('--image_size', type=int, default=384, help='the size of input images')
    parser.add_argument('--datasetTrain', nargs='+', type=int, default=1,
                        help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=1,
                        help='test folder id contain images ROIs to test one of [1,2,3,4]')
    # args for training
    parser.add_argument('--backbone', type=str, default='mobilenet', help='mobilenet')
    parser.add_argument('--optim', type=str, default='adamw', help='sgd, adamw')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max_epoch', type=int, default=80, help='max epoch')
    parser.add_argument('--stop_epoch', type=int, default=100, help='stop epoch')
    parser.add_argument('--interval_validate', type=int, default=1, help='interval epoch number to valide the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate', )
    parser.add_argument('--lr-decrease-rate', type=float, default=0.5, help='ratio multiplied to initial lr')
    parser.add_argument('--pretrained-model', default='path/to/pretrained/model',
                        help='pretrained model of FCN16s')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    # args for testing
    parser.add_argument('--sync-bn', type=bool, default=False, help='sync-bn in deeplabv3+')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='freeze batch normalization of deeplabv3+')
    parser.add_argument('--movingbn', type=bool, default=False,
                        help='moving batch normalization of deeplabv3+ in the test phase', )
    parser.add_argument('--local_path', type=str, default='./',
                        help='Path root for test image and mask')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha', )
    parser.add_argument('--beta', type=float, default=0.5, help='beta', )
    args = parser.parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    all_domain = [[[2, 3, 4, 5, 6], [1]],
                  [[1, 3, 4, 5, 6], [2]],
                  [[1, 2, 4, 5, 6], [3]],
                  [[1, 2, 3, 5, 6], [4]],
                  [[1, 2, 3, 4, 6], [5]],
                  [[1, 2, 3, 4, 5], [6]]]

    args.data_dir = osp.join(args.data_dir, "prostate")

    for domain in all_domain:
        args.datasetTrain = domain[0]
        args.datasetTest = domain[1]
        now = datetime.now()

        args.out = osp.join(args.local_path, 'logs/prostate/',
                            'train_d' + ''.join(map(str, args.datasetTrain)) + '_test_d' + str(args.datasetTest[0]),
                            args.model_name + "_" + args.backbone, now.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(args.out)
        main(args)

