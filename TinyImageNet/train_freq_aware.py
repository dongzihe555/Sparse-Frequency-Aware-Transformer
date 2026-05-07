
import argparse
import time
import yaml
import os
import logging
import json
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from loader import create_loader
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

import model
import model_sparse
import model_frequency_aware

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training (Frequency-Aware SWformer)')

parser.add_argument('--model', default='swformer_freq_aware', type=str, metavar='MODEL',
                    help='Model: swformer, swformer_sparse, swformer_freq_aware (default: swformer_freq_aware)')
parser.add_argument('-T', '--time-step', type=int, default=4, metavar='time',
                    help='Simulation time step (default: 4)')
parser.add_argument('-L', '--layer', type=int, default=4, metavar='layer',
                    help='Model layers (default: 4)')
parser.add_argument('--flblocks', type=int, default=4, metavar='N',
                    help='Frequency learner splitting blocks (default: 4)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N')
parser.add_argument('--img-size', type=int, default=None, metavar='N')
parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N N N')
parser.add_argument('--dim', type=int, default=None, metavar='N')
parser.add_argument('--patch-size', type=int, default=None, metavar='N')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N')

parser.add_argument('--vth-low-l1', type=float, default=0.5,
                    help='Level 1 threshold for low-frequency band')
parser.add_argument('--vth-high-l1', type=float, default=0.6,
                    help='Level 1 threshold for high-frequency band')
parser.add_argument('--vth-ll', type=float, default=0.5,
                    help='Level 2 threshold for LL subband')
parser.add_argument('--vth-hl', type=float, default=0.6,
                    help='Level 2 threshold for HL subband')
parser.add_argument('--vth-lh', type=float, default=0.6,
                    help='Level 2 threshold for LH subband')
parser.add_argument('--vth-hh', type=float, default=0.7,
                    help='Level 2 threshold for HH subband')

parser.add_argument('--l2-sparsity-mode', type=str, default='channel',
                    choices=['channel', 'coeff4', 'ch_individual', 'ener_ch'],
                    help='Level 2 sparsity mode: channel (original gate-based), '
                         'coeff4 (4-subband coefficient), ch_individual (channel-wise threshold), '
                         'ener_ch (energy-driven channel sparsity)')
parser.add_argument('--l2-tau-ll', type=float, default=0.01,
                    help='Level 2 initial threshold for LL subband (default: 0.01)')
parser.add_argument('--l2-tau-hl', type=float, default=0.02,
                    help='Level 2 initial threshold for HL subband (default: 0.02)')
parser.add_argument('--l2-tau-lh', type=float, default=0.02,
                    help='Level 2 initial threshold for LH subband (default: 0.02)')
parser.add_argument('--l2-tau-hh', type=float, default=0.05,
                    help='Level 2 initial threshold for HH subband (default: 0.05)')

parser.add_argument('--ener-gate-low', type=float, default=0.4,
                    help='Energy-driven gate low center (default: 0.4)')
parser.add_argument('--ener-gate-high', type=float, default=0.6,
                    help='Energy-driven gate high center (default: 0.6)')
parser.add_argument('--ener-sigma', type=float, default=0.2,
                    help='Energy-driven sampling sigma (default: 0.2)')
parser.add_argument('--ener-tau-E', type=float, default=1.0,
                    help='Energy selection temperature (default: 1.0)')

parser.add_argument('--use-freq-compensatory-mlp', action='store_true', default=False,
                    help='Enable frequency compensatory MLP')
parser.add_argument('--skip-base-mlp', action='store_true', default=False,
                    help='Skip base MLP path for lightweight MLP (default: False)')

parser.add_argument('--freq-experts-ratio', type=float, default=0.5,
                    help='Ratio for frequency experts hidden dim (default: 0.5)')

parser.add_argument('-data-dir', metavar='DIR', default="", help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='torch/cifar10')
parser.add_argument('--train-split', metavar='NAME', default='train')
parser.add_argument('--val-split', metavar='NAME', default='validation')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--no-resume-opt', action='store_true', default=False)
parser.add_argument('--gp', default=None, type=str, metavar='POOL')
parser.add_argument('--crop-pct', default=None, type=float)
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N')
parser.add_argument('-vb', '--val-batch-size', type=int, default=16, metavar='N')

parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--weight-decay', type=float, default=0.06)
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
parser.add_argument('--clip-mode', type=str, default='norm')

parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
parser.add_argument('--lr', type=float, default=1.5e-3, metavar='LR')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR')
parser.add_argument('--epochs', type=int, default=400, metavar='N')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N')
parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')

parser.add_argument('--no-aug', action='store_true', default=False)
parser.add_argument('--scale', type=float, nargs='+', default=[1.0, 1.0], metavar='PCT')
parser.add_argument('--ratio', type=float, nargs='+', default=[1.0, 1.0], metavar='RATIO')
parser.add_argument('--hflip', type=float, default=0.5)
parser.add_argument('--vflip', type=float, default=0.)
parser.add_argument('--color-jitter', type=float, default=0., metavar='PCT')
parser.add_argument('--aa', type=str, default='rand-m9-n1-mstd0.4-inc1', metavar='NAME')
parser.add_argument('--aug-splits', type=int, default=0)
parser.add_argument('--jsd', action='store_true', default=False)
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT')
parser.add_argument('--remode', type=str, default='const')
parser.add_argument('--recount', type=int, default=1)
parser.add_argument('--resplit', action='store_true', default=False)
parser.add_argument('--mixup', type=float, default=0.5)
parser.add_argument('--cutmix', type=float, default=0.0)
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None)
parser.add_argument('--mixup-prob', type=float, default=1.0)
parser.add_argument('--mixup-switch-prob', type=float, default=0.5)
parser.add_argument('--mixup-mode', type=str, default='batch')
parser.add_argument('--mixup-off-epoch', default=200, type=int, metavar='N')
parser.add_argument('--smoothing', type=float, default=0.1)
parser.add_argument('--train-interpolation', type=str, default='bicubic')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT')

parser.add_argument('--bn-tf', action='store_true', default=False)
parser.add_argument('--bn-momentum', type=float, default=None)
parser.add_argument('--bn-eps', type=float, default=None)
parser.add_argument('--sync-bn', action='store_true')
parser.add_argument('--dist-bn', type=str, default='')
parser.add_argument('--split-bn', action='store_true')

parser.add_argument('--model-ema', action='store_true', default=False)
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False)
parser.add_argument('--model-ema-decay', type=float, default=0.9998)

parser.add_argument('--seed', type=int, default=42, metavar='S')
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N')
parser.add_argument('--checkpoint-hist', type=int, default=1, metavar='N')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N')
parser.add_argument('--save-images', action='store_true', default=False)
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--apex-amp', action='store_true', default=False)
parser.add_argument('--native-amp', action='store_true', default=False)
parser.add_argument('--channels-last', action='store_true', default=False)
parser.add_argument('--pin-mem', action='store_true', default=False)
parser.add_argument('--no-prefetcher', action='store_true', default=False)
parser.add_argument('--output', default='', type=str, metavar='PATH')
parser.add_argument('--experiment', default='', type=str, metavar='NAME')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC')
parser.add_argument('--tta', type=int, default=0, metavar='N')
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False)
parser.add_argument('--torchscript', dest='torchscript', action='store_true')
parser.add_argument('--log-wandb', action='store_true', default=False)

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

max_accuracy = 0.0

def main():
    setup_default_logging()
    args, args_text = _parse_args()
    
    if args.log_wandb:
        if has_wandb:
            wandb.init(project="swformer_freq_aware",
                      name=args.experiment,
                      config=args)
        else:
            _logger.warning("wandb not installed.")
    
    args.prefetcher = not args.no_prefetcher
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.distributed = False
    
    use_amp = None
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    
    random_seed(args.seed, int(args.rank))
    
    print(f"Creating model: {args.model}")
    print(f"\n{'='*60}")
    print("Feature Configuration:")
    print('='*60)
    
    if args.model == 'swformer_freq_aware':
        print(f"  use_freq_compensatory_mlp: {'✓ ENABLED' if args.use_freq_compensatory_mlp else '✗ DISABLED'}")

        print(f"  skip_base_mlp: {'✓ ENABLED' if args.skip_base_mlp else '✗ DISABLED'}")
        print(f"  l2_sparsity_mode: {args.l2_sparsity_mode}")
        print(f"  freq_experts_ratio: {args.freq_experts_ratio}")
        print('='*60 + '\n')
        
        model = create_model(
            'swformer_freq_aware',
            img_size_h=args.img_size, img_size_w=args.img_size, patch_size=args.patch_size,
            in_channels=3, num_classes=args.num_classes, embed_dims=args.dim, drop_rate=0.,
            FL_blocks=args.flblocks, mlp_ratios=args.mlp_ratio, depths=args.layer, T=args.time_step,

            vth_low_l1=args.vth_low_l1, vth_high_l1=args.vth_high_l1,
            vth_ll=args.vth_ll, vth_hl=args.vth_hl, vth_lh=args.vth_lh, vth_hh=args.vth_hh,

            l2_sparsity_mode=args.l2_sparsity_mode,
            l2_tau_ll=args.l2_tau_ll, l2_tau_hl=args.l2_tau_hl,
            l2_tau_lh=args.l2_tau_lh, l2_tau_hh=args.l2_tau_hh,

            ener_gate_low=args.ener_gate_low, ener_gate_high=args.ener_gate_high,
            ener_sigma=args.ener_sigma, ener_tau_E=args.ener_tau_E,

            use_freq_compensatory_mlp=args.use_freq_compensatory_mlp,
            skip_base_mlp=args.skip_base_mlp,
            freq_experts_ratio=args.freq_experts_ratio,
        )
    elif args.model == 'swformer_sparse':
        model = create_model(
            'swformer_sparse',
            img_size_h=args.img_size, img_size_w=args.img_size, patch_size=args.patch_size,
            in_channels=3, num_classes=args.num_classes, embed_dims=args.dim, drop_rate=0.,
            FL_blocks=args.flblocks, mlp_ratios=args.mlp_ratio, depths=args.layer, T=args.time_step,
            vth_low_l1=args.vth_low_l1, vth_high_l1=args.vth_high_l1,
            vth_ll=args.vth_ll, vth_hl=args.vth_hl, vth_lh=args.vth_lh, vth_hh=args.vth_hh,

            l2_sparsity_mode=args.l2_sparsity_mode,
            l2_tau_ll=args.l2_tau_ll, l2_tau_hl=args.l2_tau_hl,
            l2_tau_lh=args.l2_tau_lh, l2_tau_hh=args.l2_tau_hh,

            ener_gate_low=args.ener_gate_low, ener_gate_high=args.ener_gate_high,
            ener_sigma=args.ener_sigma, ener_tau_E=args.ener_tau_E,
        )
    else:
        model = create_model(
            'swformer',
            img_size_h=args.img_size, img_size_w=args.img_size, patch_size=args.patch_size,
            in_channels=3, num_classes=args.num_classes, embed_dims=args.dim, drop_rate=0.,
            FL_blocks=args.flblocks, mlp_ratios=args.mlp_ratio, depths=args.layer, T=args.time_step,
        )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters / 1e6:.2f}M")
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes')
        args.num_classes = model.num_classes
    
    if args.local_rank == 0:
        _logger.info(f'Model {safe_model_name(args.model)} created, param count: {sum([m.numel() for m in model.parameters()])}')
    
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1
        num_aug_splits = args.aug_splits
    
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    if args.torchscript:
        assert not use_amp == 'apex'
        assert not args.sync_bn
        model = torch.jit.script(model)
    
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    
    amp_autocast = suppress
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP.')
    
    resume_epoch = None
    if args.resume:

        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if args.local_rank == 0:
            if missing_keys:
                _logger.warning(f'Missing keys: {missing_keys}')
            if unexpected_keys:
                _logger.warning(f'Unexpected keys: {unexpected_keys}')
            _logger.info(f'Resumed model from {args.resume}')
        
        if not args.no_resume_opt and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            _logger.info('Resumed optimizer state')
        
        if not args.no_resume_opt and 'scaler' in checkpoint and loss_scaler is not None:
            loss_scaler.load_state_dict(checkpoint['scaler'])
            _logger.info('Resumed loss scaler state')
        
        resume_epoch = checkpoint.get('epoch', None)
        if resume_epoch is not None:
            _logger.info(f'Resumed from epoch {resume_epoch}')
        
        del checkpoint
        torch.cuda.empty_cache()
    
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
    
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    
    if args.local_rank == 0:
        _logger.info(f'Scheduled epochs: {num_epochs}')
    
    if (args.dataset == '' or args.dataset is None) and args.data_dir and 'tiny-imagenet' in args.data_dir.lower():
        from torchvision.datasets import ImageFolder
        dataset_train = ImageFolder(os.path.join(args.data_dir, 'train'))
        dataset_eval = ImageFolder(os.path.join(args.data_dir, 'val'))
    else:
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            batch_size=args.batch_size, repeats=args.epoch_repeats, download=True)
        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            batch_size=args.batch_size, download=True)
    
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)
    
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    
    loader_train = create_loader(
        dataset_train, input_size=data_config['input_size'], batch_size=args.batch_size,
        is_training=True, use_prefetcher=args.prefetcher, no_aug=args.no_aug,
        re_prob=args.reprob, re_mode=args.remode, re_count=args.recount, re_split=args.resplit,
        scale=args.scale, ratio=args.ratio, hflip=args.hflip, vflip=args.vflip,
        color_jitter=args.color_jitter, auto_augment=args.aa, num_aug_splits=num_aug_splits,
        interpolation=train_interpolation, mean=data_config['mean'], std=data_config['std'],
        num_workers=args.workers, distributed=args.distributed, collate_fn=collate_fn,
        pin_memory=args.pin_mem, use_multi_epochs_loader=args.use_multi_epochs_loader)
    
    loader_eval = create_loader(
        dataset_eval, input_size=data_config['input_size'], batch_size=args.val_batch_size,
        is_training=False, use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'], mean=data_config['mean'], std=data_config['std'],
        num_workers=args.workers, distributed=args.distributed, crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem)
    
    if args.jsd:
        assert num_aug_splits > 1
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.local_rank == 0:

        dataset_name = 'cifar10'
        if args.dataset and 'cifar100' in args.dataset.lower():
            dataset_name = 'cifar100'
        elif args.num_classes == 100:
            dataset_name = 'cifar100'
        elif (args.data_dir and 'tiny-imagenet' in args.data_dir.lower()) or args.num_classes == 200:
            dataset_name = 'tiny_imagenet'
        
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                dataset_name,
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        
        if args.output:
            output_base = args.output
        else:
            output_base = f'./output/{dataset_name}'
        
        output_dir = get_outdir(output_base, exp_name)
        
        _logger.info(f"Dataset: {dataset_name.upper()}")
        _logger.info(f"Output directory: {output_dir}")
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
            
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            
            if args.local_rank == 0:
                global max_accuracy
                max_accuracy = max(max_accuracy, eval_metrics['top1'])
                if args.log_wandb and has_wandb:
                    wandb.log({
                        "train/loss": train_metrics['loss'],
                        "val/loss": eval_metrics['loss'],
                        "val/acc_tp1": eval_metrics['top1'],
                        "val/acc_tp5": eval_metrics['top5'],
                        "val/max": max_accuracy,
                        "lr": float(optimizer.param_groups[0]['lr']),
                    }, step=epoch)
            
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics
            
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
            
            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)
                
                dataset_name = 'cifar10'
                if args.dataset and 'cifar100' in args.dataset.lower():
                    dataset_name = 'cifar100'
                elif args.num_classes == 100:
                    dataset_name = 'cifar100'
                elif (args.data_dir and 'tiny-imagenet' in args.data_dir.lower()) or args.num_classes == 200:
                    dataset_name = 'tiny_imagenet'
            
            if saver is not None:
                save_metric = eval_metrics[eval_metric]

                checkpoint_path = os.path.join(output_dir, f'checkpoint-{epoch}.pth.tar')
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    _logger.info(f'Removed existing checkpoint: {checkpoint_path}')
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                _logger.info(f'*** Best metric: {best_metric} (epoch {best_epoch})')
    
    except KeyboardInterrupt:
        pass
    
    if best_metric is not None:
        _logger.info(f'*** Best metric: {best_metric} (epoch {best_epoch})')

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args,
                    lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
                    loss_scaler=None, model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    model.train()
    
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        
        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)
        
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
        
        optimizer.zero_grad()
        
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()
        
        functional.reset_net(model)
        
        if model_ema is not None:
            model_ema.update(model)
        
        torch.cuda.synchronize()
        
        num_updates += 1
        batch_time_m.update(time.time() - end)
        
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
            
            if args.local_rank == 0:
                _logger.info(
                    f'Train: {epoch} [{batch_idx:>4d}/{len(loader)} ({100. * batch_idx / last_idx:>3.0f}%)]  '
                    f'Loss: {losses_m.val:>9.6f} ({losses_m.avg:>6.4f})  '
                    f'Time: {batch_time_m.val:.3f}s  LR: {lr:.3e}')
        
        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        
        end = time.time()
    
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    
    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    model.eval()
    
    end = time.time()
    last_idx = len(loader) - 1
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            
            with amp_autocast():
                output = model(input)
            
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            
            loss = loss_fn(output, target)
            
            functional.reset_net(model)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            
            torch.cuda.synchronize()
            
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f}  '
                    f'Loss: {losses_m.val:>7.4f}  '
                    f'Acc@1: {top1_m.val:>7.4f}  Acc@5: {top5_m.val:>7.4f}')
    
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    
    return metrics

if __name__ == '__main__':
    main()
