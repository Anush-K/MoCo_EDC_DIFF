"""
runners_edc/edc_aptos_moco.py — EDC + MoCo encoder for APTOS

Path 1 (frozen encoder):
  python runners_edc/edc_aptos_moco.py --freeze_encoder True  --save_name edc_aptos_moco_frozen

Path 2 (unfrozen encoder):
  python runners_edc/edc_aptos_moco.py --freeze_encoder False --save_name edc_aptos_moco_unfrozen

Diffusion: runner.diffusion is NEVER set → stays None → zero diffusion interference.
"""

# ── Fix Python path so all EDC-master imports resolve correctly ───────────────
import sys
import os
# This file lives at EDC-master/runners_edc/edc_aptos_moco.py
# We need EDC-master/ on the path so imports like "from models.edc import R50_R50" work
_EDCMASTER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDCMASTER not in sys.path:
    sys.path.insert(0, _EDCMASTER)
# ─────────────────────────────────────────────────────────────────────────────

import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC
from sklearn.metrics import confusion_matrix, classification_report

from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50
from configs.config_aptos import DATASET_DIR, MOCO_WEIGHTS, SAVED_MODELS_DIR
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def main_worker(gpu, args):

    args.gpu = gpu
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # ── Paths & Logger ────────────────────────────────────────────────────────
    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.warning(f"GPU: {args.gpu}")
    logger.info(f"MoCo weights   : {args.moco_weights}")
    logger.info(f"Freeze encoder : {args.freeze_encoder}")
    logger.info(f"Data dir       : {args.data_dir}")

    # ── Verify paths exist before starting ───────────────────────────────────
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset not found: {args.data_dir}")
    if args.moco_weights and not os.path.exists(args.moco_weights):
        raise FileNotFoundError(f"MoCo weights not found: {args.moco_weights}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_dset = AD_Dataset(name=args.dataset, train=True,  data_dir=args.data_dir).get_dset()
    eval_dset  = AD_Dataset(name=args.dataset, train=False, data_dir=args.data_dir).get_dset()

    print(f"TrainSet: {len(train_dset)} images")
    print(f"EvalSet : {len(eval_dset)} images")

    train_counts = Counter(np.array(train_dset.targets).astype(int))
    eval_counts  = Counter(np.array(eval_dset.targets).astype(int))
    print(f"Train → Normal: {train_counts[0]}  Abnormal: {train_counts[1]}")
    print(f"Eval  → Normal: {eval_counts[0]}   Abnormal: {eval_counts[1]}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    generator_lb = torch.Generator()
    generator_lb.manual_seed(args.seed)

    loader_dict = {}
    loader_dict["train"] = get_data_loader(
        train_dset,
        args.batch_size,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers,
        distributed=False,
        generator=generator_lb,
    )
    loader_dict["eval"] = get_data_loader(
        eval_dset,
        args.eval_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # train_encoder=False when frozen (keeps BN in eval, detaches gradients)
    train_enc = not args.freeze_encoder

    model = R50_R50(
        img_size=args.img_size,
        train_encoder=train_enc,
        stop_grad=True,
        reshape=True,
        bn_pretrain=False,
        moco_weights=args.moco_weights,
        freeze_encoder=args.freeze_encoder,
    )

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    # ── Runner — diffusion NEVER set, stays None ──────────────────────────────
    runner = EDC(
        model=model,
        num_eval_iter=args.num_eval_iter,
        tb_log=None,
        logger=logger
    )
    assert runner.diffusion is None, "Diffusion must be None!"
    logger.info("✅ Diffusion inactive (runner.diffusion = None).")
    logger.info(f"Trainable params: {count_parameters(runner.model):,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # lr_encoder=0 for frozen path (double safety: params already require_grad=False)
    lr_encoder = 0.0 if args.freeze_encoder else args.lr_encoder

    optimizer = get_optimizer_v2(
        runner.model,
        args.optim,
        args.lr,
        args.momentum,
        lr_encoder=lr_encoder,
        weight_decay=args.weight_decay,
    )
    scheduler = get_multistep_schedule_with_warmup(
        optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0
    )
    runner.set_optimizer(optimizer, scheduler)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"GPU: {torch.cuda.get_device_name(int(args.gpu))}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    runner.model = runner.model.to(device)
    args.device  = device

    runner.set_data_loader(loader_dict)

    if args.resume:
        runner.load_model(args.load_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)
    eval_dict = runner.train(args, device=device, logger=logger)
    logging.warning("Training and Evaluation COMPLETED!")

    # ── Final Metrics ─────────────────────────────────────────────────────────
    encoder_mode = "FROZEN" if args.freeze_encoder else "UNFROZEN"
    header = f"APTOS — MoCo {encoder_mode} ENCODER"

    metrics_table = pd.DataFrame({
        "Metric": ["AUC", "F1-score", "Accuracy", "Recall (Sensitivity)", "Specificity"],
        "Value":  [
            eval_dict["eval/AUC"],
            eval_dict["eval/f1"],
            eval_dict["eval/acc"],
            eval_dict["eval/recall"],
            eval_dict["eval/specificity"],
        ]
    })

    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION METRICS — {header}")
    print(f"{'='*60}\n")
    print(metrics_table.to_string(index=False, float_format="%.4f"))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    y_true  = np.array(eval_dict["eval/y_true"])
    y_score = np.array(eval_dict["eval/y_score"])
    thr     = eval_dict["eval/best_thr"]
    y_pred  = (y_score >= thr).astype(int)

    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=["Actual NORMAL", "Actual ABNORMAL"],
                         columns=["Predicted NORMAL", "Predicted ABNORMAL"])

    print(f"\n{'='*60}")
    print("  CONFUSION MATRIX")
    print(f"{'='*60}\n")
    print(cm_df)

    print(f"\n{'='*60}")
    print("  CLASSIFICATION REPORT")
    print(f"{'='*60}\n")
    print(classification_report(y_true, y_pred,
                                target_names=["NORMAL", "ABNORMAL"], digits=4))
    print(f"Best Threshold (F1-optimized): {thr:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(save_path, "final_metrics.csv")
    metrics_table.to_csv(csv_path, index=False)
    print(f"\nMetrics saved → {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EDC + MoCo — APTOS')

    # Saving
    parser.add_argument('--save_dir',        type=str,      default=SAVED_MODELS_DIR)
    parser.add_argument('--save_name',       type=str,      default='edc_aptos_moco_frozen')
    parser.add_argument('--resume',          action='store_true', default=False)
    parser.add_argument('--load_path',       type=str,      default=None)
    parser.add_argument('--overwrite',       action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', default=True)

    # MoCo
    parser.add_argument('--moco_weights',   type=str,      default=MOCO_WEIGHTS,
                        help='Path to MoCo .pth file (auto-set from config)')
    parser.add_argument('--freeze_encoder', type=str2bool, default=True,
                        help='True=frozen (Path 1), False=unfrozen (Path 2)')

    # Training
    parser.add_argument('--epoch',           type=int,   default=1)
    parser.add_argument('--num_train_iter',  type=int,   default=1000)
    parser.add_argument('--num_eval_iter',   type=int,   default=250)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--eval_batch_size', type=int,   default=64)

    # Optimizer
    parser.add_argument('--optim',        type=str,      default='AdamW')
    parser.add_argument('--lr',           type=float,    default=5e-4)
    parser.add_argument('--lr_encoder',   type=float,    default=1e-5)
    parser.add_argument('--momentum',     type=float,    default=0.9)
    parser.add_argument('--weight_decay', type=float,    default=1e-4)
    parser.add_argument('--amp',          type=str2bool, default=False)
    parser.add_argument('--clip',         type=float,    default=1)

    # Data
    parser.add_argument('--data_dir',      type=str, default=DATASET_DIR)
    parser.add_argument('--dataset',       type=str, default='aptos')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--img_size',      type=int, default=256)
    parser.add_argument('--num_workers',   type=int, default=4)

    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu',  type=str, default='0')
    parser.add_argument('--c',    type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    # Clean old save dir if overwriting
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and not args.resume:
        import shutil
        shutil.rmtree(save_path)

    main_worker(int(args.gpu), args)