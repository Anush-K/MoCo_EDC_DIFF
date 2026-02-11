# import needed library
import os
import logging
import random
from unittest import runner

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils import get_logger, count_parameters, over_write_args_from_file
from train_utils import TBLog, get_optimizer_v2, get_multistep_schedule_with_warmup
from methods.edc1 import EDC, return_best_thr
import pandas as pd
import shutil
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score, recall_score, classification_report

from datasets.dataset import AD_Dataset
from datasets.data_utils import get_data_loader
from models.edc import R50_R50, WR50_WR50
import warnings
from configs.config_lungct import DATASET_DIR, TEST_DIR
from collections import Counter

from models.latent_diffusion import LatentDiffusion

warnings.filterwarnings("ignore")

class CombinedModel(torch.nn.Module):
    def __init__(self, edc_model, diffusion_model):
        super().__init__()
        self.edc = edc_model
        self.diffusion = diffusion_model


def get_label(sample):
    """
    Extract label from dataset sample.
    Adjust index depending on dataset return format.
    """
    # If dataset returns (img, label)
    if isinstance(sample, tuple) and len(sample) >= 2:
        label = sample[1]
    else:
        label = sample[-1]

    # If label is one-hot → convert to class index
    if isinstance(label, (np.ndarray, list)) and len(label) > 1:
        label = np.argmax(label)

    return int(label)


def main_worker(gpu, args):
    """ """

    args.gpu = gpu
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    # cudnn.benchmark = True

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "INFO"
    tb_log = None

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    # Construct Dataset & DataLoader
    train_dset = AD_Dataset(name=args.dataset, train=True, data_dir=args.data_dir)
    train_dset = train_dset.get_dset()
    print("TrainSet Image Number:", len(train_dset))
    eval_dset = AD_Dataset(name=args.dataset, train=False, data_dir=args.data_dir)
    eval_dset = eval_dset.get_dset()
    print("EvalSet Image Number:", len(eval_dset))

    # Access labels directly
    train_labels = np.array(train_dset.targets)
    eval_labels = np.array(eval_dset.targets)

    # Count class distribution
    train_counts = Counter(train_labels)
    eval_counts = Counter(eval_labels)

    print("=== Train Split ===")
    print(f"Total: {len(train_labels)}")
    print(f"Normal:   {train_counts.get(0, 0)}")
    print(f"Abnormal: {train_counts.get(1, 0)}")

    print("\n=== Eval/Test Split ===")
    print(f"Total: {len(eval_labels)}")
    print(f"Normal:   {eval_counts.get(0, 0)}")
    print(f"Abnormal: {eval_counts.get(1, 0)}")
    loader_dict = {}
    dset_dict = {"train": train_dset, "eval": eval_dset}

    generator_lb = torch.Generator()
    generator_lb.manual_seed(args.seed)
    loader_dict["train"] = get_data_loader(
        dset_dict["train"],
        args.batch_size,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers,
        distributed=False,
        generator=generator_lb,
    )

    loader_dict["eval"] = get_data_loader(
        dset_dict["eval"],
        args.eval_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = R50_R50(
        img_size=args.img_size,
        train_encoder=True,
        stop_grad=True,
        reshape=True,
        bn_pretrain=False,
    )

    # Add Latent Diffusion Model
    diffusion = LatentDiffusion(
        channels=1024,
        timesteps=1000
        )

    combined_model = CombinedModel(model, diffusion)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01

    runner = EDC(
        model=model, num_eval_iter=args.num_eval_iter, tb_log=tb_log, logger=logger
    )
    # Attach diffusion model to runner
    runner.diffusion = diffusion

    logger.info(f"Number of Trainable Params: {count_parameters(runner.model)}")

    # SET Optimizer & LR Scheduler
    # optimizer = get_optimizer_v2(
    #     runner.model,
    #     args.optim,
    #     args.lr,
    #     args.momentum,
    #     lr_encoder=args.lr_encoder,
    #     weight_decay=args.weight_decay,
    # )
    optimizer = get_optimizer_v2(
        combined_model,
        args.optim,
        args.lr,
        args.momentum,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
    )

    scheduler = get_multistep_schedule_with_warmup(
        optimizer, milestones=[1e10], gamma=0.2, num_warmup_steps=0
    )
    runner.set_optimizer(optimizer, scheduler)

    # ===== Device setup (Universal) =====
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS backend)")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
        logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ Using CPU (no GPU backend detected)")

    #runner.model = runner.model.to(device)
    combined_model = combined_model.to(device)

    runner.model = combined_model.edc
    runner.diffusion = combined_model.diffusion

    args.device = device  # optional, to use later inside train/eval

    # Move diffusion model to device
    #diffusion = diffusion.to(device)


    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    ## set DataLoader
    runner.set_data_loader(loader_dict)
    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        runner.load_model(args.load_path)

    # START TRAINING
    runner.tb_log = TBLog(save_path, "tb", use_tensorboard=args.use_tensorboard)
    # runner.train(args, device=device, logger=logger)
    # logging.warning(f"Training and Evaluation are COMPLETED!")
    eval_dict = runner.train(args, device=device, logger=logger)
    #best_thr = eval_dict['eval/best_thr'] 
    logging.warning(f"Training and Evaluation are COMPLETED!")

    # ------------------------------
    # 2. Testing/Evaluation phase
    # ------------------------------

    # test_dir = TEST_DIR  # full path to test folder
    # label_map = {0: "NORMAL", 1: "ABNORMAL"}  

    # --- Step 1: Collect scores + labels ---
    #all_scores, all_labels, all_paths = [], [], []
    A_final, all_labels, all_paths = [], [], []
    all_scores_edc = []
    all_scores_diff = []

    model.eval()
    # with torch.no_grad():
    #     for _, x, _, y, filenames in loader_dict["eval"]:
    #         x = x.to(device)
    #         result = model(x)

    #         scores = result["p_all"].mean(dim=(1,2,3)).cpu().numpy()
    #         labels = y.cpu().numpy()

    #         all_scores.extend(scores)
    #         all_labels.extend(labels)
    #         all_paths.extend(filenames)

    with torch.no_grad():
        for _, x, _, y, filenames in loader_dict["eval"]:
            x = x.to(device)

            # ---------- EDC anomaly score ----------
            result = model(x)
            A_edc = result["p_all"].mean(dim=(1, 2, 3))  # [B]

            # ---------- Diffusion anomaly score ----------
            e3 = model.edc_encoder(x)[2]  # e3 feature
            # A_diff = diffusion.compute_anomaly_score(e3)  # [B]
            # MODIFY
            with diffusion.ema_scope():
                A_diff = diffusion.compute_anomaly_score(e3)


            # Move to CPU
            A_edc = A_edc.cpu().numpy()
            A_diff = A_diff.cpu().numpy()
            labels = y.cpu().numpy()

            all_scores_edc.extend(A_edc)
            all_scores_diff.extend(A_diff)
            all_labels.extend(labels)
            all_paths.extend(filenames)

    # Fusion of scores
    A_edc = np.array(all_scores_edc)
    A_diff = np.array(all_scores_diff)
    # Min-Max normalization
    A_edc = (A_edc - A_edc.min()) / (A_edc.max() - A_edc.min() + 1e-8)
    A_diff = (A_diff - A_diff.min()) / (A_diff.max() - A_diff.min() + 1e-8)

    # Final fused score
    A_final = (
        args.lambda_fuse * A_edc + 
        (1.0 - args.lambda_fuse) * A_diff
    )

    # --- Step 2: Best threshold ---
    # best_thr = return_best_thr(all_labels, all_scores)
    # best_thr = return_best_thr(all_labels, A_final)
    # print(f"Best threshold (F1-optimized): {best_thr:.4f}")
    best_thr = return_best_thr(all_labels, A_final)

    # ================================
    # FINAL METRICS SUMMARY
    # ================================
    y_true = np.array(all_labels)
    y_score = A_final 
    thr = best_thr # EDC + Diffusion best threshold

    y_pred = (y_score >= thr).astype(int)

    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-8)

    metrics_table = pd.DataFrame({
    "Metric": [
        "AUC",
        "F1-score",
        "Accuracy",
        "Recall (Sensitivity)",
        "Specificity"
    ],
    "Value": [
        auc,
        f1,
        acc,
        recall,
        specificity
    ]
    })
    print("\n================ FINAL FUSED EVALUATION METRICS - LungCT ================\n")
    print(metrics_table.to_string(index=False, float_format="%.4f"))

    # ================================
    # CONFUSION MATRIX
    # ================================

    cm_df = pd.DataFrame(
        cm,
        index=["Actual NORMAL", "Actual ABNORMAL"],
        columns=["Predicted NORMAL", "Predicted ABNORMAL"]
    )

    print("\n================ CONFUSION MATRIX ================\n")
    print(cm_df)

    # ================================
    # CLASSIFICATION REPORT
    # ================================

    print("\n================ CLASSIFICATION REPORT ================\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["NORMAL", "ABNORMAL"],
            digits=4
        )
    )

    print("\nBest Threshold (F1-optimized): {:.4f}".format(thr))

    # # ======================================================================
    # # Lung_CT: SAVE CSVs + FULL MISCLASSIFICATION SUMMARY
    # # ======================================================================

    # test_dir = os.path.join(args.data_dir, "test")
    # label_map = {0: "NORMAL", 1: "ABNORMAL"}

    # mis_dir = "misclassified_lungct"
    # os.makedirs(mis_dir, exist_ok=True)

    # # Subfolders for misclassification types
    # normal_as_abnormal_dir = os.path.join(mis_dir, "Normal_as_Abnormal")
    # abnormal_as_normal_dir = os.path.join(mis_dir, "Abnormal_as_Normal")

    # os.makedirs(normal_as_abnormal_dir, exist_ok=True)
    # os.makedirs(abnormal_as_normal_dir, exist_ok=True)

    # results = []
    # misclassified = []

    # all_labels_np = np.asarray(all_labels)
    # # Counters
    # total_normal = np.sum(all_labels_np == 0)
    # total_abnormal = np.sum(all_labels_np == 1)

    # mis_normal = 0
    # mis_abnormal = 0

    # #for i, (score, gt, fname) in enumerate(zip(all_scores, all_labels, all_paths), start=1):
    # for i, (score, gt, img_path) in enumerate(zip(A_final, all_labels_np, all_paths), start=1):
    #     fname = os.path.basename(img_path)
    #     pred = int(score >= best_thr)
    #     results.append([i, fname, gt, pred])

    #     # Misclassified
    #     if pred != gt:
    #         misclassified.append([i, fname, gt, pred])

    #         # src_path = os.path.join(test_dir, label_map[gt], fname)
    #         src_path = img_path
    #         # base_fname = os.path.basename(fname)
    #         if not os.path.exists(src_path):
    #             print(f"⚠️ Missing file: {src_path}")
    #             continue


    #         # NORMAL → ABNORMAL
    #         if gt == 0 and pred == 1:
    #             mis_normal += 1
    #             dst = os.path.join(normal_as_abnormal_dir, fname)

    #         # ABNORMAL → NORMAL
    #         else:
    #             mis_abnormal += 1
    #             dst = os.path.join(abnormal_as_normal_dir, fname)

    #         shutil.copy(src_path, dst)

    # print("Confusion Matrix Lung_CT (FINAL test phase):\n", confusion_matrix(all_labels_np, (A_final >= best_thr).astype(int)))



    # # ======================================================================
    # # SAVE CSVs
    # # ======================================================================

    # pd.DataFrame(results, columns=["S.No", "Filename", "GT", "Pred"]).to_csv(
    #     "results_test_edc_diff_lungct.csv", index=False
    # )

    # pd.DataFrame(misclassified, columns=["S.No", "Filename", "GT", "Pred"]).to_csv(
    #     "misclassified_test_edc_diff_lungct.csv", index=False
    # )

    # print("\nSaved results_test_edc_diff_lungct.csv & misclassified_test_edc_diff_lungct.csv\n")

    # # ======================================================================
    # # BASIC SUMMARY
    # # ======================================================================

    # total_samples = len(results)
    # total_mis = len(misclassified)

    # print(f"Total test samples: {total_samples}")
    # print(f"Total Misclassified: {total_mis}")
    # print(f"NORMAL as ABNORMAL: {mis_normal}")
    # print(f"ABNORMAL as NORMAL: {mis_abnormal}\n")

    # # Accuracies
    # acc_normal = 1 - (mis_normal / total_normal)
    # acc_abnormal = 1 - (mis_abnormal / total_abnormal)

    # print(f"Normal class:     Total={total_normal}, Misclassified={mis_normal}, Accuracy={acc_normal:.4f}")
    # print(f"Abnormal class:   Total={total_abnormal}, Misclassified={mis_abnormal}, Accuracy={acc_abnormal:.4f}\n")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str,
                        default='edc_diff_lungct',
                        )
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

    '''  
    Training Configuration
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=1000,
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=250,
                        help='evaluation frequency')
    parser.add_argument('-bsz', '--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=0., help='ema momentum for eval_model')
    parser.add_argument('--lambda_diff', type=float, default=0.1,
                    help='Weight for diffusion loss')
    parser.add_argument('--lambda_fuse', type=float, default=0.5,
                    help='Fusion weight for EDC and diffusion anomaly scores')
    ''' 
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_encoder', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip', type=float, default=1)
    ''' 
    Data Configurations
    '''
    parser.add_argument('--data_dir', type=str, default=DATASET_DIR)
    parser.add_argument('-ds', '--dataset', type=str, default='lung_ct')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--gpu', default='1', type=str,
                        help='GPU id to use.')

    # config file
    parser.add_argument('--c', type=str, default='')
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil

        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    main_worker(int(args.gpu), args)
