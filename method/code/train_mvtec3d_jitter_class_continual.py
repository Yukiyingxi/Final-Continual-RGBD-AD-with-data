"""Replay-free MVTec3D continual training with jitter and class decoupling.

Stage 0 is trained from random bottleneck/decoder/classifier initialisation on
the base split. Incremental stages use the same shared model and a
single fixed ten-way classifier. Old examples or per-example features are
never stored. Compact aggregate CFNS/DSNS bases constrain only cumulative
updates of the shared anomaly branch; old classifier rows are restored after
each optimiser step while the new row and shared representation learn the
current object.

The supported protocols are 10-0, 9-1, 6-4 and 6-1x4. The 6-1x4 protocol is
the retained, experimentally verified configuration; the other protocols use
the same model and hyperparameters and are provided as reproducible settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader

from dataset_rgbd_xyz import build_category_datasets
from joint_cfns_dsns_v2 import JointCFNSDSNS
from models import vit_encoder
from models.uad_jitter_class import DinomalyRGBDJitterClass
from models.vision_transformer import Block as VitBlock, LinearAttention2
from optimizers import StableAdamW
from utils import (
    WarmupCosineScheduler,
    cal_anomaly_maps,
    compute_pro,
    get_gaussian_kernel,
    global_cosine_hm_percent,
)


CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel", "foam",
    "peach", "potato", "rope", "tire",
]
SETTING_TASKS = {
    "10-0": [CATEGORIES],
    "9-1": [CATEGORIES[:9], CATEGORIES[9:]],
    "6-4": [CATEGORIES[:6], CATEGORIES[6:]],
    "6-1x4": [CATEGORIES[:6], ["peach"], ["potato"], ["rope"], ["tire"]],
}
CLASS_TO_INDEX = {name: index for index, name in enumerate(CATEGORIES)}


def tasks_for(setting):
    """Return a private task list so callers cannot mutate the definitions."""
    return [list(task) for task in SETTING_TASKS[setting]]


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logger_for(output: Path) -> logging.Logger:
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(output))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(output / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def make_sets(args, categories, phase):
    datasets = build_category_datasets(
        args.data_path, categories, phase, args.image_size, args.crop_size
    )
    for category, dataset in zip(categories, datasets):
        dataset.class_index = CLASS_TO_INDEX[category]
    return datasets


def build_model(args, device):
    encoder = vit_encoder.load(args.backbone)
    if "base" not in args.backbone:
        raise ValueError("The audited configuration requires DINOv2 ViT-B/14")
    embed_dim, num_heads = 768, 12
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(args.dropout)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(args.dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(args.dropout),
        ),
    ])
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-8),
            attn=lambda *a, **kw: LinearAttention2(*a, eps=args.linear_attention_eps, **kw),
        )
        for _ in range(8)
    ])
    model = DinomalyRGBDJitterClass(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        embed_dim=embed_dim,
        num_classes=len(CATEGORIES),
        feature_jitter_scale=args.feature_jitter_scale,
        feature_jitter_prob=args.feature_jitter_prob,
        target_layers=list(range(2, 10)),
        fuse_layer_encoder=groups,
        fuse_layer_decoder=groups,
        add_class_token_weight=-1.0,
        norm_encoder_token=True,
        rgb_ratio=0.5,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model.init_weights()
    return model.to(device)


def count_classifier_heads(model: nn.Module) -> int:
    return sum(
        1 for name, module in model.named_modules()
        if name == "classifier" and isinstance(module, nn.Linear)
    )


def configure_base_trainability(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for root in (model.bottleneck, model.decoder, model.classifier):
        for parameter in root.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


def configure_continual_trainability(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for root in (model.bottleneck, model.decoder):
        for module in root.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad_(True)
                trainable.append(module.weight)
    model.classifier.weight.requires_grad_(True)
    trainable.append(model.classifier.weight)
    return trainable


def local_classification_loss(logits, labels, seen):
    active = torch.tensor([CLASS_TO_INDEX[name] for name in seen], device=logits.device)
    local_targets = torch.empty_like(labels)
    for local_index, global_index in enumerate(active.tolist()):
        local_targets[labels == global_index] = local_index
    if not torch.isin(labels, active).all():
        raise AssertionError("batch contains a label outside the active class set")
    return F.cross_entropy(logits.index_select(1, active), local_targets)


def compute_loss(model, image, labels, seen, iteration, args, disable_jitter=False):
    output = model(image, disable_jitter=disable_jitter)
    hard_fraction = min(0.9 * iteration / 1000.0, 0.9)
    reconstruction = global_cosine_hm_percent(
        output["en"], output["de"], p=hard_fraction, factor=0.1
    )
    classification = local_classification_loss(output["class_logits"], labels, seen)
    total = reconstruction + args.class_loss_weight * classification
    return total, reconstruction, classification, output


def build_optimizer(parameters, args, total_iters):
    optimizer = StableAdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=False,
        eps=1e-10,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        final_ratio=0.1,
        total_epochs=max(1, total_iters),
        warmup_epochs=min(100, max(1, total_iters // 10)),
    )
    return optimizer, scheduler


def train_stage(model, parameters, bank, datasets, current, seen, stage, total_iters, args, logger):
    generator = torch.Generator().manual_seed(args.seed + stage * 997)
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    optimizer, scheduler = build_optimizer(parameters, args, total_iters)
    classifier_anchor = model.classifier.weight.detach().clone()
    allowed_rows = torch.zeros(len(CATEGORIES), dtype=torch.bool, device=args.device)
    allowed_rows[[CLASS_TO_INDEX[name] for name in current]] = True
    if stage > 0:
        if bank is None:
            raise AssertionError("incremental stage requires a populated joint null-space bank")
        bank.set_stage_anchor()

    iteration = epoch = 0
    maxima = {"cfns_left": 0.0, "cfns_right": 0.0, "dsns_block": 0.0}
    while iteration < total_iters:
        model.train()
        for image, _, _, _, labels in loader:
            image = image.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            loss, reconstruction, classification, output = compute_loss(
                model, image, labels, seen, iteration, args
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=0.1)
            optimizer.step()
            with torch.no_grad():
                model.classifier.weight[~allowed_rows].copy_(classifier_anchor[~allowed_rows])
            iteration += 1
            audit = iteration == 1 or iteration % args.log_every == 0 or iteration == total_iters
            diagnostics = (
                bank.hard_project_parameters(compute_diagnostics=audit)
                if stage > 0 else {}
            )
            scheduler.step()

            if audit:
                left = max((item["cfns_left_residual"] for item in diagnostics.values()), default=0.0)
                right = max((item["cfns_right_residual"] for item in diagnostics.values()), default=0.0)
                dsns = max((item["dsns_block_residual"] for item in diagnostics.values()), default=0.0)
                maxima["cfns_left"] = max(maxima["cfns_left"], left)
                maxima["cfns_right"] = max(maxima["cfns_right"], right)
                maxima["dsns_block"] = max(maxima["dsns_block"], dsns)
                ratio = float(output["jitter_rms"] / output["feature_rms"].clamp_min(1e-12))
                logger.info(
                    "stage=%d epoch=%d iter=%d/%d total=%.5f recon=%.5f cls=%.5f "
                    "jitter_ratio=%.4f cfns_l=%.2e cfns_r=%.2e dsns=%.2e",
                    stage, epoch, iteration, total_iters, loss, reconstruction,
                    classification, ratio, left, right, dsns,
                )
            if iteration >= total_iters:
                break
        epoch += 1
    return maxima


def collect_bank(model, bank, datasets, seen, args, logger):
    generator = torch.Generator().manual_seed(args.seed + 1009 * len(seen))
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=args.collect_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
    )
    model.eval()
    bank.begin_collection()
    for batch_index, (image, _, _, _, labels) in enumerate(loader):
        model.zero_grad(set_to_none=True)
        image = image.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        loss, reconstruction, classification, _ = compute_loss(
            model, image, labels, seen, 1000, args, disable_jitter=True
        )
        loss.backward()
        if batch_index == 0 or (batch_index + 1) % 4 == 0:
            logger.info(
                "collect seen=%d batch=%d/%d recon=%.5f cls=%.5f",
                len(seen), batch_index + 1, args.collect_batches,
                reconstruction, classification,
            )
        if batch_index + 1 >= args.collect_batches:
            break
    model.zero_grad(set_to_none=True)
    summary = bank.end_collection()
    torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def evaluate(model, datasets, categories, seen, args, stage, output_dir, logger):
    model.eval()
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(args.device)
    active = torch.tensor([CLASS_TO_INDEX[name] for name in seen], device=args.device)
    rows = []
    for category, dataset in zip(categories, datasets):
        loader = DataLoader(
            dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        gt_px, pred_px, gt_im, pred_im = [], [], [], []
        correct = total = 0
        for image, mask, label, _, class_index in loader:
            image = image.to(args.device, non_blocking=True)
            output = model(image, disable_jitter=True)
            anomaly = cal_anomaly_maps(output["en"], output["de"], image.shape[-1])[0]
            if args.metric_size:
                anomaly = F.interpolate(
                    anomaly, (args.metric_size, args.metric_size),
                    mode="bilinear", align_corners=False,
                )
                mask = F.interpolate(mask, (args.metric_size, args.metric_size), mode="nearest")
            anomaly = gaussian(anomaly)
            flat = anomaly.flatten(1)
            count = max(1, int(flat.shape[1] * args.top_ratio))
            image_score = torch.topk(flat, count, dim=1).values.mean(1)
            predicted = active[output["class_logits"].index_select(1, active).argmax(1)].cpu()
            correct += int((predicted == class_index).sum())
            total += class_index.numel()
            gt_px.append(mask[:, 0].numpy().astype(np.uint8))
            pred_px.append(anomaly[:, 0].cpu().numpy())
            gt_im.append(label.numpy())
            pred_im.append(image_score.cpu().numpy())

        true_px, score_px = np.concatenate(gt_px), np.concatenate(pred_px)
        true_im, score_im = np.concatenate(gt_im), np.concatenate(pred_im)
        row = {
            "stage": stage,
            "class": category,
            "i_auroc": roc_auc_score(true_im, score_im),
            "i_aupr": average_precision_score(true_im, score_im),
            "p_auroc": roc_auc_score(true_px.ravel(), score_px.ravel()),
            "p_aupr": average_precision_score(true_px.ravel(), score_px.ravel()),
            "aupro": compute_pro(true_px, score_px),
            "class_acc": correct / max(1, total),
        }
        rows.append(row)
        logger.info(
            "eval stage=%d %s I=%.4f P=%.4f AUPRO=%.4f cls=%.4f",
            stage, category, row["i_auroc"], row["p_auroc"], row["aupro"], row["class_acc"],
        )
    mean = {"stage": stage, "class": "Mean"}
    for key in ("i_auroc", "i_aupr", "p_auroc", "p_aupr", "aupro", "class_acc"):
        mean[key] = float(np.mean([row[key] for row in rows]))
    rows.append(mean)
    logger.info(
        "eval stage=%d Mean I=%.4f P=%.4f AUPRO=%.4f cls=%.4f",
        stage, mean["i_auroc"], mean["p_auroc"], mean["aupro"], mean["class_acc"],
    )
    with (output_dir / "metrics.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def save_checkpoint(model, bank, stage, seen, output_dir, diagnostics, setting, tasks):
    path = output_dir / f"stage_{stage}_checkpoint.pth"
    torch.save({
        "model": model.state_dict(),
        "joint_cfns_dsns": bank.state_dict(),
        "setting": setting,
        "tasks": tasks,
        "stage": stage,
        "seen": list(seen),
        "diagnostics": diagnostics,
        "classifier_heads": count_classifier_heads(model),
        "replay_samples": 0,
        "per_example_features_saved": 0,
        "feature_jitter": {
            "scale": model.feature_jitter_scale,
            "prob": model.feature_jitter_prob,
        },
        "ibfm": False,
    }, path)
    return path


def main(args):
    setup_seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = tasks_for(args.setting)
    output_dir = Path(args.output_dir)
    logger = logger_for(output_dir)
    if (output_dir / "metrics.csv").exists() and not args.resume:
        raise FileExistsError("refusing to append a fresh run to an existing metrics.csv")
    with (output_dir / "args.json").open("w") as handle:
        json.dump({
            **vars(args), "device": str(args.device), "tasks": tasks,
            "fusion": {"rgb": 0.5, "depth": 0.5},
            "replay_samples": 0, "ibfm": False,
        }, handle, indent=2)

    model = build_model(args, args.device)
    if count_classifier_heads(model) != 1:
        raise AssertionError("the protocol requires exactly one classifier head")

    seen = []
    start_stage = 0
    bank = None
    if args.resume:
        configure_continual_trainability(model)
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        bank = JointCFNSDSNS(
            model,
            cfns_energy=args.cfns_energy, cfns_rank=args.cfns_rank,
            cfns_rows=args.cfns_rows, dsns_energy=args.dsns_energy,
            dsns_rank=args.dsns_rank, dsns_rows=args.dsns_rows,
            rows_per_call=args.rows_per_call, seed=args.seed,
            exclude_prefixes=("encoder.", "classifier."),
        )
        bank.load_state_dict(state["joint_cfns_dsns"])
        seen = list(state["seen"])
        start_stage = int(state["stage"]) + 1
        expected_seen = [name for task in tasks[:start_stage] for name in task]
        if seen != expected_seen:
            raise ValueError(
                f"checkpoint seen classes {seen} do not match "
                f"setting={args.setting} prefix {expected_seen}"
            )
        saved_setting = state.get("setting")
        if saved_setting is not None and saved_setting != args.setting:
            raise ValueError(
                f"checkpoint setting={saved_setting} cannot resume as {args.setting}"
            )
        logger.info("resumed=%s next_stage=%d seen=%s", args.resume, start_stage, seen)

    all_test = make_sets(args, CATEGORIES, "test")
    for stage in range(start_stage, len(tasks)):
        current = tasks[stage]
        seen.extend(current)
        train_sets = make_sets(args, current, "train")
        total_iters = args.base_iters if stage == 0 else args.incremental_iters

        if stage == 0:
            parameters = configure_base_trainability(model)
        else:
            parameters = configure_continual_trainability(model)
        logger.info(
            "begin stage=%d current=%s seen=%s iters=%d trainable=%d jitter=%.1f@%.2f cls_w=%.3f",
            stage, current, seen, total_iters, sum(p.numel() for p in parameters),
            args.feature_jitter_scale, args.feature_jitter_prob, args.class_loss_weight,
        )
        projection_diagnostics = train_stage(
            model, parameters, bank, train_sets, current, seen,
            stage, total_iters, args, logger,
        )

        seen_test = [all_test[CATEGORIES.index(name)] for name in seen]
        rows = evaluate(model, seen_test, seen, seen, args, stage, output_dir, logger)

        configure_continual_trainability(model)
        if bank is None:
            bank = JointCFNSDSNS(
                model,
                cfns_energy=args.cfns_energy, cfns_rank=args.cfns_rank,
                cfns_rows=args.cfns_rows, dsns_energy=args.dsns_energy,
                dsns_rank=args.dsns_rank, dsns_rows=args.dsns_rows,
                rows_per_call=args.rows_per_call, seed=args.seed,
                exclude_prefixes=("encoder.", "classifier."),
            )
        summary = collect_bank(model, bank, train_sets, seen, args, logger)
        ranks = [
            value["cfns_input_rank"] + value["cfns_error_rank"]
            + value["dsns_input_rank"] + value["dsns_output_rank"]
            for value in summary.values()
        ]
        diagnostics = {
            **projection_diagnostics,
            "joint_storage_bytes": bank.storage_bytes(),
            "mean_total_rank": float(np.mean(ranks)),
            "projected_layers": bank.layer_count,
            "classifier_heads": count_classifier_heads(model),
            "replay_samples": 0,
        }
        path = save_checkpoint(
            model, bank, stage, seen, output_dir, diagnostics,
            args.setting, tasks,
        )
        logger.info("saved=%s diagnostics=%s", path, diagnostics)

        if args.stop_after_stage >= 0 and stage >= args.stop_after_stage:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting", choices=tuple(SETTING_TASKS), default="6-1x4",
        help="continual class split; 6-1x4 is the verified configuration",
    )
    parser.add_argument("--data-path", default="/root/autodl-tmp/MVTec3D-dinomaly2-processed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="dinov2reg_vit_base_14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--metric-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--collect-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-iters", type=int, default=40000)
    parser.add_argument("--incremental-iters", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--linear-attention-eps", type=float, default=1e-8)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--feature-jitter-scale", type=float, default=20.0)
    parser.add_argument("--feature-jitter-prob", type=float, default=1.0)
    parser.add_argument("--top-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cfns-energy", type=float, default=0.995)
    parser.add_argument("--cfns-rank", type=int, default=128)
    parser.add_argument("--cfns-rows", type=int, default=512)
    parser.add_argument("--dsns-energy", type=float, default=0.97)
    parser.add_argument("--dsns-rank", type=int, default=32)
    parser.add_argument("--dsns-rows", type=int, default=128)
    parser.add_argument("--rows-per-call", type=int, default=32)
    parser.add_argument("--collect-batches", type=int, default=16)
    parser.add_argument("--stop-after-stage", type=int, default=-1)
    parser.add_argument("--resume", default="")
    main(parser.parse_args())

