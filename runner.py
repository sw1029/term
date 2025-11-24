"""
실험 파이프라인을 구성하는 모듈.

main.py 에서는 이 모듈의 run_experiment 만 호출하며,
실제 LLM 로딩 / SAE 학습 / GNN-XAI 분석 / 로그 및 플롯 생성은
모두 이 스크립트에서 담당한다.
"""

import os
import datetime
from typing import Any, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from SAE import H_SAE
from analysis import run_steering_analysis
from gnn_xai import FeatureGNN, build_feature_graph_from_decoder, select_topk_neighbors
from train import train_model
from utils import (
    concept_vector,
    get_activation,
    plot_feature_changes,
    save_experiment_log,
    compute_feature_stats,
    plot_feature_std_curve,
    compute_llm_hidden_stats,
    plot_llm_std_curve,
)


def build_dataloader(cfg, tokenizer) -> DataLoader:
    """
    Hydra config를 기반으로 데이터셋을 로드하고 DataLoader를 구성한다.
    """
    dataset = load_dataset(
        cfg.experiment.dataset_name,
        cfg.experiment.dataset_config,
        split="train",
    )
    return DataLoader(
        dataset,
        batch_size=cfg.sae.batch_size,
        shuffle=True,
    )


def prepare_concept_vectors(
    cfg, model, tokenizer, device, layer_idx: int
) -> torch.Tensor:
    """
    positive/negative 프롬프트를 이용해 concept vector를 계산한다.
    """
    pos = cfg.experiment.positive_prompts
    nag = cfg.experiment.negative_prompts

    p_acts = torch.stack(
        [get_activation(t, layer_idx, model, tokenizer) for t in pos]
    ).mean(dim=0)
    n_acts = torch.stack(
        [get_activation(t, layer_idx, model, tokenizer) for t in nag]
    ).mean(dim=0)

    c_vector = concept_vector(p_acts, n_acts)
    c_vectors = c_vector.unsqueeze(0)
    return c_vectors.to(device)


def attach_gnn_if_needed(cfg, sae: H_SAE, device: torch.device) -> None:
    """
    설정값에 따라 SAE에 feature GNN을 부착한다.
    """
    if not cfg.gnn.use_gnn:
        return

    A_norm = build_feature_graph_from_decoder(
        sae.decoder.data, top_k=cfg.gnn.top_k
    ).to(device)
    feature_gnn = FeatureGNN(A_norm)
    sae.set_feature_gnn(feature_gnn)


def build_experiment_log(
    cfg,
    train_stats: Dict[str, Any],
    steering_result: Dict[str, Any],
    feature_indices: torch.Tensor,
    layer_idx: int,
    strength: float,
    std_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    한 번의 steering 실험(layer, strength 조합)에 대한 로그 정보를 구성한다.
    """
    f_before = steering_result["f_before"][0, feature_indices].tolist()
    f_after = steering_result["f_after"][0, feature_indices].tolist()

    g_before_tensor = steering_result["g_before"]
    g_after_tensor = steering_result["g_after"]
    if g_before_tensor is not None and g_after_tensor is not None:
        g_before = g_before_tensor[0, feature_indices].tolist()
        g_after = g_after_tensor[0, feature_indices].tolist()
    else:
        g_before = [0.0 for _ in feature_indices]
        g_after = [0.0 for _ in feature_indices]

    log_dict = {
        "model_name": cfg.experiment.model_name,
        "layer_idx": int(layer_idx),
        "sae": {
            "input_dim": cfg.sae.input_dim,
            "sae_dim": cfg.sae.sae_dim,
            "fixed_v_cnt": int(cfg.sae.fixed_v_cnt),
        },
        "gnn": {
            "use_gnn": bool(cfg.gnn.use_gnn),
            "top_k": int(cfg.gnn.top_k),
        },
        "steering": {
            "feature_idx": int(cfg.experiment.steered_feature_idx),
            "strength": float(strength),
        },
        "prompt": cfg.experiment.prompt,
        "output": {
            "before": steering_result["y_before"],
            "after": steering_result["y_after"],
        },
        "feature_stats": {
            "indices": [int(i) for i in feature_indices.tolist()],
            "f_before": f_before,
            "f_after": f_after,
            "g_before": g_before,
            "g_after": g_after,
        },
        "loss_summary": {
            "train_mean_loss": train_stats["mean_loss"],
            "train_mean_l2": train_stats["mean_l2"],
            "train_mean_l1": train_stats["mean_l1"],
            "num_steps": train_stats["num_steps"],
            "num_batches": train_stats["num_batches"],
        },
    }
    if std_info is not None:
        log_dict["feature_std_summary"] = std_info
    return log_dict


def run_experiment(cfg) -> None:
    """
    전체 실험 파이프라인:
        - LLM/토크나이저 로드
        - 데이터 로딩
        - layer / strength sweep 설정
        - SAE 학습
        - GNN-XAI 기반 steering 분석
        - JSON 로그 및 플롯 저장
    """
    device = torch.device(
        cfg.experiment.device if torch.cuda.is_available() else "cpu"
    )

    # 1. 모델 / 토크나이저 로드
    model_name = cfg.experiment.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 2. 데이터로더 준비 (공통)
    data_loader = build_dataloader(cfg, tokenizer)

    # 2-1. 실행 단위 timestamp 디렉토리 생성 (config 기반)
    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root_dir = os.path.join(cfg.paths.logs_root, run_ts)
    plots_dir = os.path.join(cfg.paths.plots_root, run_ts)
    os.makedirs(log_root_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # layer / strength sweep 구성
    layer_list = list(getattr(cfg.experiment, "layer_sweep", [cfg.sae.layer_idx]))
    strength_list = list(
        getattr(cfg.experiment, "strength_sweep", [cfg.experiment.steering_strength])
    )

    for layer_idx in layer_list:
        # 3. 컨셉 벡터 계산 (layer별)
        c_vectors = prepare_concept_vectors(
            cfg, model, tokenizer, device, layer_idx=layer_idx
        )

        # 4. SAE 초기화 (layer별)
        d_model = cfg.sae.input_dim
        d_sae = cfg.sae.sae_dim
        sae = H_SAE(d_model, d_sae, c_vectors.to(device))

        # fixed_v_cnt를 config에서도 접근 가능하게 동기화
        cfg.sae.fixed_v_cnt = int(sae.fixed_v_cnt)

        # 5. GNN 부착 (옵션)
        attach_gnn_if_needed(cfg, sae, device)

        # 6. SAE 학습
        train_cfg = {
            "batch_size": cfg.sae.batch_size,
            "epochs": cfg.sae.epochs,
            "lr": cfg.sae.lr,
            "l1_coeff": cfg.sae.l1_coeff,
            "layer_idx": layer_idx,
            "max_steps": cfg.sae.max_steps,
            "device": device,
            "c_vectors_norm": F.normalize(c_vectors.to(device), dim=1),
            "concept_feature_indices": getattr(cfg.sae, "concept_feature_indices", {}),
            "alpha_concept": getattr(cfg.sae, "alpha_concept", {}),
            "positive_targets": getattr(cfg.sae, "positive_targets", None),
        }

        trained_sae, train_stats = train_model(
            sae, model, tokenizer, data_loader, train_cfg
        )

        # 7-a. SAE feature mean/std 계산 및 std plot
        mean_f, std_f = compute_feature_stats(
            model=model,
            tokenizer=tokenizer,
            sae=trained_sae,
            data_loader=data_loader,
            layer_idx=layer_idx,
            device=device,
            max_batches=cfg.experiment.num_batches_for_stats,
        )
        std_plot_path = os.path.join(
            plots_dir, f"feature_std_layer_{layer_idx}.html"
        )
        plot_feature_std_curve(std_f, save_path=std_plot_path)
        std_info = (
            {
                "plot_path": std_plot_path,
                "mean_std": float(mean_f.std().item()) if mean_f is not None else None,
                "max_std": float(std_f.max().item()) if std_f is not None else None,
            }
            if std_f is not None and mean_f is not None
            else None
        )

        # 7-a-2. LLM hidden mean/std 계산 및 std plot
        mean_h, std_h = compute_llm_hidden_stats(
            model=model,
            tokenizer=tokenizer,
            data_loader=data_loader,
            layer_idx=layer_idx,
            device=device,
            max_batches=cfg.experiment.num_batches_for_stats,
        )
        llm_std_plot_path = os.path.join(
            plots_dir, f"llm_std_layer_{layer_idx}.html"
        )
        plot_llm_std_curve(std_h, save_path=llm_std_plot_path)

        # 7-b. strength sweep
        feature_idx = cfg.experiment.steered_feature_idx

        for strength in strength_list:
            steering_vector = trained_sae.decoder.data[feature_idx].to(model.device)

            steering_result = run_steering_analysis(
                model=model,
                tokenizer=tokenizer,
                sae=trained_sae,
                prompt=cfg.experiment.prompt,
                layer_idx=layer_idx,
                steering_vector=steering_vector,
                strength=strength,
            )

            # steered feature 중심으로 plot / log에 쓸 index selection
            if trained_sae.feature_gnn is not None:
                A_norm = trained_sae.feature_gnn.A_norm
                indices = select_topk_neighbors(
                    A_norm, center_idx=feature_idx, k=cfg.experiment.top_k_for_plot
                )
            else:
                indices = torch.tensor([feature_idx], device=device)

            # steering_result의 f/g는 CPU 텐서이므로 인덱스도 CPU로 맞춰준다.
            indices_cpu = indices.cpu()

            f_before_sel = steering_result["f_before"][0, indices_cpu]
            f_after_sel = steering_result["f_after"][0, indices_cpu]

            if (
                steering_result["g_before"] is not None
                and steering_result["g_after"] is not None
            ):
                g_before_sel = steering_result["g_before"][0, indices_cpu]
                g_after_sel = steering_result["g_after"][0, indices_cpu]
            else:
                g_before_sel = torch.zeros_like(f_before_sel)
                g_after_sel = torch.zeros_like(f_after_sel)

            # JSON 로그 저장
            log_dict = build_experiment_log(
                cfg,
                train_stats,
                steering_result,
                indices_cpu,
                layer_idx=layer_idx,
                strength=strength,
                std_info=std_info,
            )
            log_filename = f"L{layer_idx}_S{strength}.json"
            log_path = save_experiment_log(
                log_dict,
                log_dir=log_root_dir,
                filename=log_filename,
            )
            print(
                f"[LOG] Experiment saved to: {log_path} "
                f"(layer={layer_idx}, strength={strength})"
            )

            # Plot 저장 (steered feature 중심)
            plot_path = os.path.join(
                plots_dir, f"feature_{feature_idx}_L{layer_idx}_S{strength}.html"
            )
            plot_feature_changes(
                indices=indices_cpu.tolist(),
                f_before=f_before_sel.cpu().tolist(),
                f_after=f_after_sel.cpu().tolist(),
                g_before=g_before_sel.cpu().tolist(),
                g_after=g_after_sel.cpu().tolist(),
                save_path=plot_path,
            )
            print(f"[PLOT] Feature change plot saved to: {plot_path}")
