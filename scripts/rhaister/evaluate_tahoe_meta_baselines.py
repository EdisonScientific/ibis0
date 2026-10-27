import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from calibration_metrics import compute_calibration  # noqa: E402
from utils import (
    DOSE_MODES,
    apply_max_predictable_dose,
    build_rhaister_data,
    get_context_measurements,
    get_fit_measurements,
    get_target_measurements,
    get_target_questions,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Rhaister baseline run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-parquet",
        type=Path,
        required=True,
        help="Tahoe-meta episodes parquet.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="holdout")
    parser.add_argument(
        "--dose-mode",
        choices=list(DOSE_MODES),
        default="require",
        help=(
            "'require': a treatment is (drug, dose, unit), and missing dose metadata is an error. "
            "'ignore': a treatment is the drug alone -- needed for the published episode datasets, "
            "which carry no concentration fields. See utils.get_treatment_key for what this costs."
        ),
    )
    parser.add_argument("--include-context-evidence", action="store_true", help="Add prompt stubs to pooled fit data.")
    parser.add_argument(
        "--exclude-eval-targets", action="store_true", help="Drop held-out target answers from fit data."
    )
    parser.add_argument(
        "--fit-mode",
        choices=["pooled", "prompt-local"],
        default="pooled",
        help="Use one pooled fit per gene or one independent fit per held-out prompt.",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def build_rhaister_tasks(args: argparse.Namespace, episodes: pd.DataFrame) -> tuple[str, list[dict]]:
    """Prepare prompt-local or pooled fit/eval rows; Rhaister matrices are built later."""
    if args.fit_mode == "prompt-local":
        # Prompt-local fits one independent model from the stubs shown in each prompt.
        tasks = []
        for episode in episodes[episodes["split"] == args.eval_split].to_dict("records"):
            episode_df = pd.DataFrame([episode])
            questions = get_target_questions(episode_df, args.eval_split, args.dose_mode)
            tasks.append(
                {
                    "name": str(episode["episode_id"]).replace("/", "_"),
                    "train": get_context_measurements(episode_df, args.dose_mode),
                    "test": get_target_measurements(episode_df, args.eval_split, args.dose_mode),
                    "questions": questions,
                    "genes": [str(questions.iloc[0]["gene"])],
                }
            )
        return "prompt_local", tasks

    # Pooled fits one model per held-out gene from the selected global evidence rows.
    test = get_target_measurements(episodes, args.eval_split, args.dose_mode)
    fit_measurements = get_fit_measurements(episodes, args, test)
    questions = get_target_questions(episodes, args.eval_split, args.dose_mode)
    tasks = []
    for gene, test_gene in test.groupby("gene", sort=False):
        # Predictions are saved one file per gene, so keep gene symbols safe for paths.
        safe_gene = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(gene))
        task_name = safe_gene[:120] or "gene"
        tasks.append(
            {
                "name": task_name,
                "train": fit_measurements[fit_measurements["gene"] == gene],
                "test": test_gene,
                "questions": questions[questions["gene"] == gene],
                "genes": [str(gene)],
            }
        )
    return "pooled", tasks


def fit_and_score_tasks(
    train,
    tasks,
    predictions_dir: Path,
    fit_scope: str,
    dose_mode: str = "require",
) -> tuple[dict, pd.DataFrame]:
    """Fit Rhaister for each task and score predictions."""
    scored_rows = []
    n_eval = 0
    n_fit = 0

    for task in tasks:
        n_eval += int(len(task["questions"]))
        fit_rows = task["train"]
        test_rows = task["test"]
        if fit_rows.empty or test_rows.empty:
            continue
        test_rows = apply_max_predictable_dose(test_rows, fit_rows)
        questions = apply_max_predictable_dose(task["questions"], fit_rows)

        # Build one-gene train matrices so every fit value is observed, not default-filled.
        data = build_rhaister_data(
            fit_rows,
            test_rows,
            genes=task["genes"],
        )
        if data["n_test"] == 0:
            continue

        predictions_path = predictions_dir / f"{task['name']}.parquet"

        # Patch Rhaister's prediction writer so it saves predictions from our explicit data dict.
        original_writer = train._save_predictions_parquet
        saved_env = {name: os.environ.get(name) for name in ("HP_ZS_MODEL", "HP_PREG", "HP_CALNET")}
        unseen_test_cells = set(data["test_cells"]) - set(data["train_cells"])

        def write_from_explicit_data(experiment_name, split_name, y_pred, d_pred, fdr_pred, data):
            """Prediction-writer hook matching Rhaister's signature."""
            genes = np.asarray(data["gene_cols"])
            cells = np.asarray(data["test_cells"])
            treatments = np.asarray(data["test_treatments"])
            n_rows, n_genes = y_pred.shape
            fdr = np.full_like(y_pred, np.nan, dtype=np.float64) if fdr_pred is None else np.asarray(fdr_pred)
            pred = pd.DataFrame(
                {
                    "cell_line": np.repeat(cells, n_genes),
                    "treatment": np.repeat(treatments, n_genes),
                    "gene": np.tile(genes, n_rows),
                    "y_pred": np.asarray(y_pred).ravel(),
                    "fdr_pred": fdr.ravel(),
                }
            )
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(predictions_path, index=False)
            return predictions_path

        # Run Rhaister with the few knobs needed for explicit-data, one-gene fits.
        os.environ["HP_SAVE_PREDICTIONS"] = str(predictions_path)
        train._save_predictions_parquet = write_from_explicit_data
        if unseen_test_cells:
            os.environ["HP_ZS_MODEL"] = "ridge"
        os.environ["HP_PREG"] = "0"
        os.environ["HP_CALNET"] = "0"
        try:
            train.train_and_evaluate(
                experiment_name=f"tahoe_meta_{fit_scope}",
                split_name=f"tahoe_meta/{fit_scope}/{task['name']}",
                log=False,
                data=data,
                zeroshot=bool(unseen_test_cells),
            )
        finally:
            train._save_predictions_parquet = original_writer
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            os.environ.pop("HP_SAVE_PREDICTIONS", None)

        # Read Rhaister LFC predictions and join them back to the Tahoe-meta binary questions.
        pred = pd.read_parquet(predictions_path)
        pred["cell_line"] = pred["cell_line"].astype(str)
        pred["treatment"] = pred["treatment"].astype(str)
        pred["gene"] = pred["gene"].astype(str)

        merged = questions.merge(pred, on=["cell_line", "treatment", "gene"], how="left")
        group_cols = [
            "episode_id",
            "target_cell",
            "cell_line",
            "drug",
            "treatment",
            "gene",
            "target_direction",
            "label",
        ]

        # Convert each matched LFC prediction into soft/hard probabilities for Brier scoring.
        for _, group in merged.groupby(group_cols, sort=False):
            available = group[group["y_pred"].notna()].copy()
            if available.empty:
                continue

            available["signed_lfc_pred"] = [
                float(lfc) if direction == "upregulates" else -float(lfc)
                for lfc, direction in zip(available["y_pred"], available["target_direction"], strict=True)
            ]
            row = available.loc[available["signed_lfc_pred"].idxmax()]
            signed_lfc = float(row["signed_lfc_pred"])
            clipped_lfc = np.clip(signed_lfc, -60.0, 60.0)
            scored_rows.append(
                {
                    **{column: group.iloc[0][column] for column in group_cols},
                    "n_matched_doses": int(len(available)),
                    "lfc_pred": float(row["y_pred"]),
                    "fdr_pred": float(row["fdr_pred"]),
                    "signed_lfc_pred": signed_lfc,
                    "p_soft": float(1.0 / (1.0 + np.exp(-clipped_lfc))),
                    "p_hard": float(signed_lfc > 0.5),
                }
            )
        n_fit += 1

    scored = pd.DataFrame(scored_rows)
    n_scored = int(len(scored))
    if scored.empty:
        soft_metrics = {"n": 0, "brier": None, "brier_reward": None, "auroc": None}
        hard_metrics = {"n": 0, "brier": None, "brier_reward": None, "auroc": None}
    else:
        labels = scored["label"].to_numpy(dtype=np.float64)
        # Reuse the shared Tahoe-meta calibration helper so AUROC matches the rest of the paper analyses.
        soft_report = compute_calibration(scored["p_soft"].to_numpy(dtype=np.float64), labels, with_smece=False)
        hard_report = compute_calibration(scored["p_hard"].to_numpy(dtype=np.float64), labels, with_smece=False)
        soft_metrics = {
            "n": n_scored,
            "brier": soft_report.brier,
            "brier_reward": float(1.0 - soft_report.brier),
            "auroc": soft_report.auroc,
        }
        hard_metrics = {
            "n": n_scored,
            "brier": hard_report.brier,
            "brier_reward": float(1.0 - hard_report.brier),
            "auroc": hard_report.auroc,
        }

    return {
        "n_eval": n_eval,
        # In dose-free mode a drug has exactly one treatment key, so there is no dose to select and
        # `apply_max_predictable_dose` is a no-op. Reporting "max-predictable" there would claim a policy
        # that never ran.
        "dose_policy": "max-predictable" if dose_mode == "require" else "none (dose-free treatments)",
        "n_models_fit": int(n_fit),
        "n_with_matched_doses": n_scored,
        "n_without_matched_doses": n_eval - n_scored,
        "direct_match_coverage": float(n_scored / n_eval) if n_eval else 0.0,
        "soft": soft_metrics,
        "hard": hard_metrics,
    }, scored


def main() -> None:
    """Run the Rhaister baseline: build matrix, fit/predict, score, write report."""
    args = parse_args()

    # Load Rhaister and the Tahoe-meta episodes.
    import rhaister.train as train

    episodes = pd.read_parquet(args.episodes_parquet)

    # Prepare the fit/eval rows for each Rhaister task.
    fit_scope, tasks = build_rhaister_tasks(args, episodes)
    predictions_dir = args.work_dir / f"{fit_scope}_predictions"

    # Fit Rhaister, convert LFC predictions to probabilities, and score Brier reward.
    binary_report, scored = fit_and_score_tasks(train, tasks, predictions_dir, fit_scope, args.dose_mode)
    report = {
        "episodes_parquet": str(args.episodes_parquet),
        "fit_scope": fit_scope,
        "fit_mode": args.fit_mode,
        "dose_mode": args.dose_mode,
        "include_context_evidence": bool(args.include_context_evidence),
        "exclude_eval_targets": bool(args.exclude_eval_targets),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "work_dir": str(args.work_dir),
        "predictions_dir": str(predictions_dir),
        "binary_tahoe_meta_metrics": binary_report,
    }
    report_json = json.dumps(report, indent=2, sort_keys=True)
    print(report_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report_json + "\n")
    scored.to_csv(args.out.with_suffix(".predictions.csv"), index=False)


if __name__ == "__main__":
    main()
