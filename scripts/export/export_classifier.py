"""Export an Ibis0 classifier checkpoint (prime-rl DCP) to a Hugging Face model directory.

The classifier trainer saves torch.distributed checkpoints in prime-rl's Qwen3-MoE layout (stacked
experts, ``router.gate``) together with optimizer state. This reads only the model weights, one
layer at a time, converts them to the per-expert layout of ``Qwen/Qwen3-30B-A3B`` and writes
safetensors shards, the scalar head as ``score.{weight,bias}``, a config with ``num_labels=1`` and
``modeling_ibis0.py`` (loaded with ``trust_remote_code=True``).

    uv run python scripts/export/export_classifier.py \\
        ${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0/checkpoints/step_2400/trainer \\
        outputs/Ibis0-30B-A3B-Classifier
"""

import argparse
import json
import shutil
from pathlib import Path

import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import convert_tt_layer_to_hf
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from transformers import AutoTokenizer

PREFIX = "app.model."
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]


def load(ckpt: Path, keys: list[str]) -> dict:
    """Load ``keys`` (without the DCP prefix) from the checkpoint into CPU tensors."""
    loaded = _load_state_dict_from_keys({PREFIX + k for k in keys}, checkpoint_id=str(ckpt))
    return loaded["app"]["model"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="DCP directory (…/checkpoints/step_N/trainer)")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B", help="source of config + tokenizer")
    args = ap.parse_args()

    model_keys = [
        k.removeprefix(PREFIX)
        for k in dcp.FileSystemReader(str(args.checkpoint)).read_metadata().state_dict_metadata
        if k.startswith(PREFIX)
    ]
    layers = sorted({int(k.split(".")[2]) for k in model_keys if k.startswith("model.layers.")})
    other = [k for k in model_keys if not k.startswith("model.layers.")]
    head = {"lm_head.proj.weight": "score.weight", "lm_head.proj.bias": "score.bias"}
    assert set(head) <= set(other), f"no scalar head in checkpoint: {other}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = [[k for k in model_keys if k.startswith(f"model.layers.{i}.")] for i in layers] + [other]
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for n, keys in enumerate(groups, start=1):
        shard = f"model-{n:05d}-of-{len(groups):05d}.safetensors"
        state = load(args.checkpoint, keys)
        if n <= len(layers):
            convert_tt_layer_to_hf(state, layers[n - 1])
        state = {head.get(k, k): v.contiguous().clone() for k, v in state.items()}
        save_file(state, args.out_dir / shard, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(state, shard))
        total_bytes += sum(v.numel() * v.element_size() for v in state.values())
        print(f"[{n}/{len(groups)}] {shard}: {len(state)} tensors")

    (args.out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": dict(sorted(weight_map.items()))}, indent=2)
    )

    base = Path(snapshot_download(args.base_model, allow_patterns=["config.json", *TOKENIZER_FILES]))
    config = json.loads((base / "config.json").read_text())
    config.update(
        architectures=["Ibis0ForSequenceClassification"],
        auto_map={"AutoModelForSequenceClassification": "modeling_ibis0.Ibis0ForSequenceClassification"},
        num_labels=1,
        id2label={"0": "LABEL_0"},
        label2id={"LABEL_0": 0},
        pad_token_id=AutoTokenizer.from_pretrained(base).pad_token_id,
        torch_dtype="bfloat16",
    )
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for name in TOKENIZER_FILES:
        shutil.copy(base / name, args.out_dir / name)
    shutil.copy(Path(__file__).with_name("modeling_ibis0.py"), args.out_dir / "modeling_ibis0.py")
    print(f"wrote {args.out_dir} ({total_bytes / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
