"""Lightweight HF-transformers OpenAI-compatible /v1/chat/completions server.

An alternative to a vLLM endpoint for the theta' simulator (or any causal LM): it serves the model
through plain HF-transformers generation behind an OpenAI-compatible endpoint, so the BED driver's
AsyncOpenAI client works unchanged. Dynamic batching: concurrent requests are coalesced into one
generate() call.

    uv run python scripts/analysis/rq5_bed/hf_openai_serve.py \
        --ckpt ibis_data/sft/bed-simulator-thetaprime/weights/step_1200 --port 8000
"""

import argparse
import asyncio
import time
from pathlib import Path

import torch
from aiohttp import web

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "envs" / "ibis_env_utils").is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--linger-ms", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.ckpt} (HF transformers, flash_attention_2)...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        .to("cuda")
        .eval()
    )
    model_name = Path(args.ckpt).as_posix()
    print("model ready", flush=True)

    queue: asyncio.Queue = asyncio.Queue()

    async def batcher():
        while True:
            first = await queue.get()
            batch = [first]
            await asyncio.sleep(args.linger_ms / 1000.0)  # let a concurrent burst coalesce
            while len(batch) < args.max_batch and not queue.empty():
                batch.append(queue.get_nowait())
            prompts = [b["prompt"] for b in batch]
            mnt = max(b["max_tokens"] for b in batch)
            temps = [b["temperature"] for b in batch]
            try:
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to("cuda")
                # one shared sampling config per batch: greedy if all temps==0 else sample at max temp
                temp = max(temps)
                gk = dict(do_sample=False) if temp == 0 else dict(do_sample=True, temperature=temp, top_p=1.0, top_k=0)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=mnt, pad_token_id=tok.pad_token_id, **gk)
                outs = tok.batch_decode(g[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            except Exception as e:
                outs = [f"__ERROR__ {e}"] * len(batch)
            for b, o in zip(batch, outs):
                b["fut"].set_result(o)

    async def chat(request):
        body = await request.json()
        msgs = body["messages"]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        fut = asyncio.get_event_loop().create_future()
        await queue.put(
            {
                "prompt": prompt,
                "max_tokens": int(body.get("max_tokens", args.max_new_tokens)),
                "temperature": float(body.get("temperature", 0.0)),
                "fut": fut,
            }
        )
        text = await fut
        return web.json_response(
            {
                "id": "chatcmpl-hf",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
        )

    async def models(request):
        return web.json_response({"object": "list", "data": [{"id": model_name, "object": "model"}]})

    async def start(app):
        app["batcher"] = asyncio.create_task(batcher())

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.on_startup.append(start)
    print(f"serving {model_name} on {args.host}:{args.port}", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
