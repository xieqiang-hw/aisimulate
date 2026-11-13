"""Build an AISim Mooncake sample from extracted Claude Code requests."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct

from transformers import AutoTokenizer

DATA = Path.home() / "project/data"
MODEL = "zai-org/GLM-5.2-FP8"
BLOCK_SIZE = 16


def input_messages(events):
    for event in events:
        content = event["content"]
        if isinstance(content, str):
            yield {"role": "user", "content": content}
            continue
        for block in content:
            if block["type"] == "tool_result":
                yield {"role": "tool", "content": block.get("content") or ""}
            else:
                yield {"role": "user", "content": [block]}


def assistant_message(events):
    text, reasoning, tools = [], [], []
    for event in events:
        content = event["content"]
        if isinstance(content, str):
            text.append(content)
            continue
        for block in content:
            kind = block["type"]
            if kind == "text":
                text.append(block["text"])
            elif kind == "thinking":
                reasoning.append(block.get("thinking") or "")
            elif kind == "tool_use":
                tools.append({"function": {"name": block["name"], "arguments": block["input"]}})
    return {"role": "assistant", "content": "".join(text),
            "reasoning_content": "".join(reasoning), "tool_calls": tools}


def prefix_ids(tokens, request_key, index):
    parent = bytes(32)
    result = []
    for start in range(0, len(tokens), BLOCK_SIZE):
        block = tokens[start:start + BLOCK_SIZE]
        payload = struct.pack(f"<{len(block)}I", *block)
        salt = request_key.encode() if len(block) < BLOCK_SIZE else b""
        parent = hashlib.sha256(parent + bytes([len(block)]) + payload + salt).digest()
        value = index.setdefault(parent, len(index))
        if value >= 2**32:
            raise ValueError("too many distinct prefix blocks for u32 IDs")
        result.append(value)
    return result


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=DATA / "GLM_5.2_tokenizer")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context-limit", type=int, default=131072)
    args = parser.parse_args()
    if args.context_limit <= 0:
        parser.error("--context-limit must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    template = (args.tokenizer / "chat_template.jinja").read_text(encoding="utf-8")
    histories = defaultdict(list)
    last_call = Counter()
    counts = Counter()
    index = {}
    emitted = []
    with args.source.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            session = row["session_id"]
            call = row["call_index"]
            if call != last_call[session] + 1:
                raise ValueError(f"out-of-order calls in {session}: {call}")
            last_call[session] = call
            history = histories[session]
            history.extend(input_messages(row["input_delta"]))
            answer = assistant_message(row["response_events"])
            prompt = tokenizer.apply_chat_template(
                history, chat_template=template, tokenize=False, add_generation_prompt=True)
            history.append(answer)
            rendered = tokenizer.apply_chat_template(
                history, chat_template=template, tokenize=False, add_generation_prompt=False)
            if not rendered.startswith(prompt):
                raise ValueError(f"template changed the prompt prefix in {session}:{call}")
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            # Encode the completion after the generation prompt, including the EOS.
            completion = rendered[len(prompt):]
            output = tokenizer.encode(completion, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                output.append(tokenizer.eos_token_id)
            if not output:
                counts["empty_output"] += 1
                continue
            if len(tokens) + len(output) > args.context_limit:
                counts["over_context_limit"] += 1
                continue
            if not row["first_response_event_at"]:
                counts["missing_timestamp"] += 1
                continue
            date = datetime.fromisoformat(row["first_response_event_at"])
            if date.utcoffset() is None:
                raise ValueError(f"timestamp lacks timezone in {session}:{call}")
            time_ms = int(date.timestamp() * 1000)
            trace = {"input_length": len(tokens), "output_length": len(output),
                     "hash_ids": prefix_ids(tokens, f"{session}:{call}", index)}
            source = {key: row[key] for key in ("session_id", "call_index", "response_id",
                                               "first_response_event_at", "time_source")}
            trace["output_token_ids"] = output
            source["output_token_ids"] = output
            emitted.append((time_ms, session, call, trace, source))
    if not emitted:
        parser.error("no usable requests")
    emitted.sort(key=lambda item: item[:3])
    start = emitted[0][0]
    traces, sources = [], []
    for number, (time_ms, _, _, trace, source) in enumerate(emitted):
        trace["timestamp"] = time_ms - start
        source.update(trace_row=number, timestamp=trace["timestamp"])
        traces.append(trace)
        sources.append(source)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, rows in (("requests.jsonl", traces), ("source_requests.jsonl", sources)):
        path = args.output_dir / name
        write_jsonl(path, rows)
        files[name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "format": "mooncake", "trace_block_size": BLOCK_SIZE,
        "context_limit": args.context_limit, "request_count": len(traces),
        "model_mapping": {"target_model": MODEL},
        "timestamp_policy": "first response event proxy, relative milliseconds",
        "history_policy": "logged messages only; original system prompts and tool schemas unavailable in extracted sample",
        "output_policy": "rendered assistant completion encoded separately plus one EOS",
        "source_sha256": sha256(args.source),
        "tokenizer_files": {name: sha256(args.tokenizer / name)
                            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        "excluded": dict(counts), "files": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(traces)} Mooncake requests in {args.output_dir}. Excluded: {dict(counts)}")


if __name__ == "__main__":
    main()
