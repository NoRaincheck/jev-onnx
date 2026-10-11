# scripts/

Throwaway probes kept because they document *why* the code looks the way it
does. Run any of them directly to reproduce a finding:

| script | what it establishes |
|---|---|
| `probe_op_support.py` | onnxruntime actually runs `StringNormalizer`, `StringSplit` and `ai.onnx.ml::CategoryMapper` (and the exact signatures: two outputs, a `delimiter` attribute, `default_int64` for OOV) |
| `probe_stringsplit.py` | `StringSplit(delimiter="")` splits on runs of **spaces only** — not tabs or newlines |
| `probe_stringsplit_rank.py` | `StringSplit`'s output rank is not trustworthy; `Reshape(C, max(counts))` fixes it |
| `probe_case_folding_examples.py` | side-by-side of ORT's lowercasing vs `str.lower()` on the awkward codepoints |
| `probe_case_folding.py` | exhaustive BMP sweep — the source of the 13-codepoint override table in `tokenization.py` |
| `probe_dataset_schema.py` | Open-Jev row layout, group fanout, option-set shapes |
| `probe_state_json.py` | `state_json` is JSON-encoded: plain strings vs objects |
| `probe_data_scopes.py` | how much of the corpus is prose vs structured, per config and split |
| `probe_structured_states.py` | what the structured states contain (emails, transcripts, telemetry) |
| `probe_lengths.py` | token-length distributions that set `max_len` / `max_label_len` |

```bash
uv run python scripts/probe_op_support.py
```