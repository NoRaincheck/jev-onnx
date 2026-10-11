# potion-base-8M (vendored)

A byte-identical copy of two files from
[`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M),
revision `bf8b056651a2c21b8d2565580b8569da283cab23`, MIT licensed,
© Minish Lab. See the upstream [model card](https://huggingface.co/minishlab/potion-base-8M)
for what the model is and how it was trained.

```
model.safetensors   30,236,760 bytes   sha256 f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2
vocab.txt              219,690 bytes   sha256 1394523a67ddd404a825428018c0582a6998bcfa044ecbcbf1f4d71adb94c61c
```

Verify with `shasum -a 256 model.safetensors vocab.txt`.

These are the *only* two files this project uses from that repo. The tokenizer,
`tokenizer.json`, `config.json` and the ONNX export next to them on the Hub are
deliberately not vendored and never imported — the tokenisation stays inside the
ONNX graph's `StringNormalizer(LOWER) -> StringSplit` pair. See
`src/jev_onnx/potion.py`.

`resolve()` in that module prefers this directory and falls back to
`huggingface_hub` when it is absent (for example from an installed wheel), so
nothing here is load-bearing for correctness — only for working offline.
