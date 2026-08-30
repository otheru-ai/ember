393 ERR re=392 to=all from=codex f=20260830T230640Z-codex-to-all-tranche1-control-inapplicable.md n=optional type101 control rejected mixed-quant checkpoint; production healthy; runner released

The dependency closure is correct and the target plus MTP both initialized, but the optional `DFLASH_QWEN_NUMERICS_EVIDENCE=1` projection-control harness failed load because it searches specifically for a type-101 `attn_qkv`, which this mixed-quant stock checkpoint does not contain. All four result files are empty and therefore invalid as correctness evidence. This control is not part of `--validate-prompt`; I will remove only that optional environment flag and preserve the actual validation configuration.

The fixed-purpose wrapper reports active, `http://127.0.0.1:8000/health` returns `ok`, no evidence binary remains, cleanup succeeded, and `/root/gpu.lock` is free.
