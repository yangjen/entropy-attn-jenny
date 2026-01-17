## To run tests on the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_kernel.py`

## To run tests on the end to end model with the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl flash_attention_2`
`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl entropy_attn`

## To run on RULER with baseline flash_attention_2
`TRITON_DISABLE_AUTOTUNE=1 CUDA_VISIBLE_DEVICES=[device_num] python run_ruler_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_root [data_path] \
  --tasks [qa_task] \
  --max_new_tokens 64 \
  --compact \
  --log_every 10 \
  --eval_mode ruler_part \
  --attn_impl flash_attention_2 \
  --dtype bf16 \
  --deterministic \
  --time \
  --time_skip 4`

## To run on RULER with entropy_attn kernel
`TRITON_DISABLE_AUTOTUNE=1 CUDA_VISIBLE_DEVICES=[device_num] python run_ruler_eval_timed.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_root [data_path] \
  --tasks [qa_task] \
  --max_new_tokens 64 \
  --compact \
  --log_every 10 \
  --eval_mode ruler_part \
  --attn_impl entropy_attn \
  --dtype bf16 \
  --deterministic \
  --time \
  --time_skip 4`

