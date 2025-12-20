## To run tests on the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_kernel.py`

## To run tests on the end to end model with the kernel

`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl flash_attention_2`
`CUDA_VISIBLE_DEVICES=[device_num] PYTHONPATH=. python test_llama.py --attn-impl entropy_attn`
