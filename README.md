# Scale LLMs

Description: This repo aims to look at different techniques that can allow for LLMs (language models over 1 billion parameters) to be scaled to run and train on consumer grade hardware.


### Notes

 - In this case, consumer grade hardware is referring to desktops that have between 8 to 32 GB of memory (RAM), with an average of 16 GB as well as 8 to 16 GB of GPU memory (VRAM).
	 - My M2 Macbook Pro
		 - OS: MacOS Sonoma
		 - Apple M2 Silicon
		 - 16 GB RAM
		 - 8 GB VRAM
	 - My Dell XPS Desktop
		 - OS: Windows 11
		 - Intel i7 CPU
		 - 16 GB RAM
		 - Nvidia 2060 SUPER (8 GB VRAM)
	 - My Darkstar GPU server
		 - OS: Ubuntu 22.04
		 - Dual Intel Xeon CPUs
		 - 64 GB RAM
		 - 3x Nvidia Tesla P100 (16 GB VRAM each)
 - Techniques to be explored in this repo:
	 - QLora (with `bitsandbytes` library in addition to huggingface's `transformers` & `peft`)
	 - GPTQ
	 - GGML/GGUF
 - QLora
	 - `bitsandbytes` requires CUDA GPU to do quantization
	 - `bitsandbytes` benefits:
		 - Can peform "zero-shot quantization" (does not require input data to calibrate the quantized model)
		 - Can quantize any model out of the box as long as it contains `torch.nn.Linear` modules 
		 - Quantization is peformed on model load, no need to run any post-processing or preparation step
		 - Zero peformance degreation when loading adapters and can also merge adapters in a dequantized model
	 - `bitsandbytes` shortcomings:
		 - 4-bit quantization not serializable
 - GPTQs
	 - `auto-gptq` requires CUDA or RoCM (AMD) GPU to do quantization
	 - `auto-gptq` benefits:
		 - fast for text generation
		 - n-bit support
		 - easily serializable
	 - `auto-gptq` shortcomings:
		 - requires a calibration dataset
		 - works for language models only (at the time of writing this 10/2020)
 - GGML/GGUF
 - Can finetune quantized models with `peft` library from huggingface (for GPTQ and QLora quantization)
	 - PEFT stands for "parameter efficient finetuning"
	 - Training on quantized models is not possible

### References

 - YouTube video from Code_Your_Own_AI: [New Tutorial on LLM Quantization w/ QLoRA, GPTQ and Llamacpp, LLama 2](https://www.youtube.com/watch?v=YEVyupJxt1Q&ab_channel=code_your_own_AI)
 - Huggingface Transformers [documentation on quantization](https://huggingface.co/docs/text-generation-inference/conceptual/quantization)
 - Huggingface Transformers [overview of quantization](https://github.com/huggingface/blog/blob/main/overview-quantization-transformers.md) on GitHub
 - Huggingface Optimum [documentation on quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)