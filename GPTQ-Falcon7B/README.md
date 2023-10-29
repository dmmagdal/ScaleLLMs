# GPTQ Falcon 7B

Description: This is a quick example of finetuning the Falcon 7B model with GPTQ given a set of text.


### Notes

 - Conda's conda-forge doesn't have the necessary/updated version of `auto-gptq` that is needed to quantize the model. Instead, use python's `virtualenv` to resolve this issue. Don't worry about pytorch detecting CUDA GPU, it will do that if you run the required install command (see pytorch's website).
 - The `auto-gptq` library requires CUDA (and RoCm for AMD GPUs) in order to run its quantization.
 - The `optimum` library from huggingface has its own GPTQ function (have yet to test it out).
	 - Questions for implementation
		 - Can you finetune with `peft`?
		 - Can you run without GPU or use MPS from Apple Silicon?
			 - Update: We cannot use the `optimum` module with MPS from Apple Silicon. Got the following error when attempted to use the module `AssertionError: Torch not compiled with CUDA enabled`
 - Testing
	 - Windows (my Dell XPS Desktop)
		 - Set the `device_map={"":0}` instead of `device_map="auto"` (from the collab notebook) for the `TheBloke/Llama-2-7b-Chat-GPTQ` model `from_pretrained()` arguments in order to be able to run the code on the machine. When using `device_map="auto"`, I see an error similar to the one described in this [issue](https://github.com/tloen/alpaca-lora/issues/368). Refer to the documentation [here](https://huggingface.co/docs/transformers/main_classes/model#large-model-loading) and [here](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained) on how using `device_map` works and how it relates to loading large models. Long story short, using `"auto"` may load parts of the model to CPU/RAM from the GPU and the 8-bit values/tensors may not be supported by CPU (this may explain why the libraries like `auto-gptq` and `bitsandbytes` rely on GPUs and have no CPU only counterparts).
         - Attempting to use huggingface's `optimum` module does kind of work to quantize a model (the `gptq_llama2.py` script is able to quantize the 125M OPT model from Facebook/MetaAI but had issues quantizing Llama 2 with the config used). Given my current experience, I think it's going to be easier to use `auto-gptq` on top of `optimum` instead of just `optimum`.
	 - Linux (my Darkstar GPU server)
		 - Using `device_map="auto"` works as well as `device_map={"": 0}`.
		 - Be aware that the `torch` version in `requirements.txt` may not work properly on the first try when you set up the `virtualenv`. I'd advise everyone to use the necessary install command from the PyTorch website (just specify the correct version).
 - Quantization
	 - Falcon 7B
		 - Programs
			 - `autogptq_quantize_falcon-7b.py`
		 - Results/Notes
			 - Was unable to quantize falcon 7b from `tiiuae/falcon-7b` on my Dell XPS Desktop. Ran into CUDA OOM issue at around 22% quantization.
			 - Was **able** to quantize falcon 7b from `tiiuae/falcon-7b` on my Darkstar GPU server. VRAM usage was around 14GB VRAM (so a 16GB card is very much recommended for 7B models). Actual RAM usage was around 7.5 GB but would still recommend using 16GB RAM. Be aware that the quantization process takes a while (around 1 hour 10 minutes).
			 - Specifying `disable_exllama=True` in the `GPTQConfig` is a requirement for falcon. The default (`disable_exllama=False`) will result in an assertion error for the input features vs the group size part way through the quantization process (actual error is: `assert infeatures % self.group_size == 0`). According to the huggingface documentation, the `disable_exllama` argument in the `GPTQConfig` is for `Whether to use exllama backend. Only works with bits = 4`. As it turns out, I am quantizing falcon 7B with 4-bit quantization in the `autogptq_quantize_falcon-7b.py` script, so this does apply.
			 - The resulting 4-bit quantized model in `autogptq_quantized_4bit_falcon-7b/` folder (and the tokenizer) comes out to around 3.8 GB on disk.
 - Finetuning


### References

 - transformers
	 - [AutoModelForCausalLM](https://huggingface.co/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForCausalLM)
	 - [GPTQConfig](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/quantization#transformers.GPTQConfig)
	 - [Quantization](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/quantization#quantization)
	 - [Quantization concept](https://huggingface.co/docs/text-generation-inference/conceptual/quantization)
	 - [Quantization in Optimum](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
	 - [PeftConfig](https://huggingface.co/docs/peft/main/en/package_reference/config#peft.PeftConfig)
	 - [Peft GitHub](https://github.com/huggingface/peft)
	 - [Huggingface blog](https://huggingface.co/blog/gptq-integration)
	 - [Huggingface blog on peft](https://huggingface.co/blog/peft)
 - tutorial
	 - [Llama 2 7B Video](https://www.youtube.com/watch?v=RlCQTtIYajM&ab_channel=1littlecoder)
	 - [Llama 2 Collab](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing)
	 - [Medium article](https://medium.com/@jain.sm/quantizing-llms-using-auto-gptq-in-colab-59e20b125e62) on Quantizing LLMs using auto gptq in colab
	 - [Towards AI blog](https://towardsai.net/p/machine-learning/gptq-quantization-on-a-llama-2-7b-fine-tuned-model-with-huggingface) on GPTQ Quantization on a Llama 2 7B Fine-Tuned Model With HuggingFace
	 - [Medium article](https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337) on a definitive guide to QLora finetuning of falcon 7b with peft
 - model
	 - [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b)
	 - [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b)
	 - [Model results](https://huggingface.co/models?search=gptq) for all models quantized with GPTQ
 - modules
	 - [auto-gptq pypi](https://pypi.org/project/auto-gptq/)
	 - [auto-gptq GitHub](https://github.com/PanQiWei/AutoGPTQ)
	 - [gptq pypi](https://pypi.org/project/gptq/)