# GPTQ Falcon 7B

Description: This is a quick example of finetuning the Falcon 7B model with GPTQ given a set of text.


### References

 - transformers
	 - [AutoModelForCausalLM](https://huggingface.co/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForCausalLM)
	 - [Quantization](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/quantization#quantization)
	 - [Quantization concept](https://huggingface.co/docs/text-generation-inference/conceptual/quantization)
     - [Quantization in Optimum](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
	 - [PeftConfig](https://huggingface.co/docs/peft/main/en/package_reference/config#peft.PeftConfig)
	 - [Huggingface blog](https://huggingface.co/blog/gptq-integration)
 - tutorial
	 - [Llama 2 7B Video](https://www.youtube.com/watch?v=RlCQTtIYajM&ab_channel=1littlecoder)
	 - [Llama 2 Collab](hhttps://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing)
	 - [Medium article](https://medium.com/@jain.sm/quantizing-llms-using-auto-gptq-in-colab-59e20b125e62) on Quantizing LLMs using auto gptq in colab
 - model
	 - [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b)
	 - [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b)
     - [Model results](https://huggingface.co/models?search=gptq) for all models quantized with GPTQ