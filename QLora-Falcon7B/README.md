# QLora Falckn 7B

Description: This is a quick example of finetuning the Falcon 7B model with QLora given a set of text.


### References

 - transformers
	 - [AutoModelForCausalLM](https://huggingface.co/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForCausalLM)
	 - [BitsAndBytesConfig](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)
	 - [Quantization](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/quantization#quantization)
	 - [Lora](https://huggingface.co/docs/peft/conceptual_guides/lora)
	 - [PeftConfig](https://huggingface.co/docs/peft/main/en/package_reference/config#peft.PeftConfig)
	 - [Huggingface blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
 - tutorial
	 - [GPT-Neo Video](https://www.youtube.com/watch?v=NRVaRXDoI3g)
	 - [GPT-Neo Collab](https://colab.research.google.com/drive/1Vvju5kOyBsDr7RX_YAvp6ZsSOoSMjhKD?usp=sharing#scrollTo=E0Nl5mWL0k2T)
	 - [Falcon-7B Video](https://www.youtube.com/watch?v=2PlPqSc3jM0)
	 - [Falcon-7B Collab](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing)
	 - [Medium article](https://medium.com/@srishtinagu19/quantizing-falcon-7b-instruct-for-running-inference-on-colab-bd97066aa49d) to Quantize Falcon 7B Instruct for running inference on colab
	 - [Medium article](https://vilsonrodrigues.medium.com/run-your-private-llm-falcon-7b-instruct-with-less-than-6gb-of-gpu-using-4-bit-quantization-ff1d4ffbabcc) to Run your private LLM: Falcon-7B-Instruct with less than 6GB of GPU using 4-bit quantization
 - model
	 - [GPT-NeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b)
	 - [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b)