# autogptq_quantize_falcon-7b.py
# Load the Falcon 7B model and quantize it with AutoGPTQ.
# Windows/MacOS/Linux


import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTQConfig
from transformers import DataCollatorForLanguageModeling


def main():
	###################################################################
	# Load the model
	###################################################################

	# Define model for download and the (auto) gptq config for the
	# quantization step.
	model_id = "tiiuae/falcon-7b"
	gptq_config = GPTQConfig(
		bits=4,						# the number of bits to quantize to, supported numbers are (2, 3, 4, 8).
		dataset="c4",				# the dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used in GPTQ paper [‘wikitext2’,‘c4’,‘c4-new’,‘ptb’,‘ptb-new’]
		desc_act=False,				# whether to quantize columns in order of decreasing activation size. Setting it to False can significantly speed up inference but the perplexity may become slightly worse. Also known as act-order.
	)

	# Initialize model tokenizer and the model.
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=gptq_config,	# quantizes the model with the above config
		device_map="auto",					# note that you will need GPU to quantize the model.
		trust_remote_code=True,
	)

	# You can make sure the model has been correctly quantized by 
	# checking the attributes of the linear layers, they should contain
	# qweight and qzeros attributes that should be in torch.int32 
	# dtype.
	print(model.model.decoder.layers[0].self_attn.q_proj.__dict__)

	# Now let's perform an inference on the quantized model. Use the 
	# same API as transformers!
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	text = "Hello my name is"
	inputs = tokenizer(text, return_tensors="pt").to(0)

	out = model.generate(**inputs)
	print(tokenizer.decode(out[0], skip_special_tokens=True))

	###################################################################
	# Upload the model
	###################################################################

	# After quantizing the model, it can be used out-of-the-box for 
	# inference or you can push the quantized weights on the 🤗 Hub to 
	# share your quantized model with the community.
	model.push_to_hub("falcon-7b-gptq-4bit")
	tokenizer.push_to_hub("falcon-7b-gptq-4bit")

	###################################################################
	# Load the quantized model
	###################################################################

	# You can load models that have been quantized using the auto-gptq 
	# library out of the box from the 🤗 Hub directly using 
	# from_pretrained method. Make sure that the model on the Hub have 
	# an attribute quantization_config on the model config file, with 
	# the field quant_method set to "gptq".

	# Most used quantized models can be found under TheBloke namespace 
	# that contains most used models converted with auto-gptq library. 
	# The integration should work with most of these models out of the 
	# box (to confirm and test).

	# Below we will load a llama 7b quantized in 4bit.
	model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
	model = AutoModelForCausalLM.from_pretrained(
		model_id, 
		device_map="auto",
	)
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	# Once tokenizer and model has been loaded, let's generate some 
	# text. Before that, we can inspect the model to make sure it has 
	# loaded a quantized model. As you can see, linear layers have been
	# modified to QuantLinear modules from auto-gptq library.
	print(model)

	# Furthermore, we can see that from the quantization config that we
	# are using exllama kernel (disable_exllama = False). Note that it 
	# only works with 4-bit model.
	print(model.config.quantization_config.to_dict())

	text = "Hello my name is"
	inputs = tokenizer(text, return_tensors="pt").to(0)

	out = model.generate(**inputs, max_new_tokens=50)
	print(tokenizer.decode(out[0], skip_special_tokens=True))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()