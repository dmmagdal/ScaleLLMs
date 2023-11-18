# bitsandbytes_load_qlora_falcon.py
# Load the (quantized + finetuned) Falcon 7B model and merge its LORA
# adapter back into the main model before saving it locally.
# Windows/MacOS/Linux


from argparse import ArgumentParser
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def main():
	# Parse arguments.
	parser = ArgumentParser(
		description="Load the quantized + finetuned falcon 7b model before merging the adapter back into the main model"
	)
	parser.add_argument(
		'--quantize_4_bit',
		action='store_true',
		default=False,
		help="load 4-bit quantized model"
	)
	args = parser.parse_args()

	if args.quantize_4_bit:
		print("Saving 4-bit quantized model is not currently supported in transformers.")
		print("No model/tokenizer was saved.")
		exit(0)

	###################################################################
	# Load the model
	###################################################################
	bits = "4bit" if args.quantize_4_bit else "8bit"
	adapter_id = f'./{bits}-falcon7b-results-model'
	adapter_config = PeftConfig.from_pretrained(adapter_id)
	model_id = adapter_config.base_model_name_or_path
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		# device_map={"": 0},						# note that you will need GPU to quantize the model.
		device_map="auto",
	)
	peft_model = PeftModel.from_pretrained(model, adapter_id)

	# IMPORTANT NOTE: The peft config base_model_name_or_path contains
	# the respective name or path of the model that was used to train.
	# For instance, if the base model used to train the adapter was the
	# 8bit quantized falcon 7b model saved under 
	# ./bitsandbytes_quantized_8bit_falcon-7b (a local save) or it can
	# use a name for a model pulled from huggingface hub (see this blog
	# post: https://huggingface.co/blog/peft).
	print(f'Base model name/path: {adapter_config.base_model_name_or_path}')
	print(f'Base model:\n{model}')
	print(f'Peft model:\n{peft_model}')

	###################################################################
	# Sample from the model
	###################################################################
	text = "Elon Musk "
	device = "cuda:0"

	tokenizer.pad_token = tokenizer.eos_token
	inputs = tokenizer(text, return_tensors="pt").to(device)
	outputs = peft_model.generate(**inputs, max_new_tokens=20)
	print(tokenizer.decode(outputs[0], skip_special_tokens=True))

	###################################################################
	# Merge adapter back into model
	###################################################################
	# This has the added benefit that we reduce (memory) overhead the 
	# next time we wish to load the finetuned model. We can now load
	# the merged QLora model rather than load the base model and merge
	# it with the Lora adapter like we did above.
	# merged_model = peft_model.merge_and_unload(progressbar=True)
	# merged_model.save_pretrained(f'{bits}-merged-qlora-and-base-falcon7b-model')

	# NOTE: This may have adverse affects depending on the GPU you have
	# on your system. I have personally gotten the
	# =============================================
	# ERROR: Your GPU does not support Int8 Matmul!
	# =============================================
	# on my GPU server because I have only P100 cards installed (those
	# do NOT support/contain Int8 Tensor Cores according to this nvidia
	# documentation: https://docs.nvidia.com/deeplearning/tensorrt/
	# support-matrix/index.html#hardware-precision-matrix) hence why I
	# commented out the lines.

	# Exit the program.
	exit(0)


def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)


if __name__ == '__main__':
	main()