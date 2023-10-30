# export_onnx_falcon.py
# Load and quantize Falcon 7B model before exporting it to ONNX.
# source: https://huggingface.co/blog/convert-transformers-to-onnx
# Windows/MacOS/Linux


from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers.onnx import FeaturesManager
from optimum.onnxruntime import ORTModelForCausalLM


def low_level_export(model, tokenizer):
	###################################################################
	# Export with torch.onnx (low-level)
	###################################################################
	dummy_inputs = tokenizer(
		'the quick brown fox jumped over the lazy dog', 
		return_tensors='pt'
	)

	# Export.
	torch.onnx.export(
		model,
		tuple(dummy_inputs.values()),
		f="bitsandbytes_quantized_4bit_falcon-7b-model.onnx",
		input_names=['input_ids', 'attention_mask'],
		output_names=['logits'],
		dynamic_axes={
			'input_ids': {0: 'batch_size', 1: 'sequence'},
			'attention_mask': {0: 'batch_size', 1: 'sequence'},
			'logits': {0: 'batch_size', 1: 'sequence'},
		},
		do_constant_folding=True,
		opset_version=17,
	)


def mid_level_export(model, tokenizer):
	###################################################################
	# Export with transformers.onnx (mid-level)
	###################################################################
	# Load config.
	feature = "text-generation"
	model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
		model, feature
	)
	onnx_config = model_onnx_config(model.config)

	# Export.
	onnx_inputs, onnx_outputs = transformers.onnx.export(
		preprocessor=tokenizer,
		model=model,
		config=onnx_config,
		opset=13,
		output=Path("bitsandbytes_quantized_4bit_falcon-7b-model.onnx")
	)


def high_level_export(model_id):
	###################################################################
	# Export with optimum.onnxruntime (high-level)
	###################################################################
	# Load model (note, there is no quantization unlike before).
	model = ORTModelForCausalLM.from_pretrained(
		model_id, 
		from_transformers=True	# argument deprecated and will be removed in optimum 2.0. Recommended use export instead
	)
	model.save_pretrained('./falcon-7b.onnx')


def main():
	###################################################################
	# Load the model
	###################################################################
	# Define model for download and the bitsandbytes config for the
	# quantization step.
	model_id = "tiiuae/falcon-7b"				# Note that the target_module specified below in LoraConfig will only work with this model
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,						# enable 4bit quantization by replacing the linear layers with fp4/nf4 layers from bitsandbytes
		bnb_4bit_quant_type='nf4',				# sets the quantization data type in the bnb.nn.Linear4Bit layers (either fp4 or np4)
		bnb_4bit_compute_dtype=torch.bfloat16,	# sets the computational type which might be different than the input type
	) # Note that np4 stands for normal point 4 bit 

	# Initialize model tokenizer and the model.
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		quantization_config=bnb_config,			# quantizes the model with the above config
		# device_map={"": 0},						# note that you will need GPU to quantize the model
		device_map="auto",
		trust_remote_code=True,
	)

	# Export the model.
	# low_level_export(model, tokenizer)	# get value-error on passing data through graph
	# mid_level_export(model, tokenizer)	# not working -> error message shows falcon not in list of supported models
	# high_level_export(model_id)			# not working -> error message shows falcon not in list of supported models

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()