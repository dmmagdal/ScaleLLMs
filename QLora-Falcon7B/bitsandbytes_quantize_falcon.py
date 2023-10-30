# bitsandbytes_quantize_falcon.py
# Load the Falcon 7B model and quantize it with bitsandbytes.
# Windows/MacOS/Linux


import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer


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
		# device_map={"": 0},						# note that you will need GPU to quantize the model.
		device_map="auto",
		trust_remote_code=True,
	)
	
	# Save the model.
	tokenizer.save_pretrained('./bitsandbytes_quantized_4bit_falcon-7b')
	model.save_pretrained('./bitsandbytes_quantized_4bit_falcon-7b')

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()