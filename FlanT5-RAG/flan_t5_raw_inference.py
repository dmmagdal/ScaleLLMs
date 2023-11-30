# flan_t5_raw_inference.py
# Build a simple bot with Flan T5 to test out its text generation and
# prompting capabilities.
# Windows/MacOS/Linux
# Python 3.10


import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
	# Determine device (cpu, mps, or cuda).
	device = 'cpu'
	if torch.backends.mps.is_available():
		device = 'mps'
	elif torch.cuda.is_available():
		device = 'cuda'

	# Model ID (variant of model to load).
	# model_id = "google/flan-t5-small"
	model_id = "google/flan-t5-base"
	# model_id = "google/flan-t5-large"			# Can run on 8GB memory (may OOM if default generate() parameters are changed)
	# model_id = "google/flan-t5-xl"			# Requires 16GB memory to run
	
	# Initialize tokenizer & model.
	# tokenizer = T5Tokenizer.from_pretrained(model_id)
	# model = T5ForConditionalGeneration.from_pretrained(model_id)
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

	# Pass model to device.
	model = model.to(device)

	# Infinite loop. Prompt the model.
	print("Ctrl + C or enter \"exit\" to end the program.")
	text_input = ''
	while text_input != "exit":
		# Take in the input text.
		text_input = input("> ")
		if text_input == "exit":
			continue

		# Tokenize and process the text in the model. Print the output.
		input_ids = tokenizer(
			text_input, 
			return_tensors='pt'
		).input_ids.to(device)
		# output = model.generate(input_ids, max_length=512)
		output = model.generate(
			input_ids, 
			min_length=64,				# default 0, the min length of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
			max_length=512,				# default 20, the max langth of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
			length_penalty=2,			# default 1.0, exponential penalty to the length that is used with beam based generation
			temperature=0.7,			# default 1.0, the value used to modulate the next token probabilities
			num_beams=16, 				# default 4, number of beams for beam search
			no_repeat_ngram_size=3,		# default 3
			early_stopping=True,		# default False, controls the stopping condition for beam-based methods
		)	# more detailed configuration for the model generation parameters. Depending on parameters, may cause OOM. Should play around with them to get desired output.
		print(tokenizer.decode(output[0], skip_special_tokens=True))

	# Notes:
	# -> Flan-T5 works best when prompted when doing 
	#	ConditionalGeneration. For instance: "what does George Bush
	#	do?" will return "president of united states" or "continue 
	# 	the text: the quick brown fox" gives "was able to run away from
	#	the tiger.". As you can see, the responses are quite short. 
	#	It's not *ideal* for a ChatGPT alternative but seems quite 
	#	capable.
	# -> Passing custom parameters to the model.generate() function
	#	yields much more detailed output. The downside is that it
	#	requires tuning and these parameters cannot be adjusted
	#	"on the fly" in a live application unless using a jupyter 
	#	notebook.
	# -> Using AutoTokenizer and AutoModelForSeq2SeqLM give no warning
	#	messages when initializing the model compared to using the
	#	T5Tokenizer and T5ModelForConditionalGeneration classes.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()