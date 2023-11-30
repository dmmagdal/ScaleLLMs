# flan_t5_raw_inference.py
# Build a simple bot with Flan T5 to test out its text generation and
# prompting capabilities.
# Windows/MacOS/Linux
# Python 3.10


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def main():
	# Determine device (cpu, mps, or cuda).
	device = 'cpu'
	if torch.backends.mps.is_available():
		device = 'mps'
	elif torch.cuda.is_available():
		device = 'cuda'

	# Model ID (variant of model to load).
	# model_id = "google/flan-t5-small"
	# model_id = "google/flan-t5-base"
	model_id = "google/flan-t5-large"			# Can run on 8GB memory
	# model_id = "google/flan-t5-xl"			# Requires 16GB memory to run
	
	# Initialize tokenizer & model.
	tokenizer = T5Tokenizer.from_pretrained(model_id)
	model = T5ForConditionalGeneration.from_pretrained(model_id)

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
		output = model.generate(input_ids, max_length=512)
		print(tokenizer.decode(output[0], skip_special_tokens=True))

	# Notes:
	# -> Flan-T5 works best when prompted when doing 
	#	ConditionalGeneration. For instance: "what does George Bush
	#	do?" will return "president of united states" or "continue 
	# 	the text: the quick brown fox" gives "was able to run away from
	#	the tiger.". As you can see, the responses are quite short. 
	#	It's not *ideal* for a ChatGPT alternative but seems quite 
	#	capable.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()