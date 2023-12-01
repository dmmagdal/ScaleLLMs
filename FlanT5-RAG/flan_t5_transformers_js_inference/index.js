// Import modules.
import * as readline from 'readline';
import { AutoModelForSeq2SeqLM, AutoTokenizer } from '@xenova/transformers'
// const readline = require('readline');
// const { AutoModelForSeq2SeqLM, AutoTokenizer } = require('@xenova/transformers');


async function  main () {
    // Initialize tokenizer & model.
    const model_id = 'dmmagdal/flan-t5-small-onnx';
    // const model_id = 'dmmagdal/flan-t5-base-onnx';
    // const model_id = 'dmmagdal/flan-t5-large-onnx';
    // const model_id = 'dmmagdal/flan-t5-xl-onnx';
    let tokenizer = await AutoTokenizer.from_pretrained(model_id);
    let model = await AutoModelForSeq2SeqLM.from_pretrained(model_id);

    // Infinite loop. Prompt the model.
    let input_text = '';
    while (input_text !== 'exit') {
        // Take in the input text.
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        // Take in the input text.
        rl.question('> ', (input) => {
            input_text = input;
            rl.close();
        });
        if (input_text === 'exit') {
            continue;
        }

        // Tokenize and process the text in the model. Print the 
        // output.
        let { input_ids } = await tokenizer(input_text);
        let outputs = await model.generate(input_ids);
        let decoded = tokenizer.decode(
            outputs[0], 
            { skip_special_tokens: true },
        );
        console.log(decoded);
    }

    // Exit the program.
    process.exit(0);
}


main()