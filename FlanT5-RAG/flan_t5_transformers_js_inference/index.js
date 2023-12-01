// Import modules.
import * as readline from 'readline';
import { AutoModelForSeq2SeqLM, AutoTokenizer } from '@xenova/transformers';
// import { pipeline } from '@xenova/transformers';


async function  main () {
    // Initialize tokenizer & model.
    const model_id = 'dmmagdal/flan-t5-small-onnx';
    // const model_id = 'dmmagdal/flan-t5-base-onnx';
    // const model_id = 'dmmagdal/flan-t5-large-onnx';
    // const model_id = 'dmmagdal/flan-t5-xl-onnx';
    let tokenizer = await AutoTokenizer.from_pretrained(
        model_id,
        {
            cache_dir: model_id.replace('dmmagdal/', ''),
        }
    );
    let model = await AutoModelForSeq2SeqLM.from_pretrained(
        model_id, 
        {
            quantized: false, // passing in quantized: false means that transformers.js wont look for the quantized onnx model files
            cache_dir: model_id.replace('dmmagdal/', ''),   // passing in cache_dir value to specify where to save files locally.
        }
    );
    // let generator = await pipeline(
    //     'text2text-generation', 
    //     model_id,
    //     {
    //         quantized: false,
    //     }
    // );   // pipeline abstraction for all-in-one

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
        // let decoded_output = await generator(input_text);
        // console.log(decoded_output);
    }

    // Exit the program.
    process.exit(0);
}


main()