// Import modules.
import * as readline from 'readline';
import { AutoModelForSeq2SeqLM, AutoTokenizer } from '@xenova/transformers';
// import { pipeline } from '@xenova/transformers';


async function  main () {
    // Initialize tokenizer & model.
    // flan-t5-small
    //  load time: 1 second
    //  inference time: 0.25 seconds
    //  load time (quantized): 1 second
    //  inference time (quantized): 0.1 seconds
    // flan-t5-base
    //  load time: 5 seconds
    //  inference time: 0.5 seconds
    //  load time (quantized): 1.5 seconds
    //  inference time (quantized): 0.25 seconds
    // flan-t5-large
    //  load time: 45 - 55 seconds
    //  inference time: 10 - 15 seconds
    //  load time (quantized): 5 seconds
    //  inference time (quantized): 0.5 seconds
    // const model_path = '../../../../flan-t5-small';
    // const model_path = '../../../../flan-t5-base';
    // const model_path = '../../../../flan-t5-large';
    // const model_id = model_path;

    // const model_id = 'dmmagdal/flan-t5-small-onnx-js';
    // const model_id = 'dmmagdal/flan-t5-base-onnx-js';
    const model_id = 'dmmagdal/flan-t5-large-onnx-js';
    // const model_id = 'dmmagdal/flan-t5-xl-onnx-js'; // Does not load :( (is probably too big)
    const cache_dir = model_id.replace('dmmagdal/', '');
    const loadStart = new Date();
    let tokenizer = await AutoTokenizer.from_pretrained(
        model_id,
        {
            // cache_dir: cache_dir,
            local_files_only: true,
        }
    );
    let model = await AutoModelForSeq2SeqLM.from_pretrained(
        model_id,       // model_id for downloading from huggingface (requires huggingface token in .env file) OR model path for loading model from local storage.
        {
            quantized: false,       // passing in quantized: false means that transformers.js wont look for the quantized onnx model files
            // quantized: true,
            // cache_dir: cache_dir,   // passing in cache_dir value to specify where to save files locally.
            local_files_only: true, // use locally stored files.
        }
    );
    // let generator = await pipeline(
    //     'text2text-generation', 
    //     model_id,
    //     {
    //         quantized: false,
    //        // quantized: true,
    //         cache_dir: cache_dir,
    //         local_files_only: true,
    //     }
    // );   // pipeline abstraction for all-in-one
    const loadEnd = new Date();
    const loadTime = loadEnd - loadStart;
    console.log(`Model load time: ${Math.round(loadTime) / 1000} seconds.`);

    // Initialize the console interface.
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    // Infinite loop. Prompt the model.
    let input_text = '';
    while (input_text !== 'exit') {
        // Take in the input text.
        rl.question('> ', async (input) => {
            // Set the input text to be the input from stdin.
            input_text = input;
    
            if (input_text !== 'exit') {
                // Tokenize and process the text in the model. Print 
                // the output.
                let infStart = new Date();
                let { input_ids } = await tokenizer(input_text);
                let outputs = await model.generate(input_ids);
                let decoded = tokenizer.decode(
                    outputs[0], 
                    { skip_special_tokens: true },
                );
                let infEnd = new Date();
                let infTime = infEnd - infStart;
                console.log(decoded);
                console.log(`Inference time: ${Math.round(infTime) / 1000} seconds.`);

                // Pass the text through the pipeline & print the
                // output.
                // let decoded_output = await generator(input_text);
                // console.log(decoded_output);
            }
            else {
                // Close the interface.
                rl.close();
            }
        });

        // Pause the loop until the user provides input (newline
        // ENTER).
        await new Promise(resolve => rl.on('line', resolve));
    }

    // Exit the program.
    process.exit(0);
}


main();