
import { pipeline } from '@xenova/transformers';

/**
 * This class uses the Singleton pattern to ensure that only one instance of the
 * pipeline is loaded. This is because loading the pipeline is an expensive
 * operation and we don't want to do it every time we want to translate a sentence.
 */
class MyClassificationPipeline {
    static task = 'text-classification';
    static model = 'vazish/mobile_bert_autofill';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { progress_callback });
        }

        return this.instance;
    }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
    // Retrieve the translation pipeline. When called for the first time,
    // this will load the pipeline and save it for future use.
    let classifier = await MyClassificationPipeline.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    });

    // Actually perform the translation

    let output = await classifier(event.data.text.split('\n').map(i => i.trim()));
    output = output.map(o => `Label: ${o.label} - Score: ${o.score.toPrecision(4)}`).join('\n');
    // Send the output back to the main thread
    self.postMessage({
        status: 'update',
        output: output
    });
    self.postMessage({
        status: 'complete',
        output: output
    });
});
