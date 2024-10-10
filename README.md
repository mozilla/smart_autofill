# Smart Autofill
Autofill HTML Tag Detection

This repo contains training code for the autofill model that can be used to predict labels (e.g. "Zip Code") from HTML tags. More information on the model can be found here: https://huggingface.co/vazish/mobile_bert_autofill.

### Requirements
First do the following steps to set up the virtual env and install the requirements,

```
cd smart_autofill
python3 -m venv smart_autofill
source smart_autofill/bin/activate
pip install -r requirements.txt
```

### Running Training & Inference
To run training, just run the commands below. The dataset will be downloaded automatically from hugging face,
and the base model fine-tuned on it.

```shell
cd smart_autofill
source smart_autofill/bin/activate
cd src
# removes local cached model, allows increased memory for training then runs training
rm -rf google && PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train.py
```

To run inference, update the `predict.py` file to add any tags and run with,
```shell
cd smart_autofill
source smart_autofill/bin/activate
cd src
python predict.py
```

### Running the Streamlit App
```
cd smart_autofill/streamlit-app
python3 -m streamlit run infer.py
```

The app startup can take a few seconds while the model is loaded. Subsequent loads should be faster.
If a local model is present, the model can be loaded by updating `model=...` to the local directory containing the model artifacts.

<p align="center">
  <img src="assets/app_screenshot.png" width="400px" height="200px" />
</p>

### Running the React App
```
cd smart_autofill/react-app
npm install
npm run dev
```

There should be a textarea with a "Classify" button. Add some HTML tags to test (separated by newlines) and click "Classify" to see the result.

<p align="center">
  <img src="assets/react_app_screenshot.png" width="400px" height="200px" />
</p>
