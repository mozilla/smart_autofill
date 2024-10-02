# Smart Autofill
Autofill HTML Tag Detection

Information on the model can be found here: https://huggingface.co/vazish/mobile_bert_autofill

## Requirements
First do the following steps to set up the virtual env and install the requirements,

```
cd smart_autofill
python3 -m venv smart_autofill
source smart_autofill/bin/activate
pip install -r requirements.txt
```

## Running Training & Inference
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

## Running the App
```
cd smart_autofill/app
python3 -m streamlit run infer.py
```

The app startup can take a few seconds while the model is loaded. Subsequent loads should be faster.

Once done, deactivate the virtual env with `deactivate` from the terminal

<p align="center">
  <img src="assets/app_screenshot.png" width="400px" height="200px" />
</p>
