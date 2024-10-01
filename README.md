# Smart Autofill
Autofill HTML Tag Detection

## Requirements
First do the following steps to set up the virtual env and install the requirements,

```
cd smart_autofill
python3 -m venv smart_autofill
source smart_autofill/bin/activate
pip install -r requirements.txt
```

## Running the App
```
cd smart_autofill/app
python3 -m streamlit run infer.py
```

The app startup can take a few seconds while the model is loaded. Subsequent loads should be faster.

Once done, deactivate the virtual env with `deactivate` from the terminal

<p align="center">
  <img src="assets/app_screenshot.png" width="500" height="500" />
</p>
