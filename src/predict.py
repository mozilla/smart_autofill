from autofill_model import AutofillModel, DEFAULT_SAVE_PATH, DEFAULT_FINE_TUNED_MODEL_PATH

USE_LOCAL_MODEL = True # if the model is stored locally
model_path = f'./{DEFAULT_SAVE_PATH}' if USE_LOCAL_MODEL else DEFAULT_FINE_TUNED_MODEL_PATH

def predict(tags):
    model = AutofillModel()
    model.load_model(model_path)
    return model.predict(tags)

if __name__ == '__main__':
    # cd smart_autofill/src
    # run with `python predict.py`
    test_tags = [
        '<input class="new-password" placeholder="add new password..." />',
        '<form><input /> <input /></form>'
    ]
    print(predict(test_tags))
