from autofill_model import AutofillModel
from utils import generate_dataset

def train():
    html_files_path = '~/Downloads/autofill_html_dataset'
    artifact_path = 'artifacts' # saves models to this path
    ds = generate_dataset(html_files_path)
    print(ds)
    model = AutofillModel()
    model.load_data(ds)
    model.train(artifact_path)

if __name__ == '__main__':
    # if re-starting a run, you might have to delete the local model folder
    # e.g. cd src/ && rm -rf tinybert-uncased-autofilll/ && PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python train.py
    train()
