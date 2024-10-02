from autofill_model import AutofillModel


def train():
    # https://huggingface.co/datasets/vazish/autofill_dataset
    dataset_path = 'vazish/autofill_dataset'
    artifact_path = './artifacts'
    model = AutofillModel()
    model.load_data(dataset_path)
    model.train(artifact_path)

if __name__ == '__main__':
    # if re-starting a run, you might have to delete the local model folder
    # e.g. rm -rf google/ && python train.py
    train()
