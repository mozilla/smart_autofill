import pandas as pd
import os
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict

NEGATIVE_LABEL = 'other'
seed = 42
STR_LIMIT = 512

def parse_html_to_dataframe(path: str, dataset_type: str) -> pd.DataFrame:
    """
    @param path: folder path of dataset_type (training, validation, testing)
    @dataset_type: fathom dataset (one of 'training', 'validation', 'testing')
    Extracts useful features from html fathom dataset
    fathom label has attribute 'data-fathom="xyz"'
    @return: pandas dataframe of all the features
    """
    features = [
        'file_id',
        'language',
        'name',
        'class',
        'id',
        'maxlen',
        'type',
        'labels', # fathom_types
        'autocomplete_text',
        'placeholder_text',
        'ml_dataset',
        'html_cleaned'
    ]

    features_dict = {f: [] for f in features}
    for filename in os.listdir(f'{path}/{dataset_type}'):
        print(filename)
        language = filename.split('_')[0]
        f = os.path.join(path, dataset_type, filename)
        if not os.path.isfile(f):
            continue
        HtmlFile = open(f, 'r', encoding='utf-8', errors='replace')
        source_code = HtmlFile.read()
        soup = BeautifulSoup(source_code, "html.parser")
        for tag in soup.find_all():
            features_dict['file_id'].append(filename)
            features_dict['language'].append(language)
            features_dict['name'].append(tag.name)
            features_dict['class'].append(tag.attrs.get('class', []))
            features_dict['id'].append(tag.attrs.get('id', ''))
            features_dict['maxlen'].append(tag.attrs.get('maxlength', ''))
            features_dict['type'].append(tag.attrs.get('type', ''))
            features_dict['labels'].append(tag.attrs.get('data-fathom', 'other'))
            features_dict['autocomplete_text'].append(tag.attrs.get('autocomplete', ''))
            features_dict['placeholder_text'].append(tag.attrs.get('placeholder', ''))
            features_dict['ml_dataset'].append(dataset_type)
            features_dict['html_cleaned'].append(str(tag)[:STR_LIMIT])
    return pd.DataFrame(features_dict).fillna('')


def df_to_dataset(df: pd.DataFrame, samples: int) -> Dataset:
    def fetch_df_by_type(df: pd.DataFrame, filter: str, samples: int):
        df_pos = df[(df['ml_dataset'] == filter) & (df['labels'] != NEGATIVE_LABEL)]
        df_neg = df[(df['ml_dataset'] == filter) & (df['labels'] == NEGATIVE_LABEL)].sample(n=samples, random_state=seed)
        return pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True).sample(frac=1)

    ds = DatasetDict()
    df_train, df_eval, df_test = (
        fetch_df_by_type(df, 'training', samples),
        fetch_df_by_type(df, 'validation', samples),
        fetch_df_by_type(df, 'testing', samples)
    )

    print('Saving dataframe to desktop...')
    df_combined = pd.concat([df_train, df_eval, df_test]).reset_index(drop=True)
    df_combined.to_csv('~/Desktop/dataset.csv', index=False)

    dataset_train, dataset_eval, dataset_test = (
        Dataset.from_pandas(df_train),
        Dataset.from_pandas(df_eval),
        Dataset.from_pandas(df_test)
    )
    ds['train'], ds['eval'], ds['test'] = (
        dataset_train,
        dataset_eval,
        dataset_test
    )
    ds['train'] = ds['train'].class_encode_column('labels')
    class_label_feature = ds['train'].features['labels']
    ds['test'] = ds['test'].cast_column('labels', class_label_feature)
    ds['eval'] = ds['eval'].cast_column('labels', class_label_feature)
    return ds

def generate_dataset(path: str, samples: int = 1500) -> Dataset:
    """
    @param path: folder path of dataset_type (training, validation, testing)
    @param sample: number of negative sample to use for training
                   (this is due to large positive vs negative imbalance)
    @return: Dataset
    """
    print('Generating dataframe from HTML files...')
    datasets = ['training', 'validation', 'testing']
    dfs = []
    for ds in datasets:
        dfs.append(parse_html_to_dataframe(path, ds))
    df_combined = pd.concat(dfs, axis=0).reset_index(drop=True)

    print('Generating huggingface dataset from dataframe...')
    dataset = df_to_dataset(df_combined, samples)
    return dataset

if __name__ == '__main__':
    html_files_path = '/Users/Vbaungally/Downloads/autofill_html_dataset'
    ds = generate_dataset(html_files_path)
    print(ds)
