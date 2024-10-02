import pandas as pd

def df_to_dataset(df: pd.DataFrame):
    def fetch_df_by_type(df: pd.DataFrame, filter: str):
        return df[df['ml_dataset'] == filter]

    ds = DatasetDict()
    df_train, df_eval, df_test = (
        fetch_df_by_type('training'),
        fetch_df_by_type('validation'),
        fetch_df_by_type('testing')
    )
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

if __name__ == '__main__':
    df_path = './data/test_dataset.csv'
    df = pd.read_csv(df_path)
    dataset = df_to_dataset(df)
