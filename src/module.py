import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from hdbscan import HDBSCAN

def get_time_of_day(hour):
    if 0 <= hour < 6:
        return '0'
    elif 6 <= hour < 12:
        return '1'
    elif 12 <= hour < 18:
        return '2'
    else:
        return '3'

def preprocess(df):
    print('Предобработка данных')
    start_date = pd.to_datetime('2017-11-30')
    df['TransactionDT'] = df['TransactionDT'].astype(int)
    df['TransactionDate'] = df['TransactionDT'].apply(lambda x: start_date + timedelta(seconds=x))
    df['Transaction_day'] = df['TransactionDate'].dt.day
    df['Transaction_month'] = df['TransactionDate'].dt.month
    df['Transaction_weekday'] = df['TransactionDate'].dt.weekday
    df['Transaction_hour'] = df['TransactionDate'].dt.hour
    df['Transaction_part_of_day'] = df['Transaction_hour'].apply(get_time_of_day)
    df['user_id'] = df[['card1', 'card2', 'card3', 'card4', 'card5', 'card6']].astype(str).agg('_'.join, axis=1)
    return df

def build_chains(df):
    print('Формирование цепочек')
    df = df.sort_values(by=['user_id', 'TransactionDT'])
    df['chain_id'] = np.nan
    chain_id_counter = 0
    for user_id, group in df.groupby('user_id'):
        if len(group) == 1:
            df.loc[group.index, 'chain_id'] = chain_id_counter
            chain_id_counter += 1
            continue
        X_time = group[['TransactionDT']].astype(float)
        X_scaled = StandardScaler().fit_transform(X_time)
        try:
            clusterer = HDBSCAN(min_cluster_size=2)
            labels = clusterer.fit_predict(X_scaled)
            for label in set(labels):
                if label != -1:
                    df.loc[group.index[labels == label], 'chain_id'] = chain_id_counter
                    chain_id_counter += 1
            df.loc[group.index[labels == -1], 'chain_id'] = chain_id_counter
            chain_id_counter += 1
        except ValueError:
            df.loc[group.index, 'chain_id'] = chain_id_counter
            chain_id_counter += 1
    df['chain_id'] = LabelEncoder().fit_transform(df['chain_id'])
    return df

def aggregate_chains(df):
    print('Подсчёт агрегированных значений внутри цепочек')
    agg = df.groupby('chain_id').agg({
        'TransactionAmt': ['mean', 'sum', 'std'],
        'TransactionID': 'count',
        'D1': ['mean', 'sum', 'std'],
        'isFraud': ['mean', 'sum'] if 'isFraud' in df.columns else {}
    }).reset_index()

    flat_cols = ['chain_id', 'chain_amt_mean', 'chain_amt_sum', 'chain_amt_std',
                 'chain_trans_count', 'chain_D1_mean', 'chain_D1_sum', 'chain_D1_std']
    if 'isFraud' in df.columns:
        flat_cols += ['chain_fraud_rate', 'chain_fraud_sum']
        agg.columns = flat_cols
        agg['isFraud_chain'] = (agg['chain_fraud_sum'] > 0).astype(int)
    else:
        agg.columns = flat_cols
    return agg

def encode_and_merge(df, chain_df):
    print('Кодирование данных')
    df = df.merge(chain_df, on='chain_id', how='left')
    binary_features = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']
    categorical_features = ['DeviceType', 'DeviceInfo', 'ProductCD',
        'card1','card2','card3','card4','card5','card6',
        'addr1','addr2','P_emaildomain','R_emaildomain']
    mapping = {'T': 1, 'F': 0, None: -999, np.nan: -999}
    for col in df.columns:
        if col in categorical_features:
            df[col] = df[col].fillna('missing').astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        if col in binary_features:
            df[col] = df[col].map(mapping)
    return df