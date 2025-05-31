import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from src.module import preprocess, build_chains, aggregate_chains, encode_and_merge
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def predict_model(df, model_path, features, label):
    print(f'Предсказание: {label}_model')
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    X = df[features]
    df[f'isFraud_{label}'] = model.predict(X)
    df[f'fraud_score_{label}'] = model.predict_proba(X)[:, 1]
    return df

def main(trans_path, id_path, model_transaction_path, model_chain_path, model_final_path, output_path):
    print('Чтение и объединение данных')
    df = pd.read_parquet(trans_path).merge(pd.read_parquet(id_path), how='left', on='TransactionID')
    print(f'Всего строк после объединения: {len(df)}')

    df = preprocess(df)
    df = build_chains(df)
    chain_agg = aggregate_chains(df)
    df = encode_and_merge(df, chain_agg)

    features_transaction = ['TransactionAmt','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
                    'D1','DeviceType','DeviceInfo','ProductCD','card1','card2','card3','card4','card5','card6',
                    'addr1','addr2','P_emaildomain','R_emaildomain',
                    'M1','M2','M3','M4','M5','M6','M7','M8','M9']
    features_chain = ['chain_id','chain_amt_mean','chain_amt_sum','chain_amt_std','chain_trans_count',
                      'chain_D1_mean','chain_D1_sum','chain_D1_std']

    df = predict_model(df, model_transaction_path, features_transaction, 'transaction')
    df = predict_model(df, model_chain_path, features_chain, 'chain')

    df['chain_proba'] = df['fraud_score_chain']
    features_final = features_transaction + ['chain_proba']
    df = predict_model(df, model_final_path, features_final, 'final')

    df[['TransactionID', 'chain_id',
        'isFraud_transaction', 'fraud_score_transaction',
        'isFraud_chain', 'fraud_score_chain',
        'isFraud_final', 'fraud_score_final'
    ]].to_csv(output_path, index=False)

    print(f'Предсказания сохранены в выполнены {output_path}')

    print('Статистика по предсказаниям:')
    total = len(df)
    fraud_pred = df['isFraud_final'].sum()
    legit_pred = total - fraud_pred
    avg_score = df['fraud_score_final'].mean()

    print(f'Всего транзакций: {total:,}')
    print(f'Мошеннических транзакций: {fraud_pred:,}')
    print(f'Легитимных транзакций: {legit_pred:,}')
    print(f'Средняя вероятность мошенничества: {avg_score:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--model_transaction', required=True)
    parser.add_argument('--model_chain', required=True)
    parser.add_argument('--model_final', required=True)
    parser.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()

    main(args.trans, args.id, args.model_transaction, args.model_chain, args.model_final, args.output)
