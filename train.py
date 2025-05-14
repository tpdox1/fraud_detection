import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from src.module import preprocess, build_chains, aggregate_chains, encode_and_merge

import logging
import warnings
logging.getLogger('mlflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_transaction_model(df, features, output_path):
    print('Обучение model_transaction')
    X = df[features]
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    with mlflow.start_run(run_name='model_transaction'):
        params = {
            'max_depth': 10, 'learning_rate': 0.1, 'n_estimators': 4000,
            'subsample': 0.8, 'colsample_bytree': 0.5,
            'missing': -1, 'random_state': 42, 'eval_metric': 'auc'
        }
        mlflow.log_params(params)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        model.save_model(output_path)
        print(f'AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}')
        return model

def train_chain_model(chain_df, features, output_path):
    print('Обучение model_chain')
    X = chain_df[features]
    y = chain_df['isFraud_chain']
    with mlflow.start_run(run_name='model_chain'):
        params = {
            'max_depth': 10, 'learning_rate': 0.1, 'n_estimators': 4000,
            'subsample': 0.8, 'colsample_bytree': 0.5,
            'missing': -1, 'random_state': 42, 'eval_metric': 'auc'
        }
        mlflow.log_params(params)
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = {
            'auc_chain': roc_auc_score(y, y_proba),
            'accuracy_chain': accuracy_score(y, y_pred),
            'precision_chain': precision_score(y, y_pred),
            'recall_chain': recall_score(y, y_pred),
            'f1_chain': f1_score(y, y_pred)
        }

        mlflow.log_metrics(metrics)
        model.save_model(output_path)

        print(f'AUC_chain: {metrics['auc_chain']:.4f}, F1_chain: {metrics['f1_chain']:.4f}, Precision_chain: {metrics['precision_chain']:.4f}, Recall_chain: {metrics['recall_chain']:.4f}')
        return model


def train_final_model(df, features, output_path):
    print('Обучение model_final')
    X = df[features]
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    with mlflow.start_run(run_name='model_final'):
        params = {
            'max_depth': 10, 'learning_rate': 0.1, 'n_estimators': 4000,
            'subsample': 0.8, 'colsample_bytree': 0.5,
            'missing': -1, 'random_state': 42, 'eval_metric': 'auc'
        }
        mlflow.log_params(params)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        model.save_model(output_path)
        print(f'AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}')
        return model


def main(trans_path, id_path, output_dir):
    print('Чтение данных')
    df = pd.read_parquet(trans_path).merge(pd.read_parquet(id_path), on='TransactionID', how='left')
    df = preprocess(df)
    df = build_chains(df)
    chain_df = aggregate_chains(df)
    df = encode_and_merge(df, chain_df)

    os.makedirs(output_dir, exist_ok=True)
    mlflow.set_tracking_uri('file:./mlruns')
    mlflow.set_experiment('fraud_detection_chain')

    features_txn = ['TransactionAmt','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
        'D1','DeviceType','DeviceInfo','ProductCD','card1','card2','card3','card4','card5','card6',
        'addr1','addr2','P_emaildomain','R_emaildomain',
        'M1','M2','M3','M4','M5','M6','M7','M8','M9']
    model_txn = train_transaction_model(df, features_txn, f'{output_dir}/model_transaction.json')

    features_chain = ['chain_id', 'chain_amt_mean','chain_amt_sum','chain_amt_std','chain_trans_count',
    'chain_D1_mean','chain_D1_sum','chain_D1_std']
    model_chain = train_chain_model(chain_df, features_chain, f'{output_dir}/model_chain.json')

    df['chain_proba'] = model_chain.predict_proba(df[features_chain])[:, 1]
    model_final = train_final_model(df, features_txn + ['chain_proba'], f'{output_dir}/model_final.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--output_dir', default='models')
    args = parser.parse_args()
    main(args.trans, args.id, args.output_dir)
