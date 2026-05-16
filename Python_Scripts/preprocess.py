import pandas as pd
import glob
import numpy as np
from datetime import datetime

def load_data(path):
    all_files = glob.glob(path + "/*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[()\-\/]", "", regex=True)
        )
    return df

def haversine_distance(lat1,lon1,lat2,lon2):
  earth_radius = 6371
  lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])

  dlat = lat2 - lat1
  dlon = lon2 - lon1
    
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  return earth_radius * 2 * np.arcsin(np.sqrt(a))

def featur_engineering(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_late_night'] = df['hour'].between(1, 4).astype(int)

    df['log_amt'] = np.log1p(df['amt'])

    df['amt_zscore'] = (df['amt'] - df['amt'].mean()) / df['amt'].std()

    df['is_high_amt'] = (df['amt'] > df['amt'].quantile(0.95)).astype(int)

    df['distance_from_home'] = haversine_distance(
       df['lat'],
        df['long'],
        df['merch_lat'],
        df['merch_long']
        )
    
    df['log_distance'] = np.log1p(df['distance_from_home'])

    df['dob'] = pd.to_datetime(df['dob'],format='%Y-%m-%d')
    
    today = pd.Timestamp('today')
    
    df['customer_age'] = ((today - df['dob']).dt.days / 365.25).astype(int)
    
    df['age_group'] = pd.cut(df['customer_age'], 
                          bins=[0, 25, 35, 50, 65, 100],
                          labels=['18-25', '26-35', '36-50', '51-65', '65+'])
    df['age_group_encoded'] = df['age_group'].cat.codes

    df['category_encoded'] = df.groupby('category')['is_fraud'].transform('mean')

    global_fraud_rate = df['is_fraud'].mean()
    smoothing = 10
    
    city_stats = df.groupby('city').agg(
       fraud_rate=('is_fraud', 'mean'),
        txn_count=('is_fraud', 'count')
        ).reset_index()
    
    city_stats['city_encoded'] = (
       (city_stats['fraud_rate'] * city_stats['txn_count'] + global_fraud_rate * smoothing) /
        (city_stats['txn_count'] + smoothing))
    
    city_map = city_stats.set_index('city')['city_encoded']
    df['city_encoded'] = df['city'].map(city_map)
    df['city_encoded'] = df['city_encoded'].fillna(global_fraud_rate)
    
    state_fraud_rate = df.groupby('state')['is_fraud'].mean()
    df['state_encoded'] = df['state'].map(state_fraud_rate)
    df['state_encoded'] = df['state_encoded'].fillna(global_fraud_rate)

    global_fraud_rate = df['is_fraud'].mean()

    merchant_stats = df.groupby('merchant').agg(
       fraud_rate=('is_fraud', 'mean'),
       txn_count=('is_fraud', 'count')
       ).reset_index()
    
    merchant_stats['merchant_encoded'] = (
       (merchant_stats['fraud_rate'] * merchant_stats['txn_count'] + global_fraud_rate * smoothing) /
       (merchant_stats['txn_count'] + smoothing))
       
    merchant_map = merchant_stats.set_index('merchant')['merchant_encoded']
    df['merchant_encoded'] = df['merchant'].map(merchant_map)
    df['merchant_encoded'] = df['merchant_encoded'].fillna(global_fraud_rate)

    job_stats = df.groupby('job').agg(
       fraud_rate=('is_fraud', 'mean'),
       txn_count=('is_fraud', 'count')
       ).reset_index()
    
    job_stats['job_encoded'] = (
       (job_stats['fraud_rate'] * job_stats['txn_count'] + global_fraud_rate * smoothing) /
       (job_stats['txn_count'] + smoothing))
       
    job_map = job_stats.set_index('job')['job_encoded']
    df['job_encoded'] = df['job'].map(job_map)
    df['job_encoded'] = df['job_encoded'].fillna(global_fraud_rate)

    return df


def delete_unnecessary_columns(df):
    columsn_to_drop = ['age_group', 'trans_date_trans_time', 'dob', 'category', 'city', 'state', 'trans_num', 'job', 'merchant']
    df.drop(columns = columsn_to_drop, inplace=True)
    return df

def save_preprocessed_data(df, path):
    df.to_csv(path, index=False)


def complete_preprocessing_pipeline(input_path, output_path):
    df = load_data(input_path)
    df = featur_engineering(df)
    df = delete_unnecessary_columns(df)
    save_preprocessed_data(df, output_path)


if __name__ == "__main__":
    input_path = '..\Data Files\Raw Files'
    output_path = '..\Data Files\Processed Files\preprocessed_data.csv'
    complete_preprocessing_pipeline(input_path, output_path)