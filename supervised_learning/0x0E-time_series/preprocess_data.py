#!/usr/bin/env python3
""" Data preprocessing to forecast"""
import pandas as pd


def preprocessor(file, days):
    """
    Data cleaning from a csv file
    :param file: path to the file
    :param days: days in file
    :return: train, test and validation dataframes
    """
    df_data = pd.read_csv(file)
    df_data = df_data.iloc[-days * 24 * 60:]
    df_data.pop("Volume_(Currency)")
    df_data.pop("Volume_(BTC)")
    df_data.pop("High")
    df_data.pop("Low")
    df_data['Date'] = pd.to_datetime(df_data['Timestamp'], unit='s')
    df_data = df_data[df_data["Date"] >= "2017"]
    df_data = df_data.set_index('Date')
    df_data = df_data.drop_duplicates(subset="Timestamp")
    df_data = df_data.interpolate()

    n = df_data.shape[0]
    train_df = df_data[0:int(n * 0.7)]
    val_df = df_data[int(n * 0.7):int(n * 0.9)]
    test_df = df_data[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
