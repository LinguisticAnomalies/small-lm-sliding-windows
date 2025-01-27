'''
Preprocess the AVH dataset to include participants' data that:
    - has manual transcripts for their audio diary
    - has EMA taken before or after the audio diary within 5 mins
'''
import argparse
import configparser
import re
import os
import json
from datetime import timedelta
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ema", action="store_true",
        help="""if set, then limit the records when there are matched EMA"""
    )
    return parser.parse_args()


def clean_tran(txtin):
    """
    transcript preprocessing, which
        - remove all texts in [] // () <>
        - remove all characters that are not letters, digits,
                whitespaces, sigine quotes, or hyphens with an empty string

    :param txtin: the input transcript
    :type txtin: str
    :return: the cleaned transcript
    :rtype: str
    """
    # set an "alert" for background noise and other "disturbing noise"
    indicator = 0
    # search for matches ANYWHERE in the string
    if re.search(r'\[.*?\].?', txtin):
        txtin = re.sub(r'\[.*?\].?', "", txtin)
        indicator = 1
    txtout = re.sub(r"\_+", " ", txtin)
    txtout = re.sub(r"\n", "", txtout)
    txtout = re.sub(r"\s+", " ", txtout)
    # non ACSII characters but keep puctuation
    txtout = re.sub(r"[^A-Za-z0-9\s!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", "", txtout)
    txtout = txtout.strip()
    return txtout, indicator


def process_pid(file_name):
    """
    get unique pid from file name

    Args:
        file_name (str): the file name of the transcripts

    Returns:
        list: a list with pid, recording time, and recording frequency
    """
    pid = file_name.split("-")
    # get pid
    pid[0] = pid[0].rstrip("@avh")
    pid[0] = pid[0].lstrip("u")
    pid[0] = pid[0].lstrip("0")
    pid[0] = int(pid[0])
    return pid


def process_uid(df):
    """
    get uid from metadata

    Args:
        df (pd.DataFrame): a dataframe with metadata

    Returns:
        pd.Series: unique pid
    """
    pid = df['uid'].str.split("@").str[0]
    pid = pid.str.lstrip("u0")
    return pd.to_numeric(pid, errors='coerce').astype('Int64')


def check_ema():
    """
    get EMA data where AVH-Occurrence = Yes, and reformat the answer to scales

    Returns:
        pd.DataFrame: a subset of EMA data
    """
    df = pd.read_csv("../data/ema_full.csv")
    scale_map = {
        "Skip": 0,
        "Not at all": 1,
        "A little": 2,
        "Moderately": 3,
        "A lot": 4,
        "Extremely": 4,
        "Yes": 1,
        "No": 0,
    }
    col_excluded = ['Appraisal-Externality', 'uid',
                    'resp_time', '__user_triggered__']
    for col in df.columns:
        if col not in col_excluded:
            df[col] = df[col].apply(
                lambda x: scale_map.get(x)
            )
    df = df.loc[df['AVH-Occurrence'] == 1]
    df['pid'] = df['uid'].str.split("@").str[0]
    df['pid'] = df['pid'].str.lstrip("u")
    df['pid'] = df['pid'].str.lstrip("0")
    df['pid'] = df['pid'].astype(int)
    # reformat date
    df['resp_time'] = pd.to_datetime(df['resp_time'])
    df['time'] = df['resp_time'].dt.strftime("%Y%m%d")
    df['time'] = df['time'].astype(int)
    df['ema_stamp'] = df['resp_time'].dt.strftime("%H:%M:%S")
    print(f"unique pid for EMA: {len(df['pid'].unique())}")
    print(f"# of EMA: {df.shape}")
    return df


def create_file_name(df):
    """
    generate the audio recording file name given the recording timestamp

    Args:
        df (pd.DataFrame): the dataframe containing timestamps when recording audio diaries

    Returns:
        pd.DataFrame: 
    """
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d')
    df = df.sort_values(['time', 'stamp'])
    df['counter'] = df.groupby(['uid', 'time']).cumcount() + 1
    df['file'] = df['uid'] + '-' + df['time'].dt.strftime('%Y%m%d') + \
        '-' + df['counter'].astype(str)
    return df


def get_audio_timestamp():
    """
    get timestamp for audio diaries

    Returns:
        pd.DataFrame: the dataframe containing timestamp and converted file names
    """
    df = pd.read_csv("../data/audio_time.csv", header=None)
    df.columns = ['uid', 'stamp']
    df['pid'] = process_uid(df)
    df['stamp'] =  pd.to_datetime(df['stamp'])
    df['time'] = df['stamp'].dt.strftime("%Y%m%d")
    # ts_df['time'] = ts_df['time'].astype(int)
    df['stamp'] = df['stamp'].dt.strftime("%H:%M:%S")
    df = create_file_name(df)
    df = df[['pid', 'time', 'stamp', 'file']]
    return df


def get_trans(intpu_path, audio_path, out_file):
    """
    preprocess the manual transcripts

    Args:
        intpu_path (str): the path to all mannual transcripts
        audio_path (str): the path to all audio recordings
        out_file (str): the file name for saving transcripts as .jsonl file
    """
    tran_list = []
    files = glob(f"{intpu_path}/*.txt")
    audio_files = glob(f"{audio_path}/*.mp3")
    for tran_file in tqdm(files, total=len(files)):
        file_name = os.path.basename(tran_file).split(".")[0]
        pid = process_pid(file_name)
        # u00002294@avh-201901003-1
        # u00002294@avh-201901006-1
        # u00001261@avh-201812120-1
        if file_name == "u00002294@avh-201901006-1":
            file_name = "u00002294@avh-20191006-1"
        if file_name == "u00002294@avh-201901003-1":
            file_name = "u00002294@avh-20191003-1"
        if file_name == "u00001261@avh-201812120-1":
            file_name = "u00001261@avh-20181220-1"
        audio_file = [item for item in audio_files if \
               file_name == os.path.basename(item).split(".")[0]]
        if audio_file:
            pass
        with open(tran_file, "rb") as tran_file:
            raw_data = tran_file.read()
            try:
                # First try to decode with cp1252
                tran_content = raw_data.decode('cp1252', errors='replace')
                # Then convert to utf-8
                tran_content = tran_content.encode('utf-8').decode('utf-8')
                tran_content, indicator = clean_tran(tran_content)
            except UnicodeDecodeError:
                # If cp1252 fails, try utf-8 directly
                tran_content = raw_data.decode('utf-8', errors='replace')
                tran_content, indicator = clean_tran(tran_content)
        if tran_content:
            tran_list.append(
                {"pid": pid[0],
                "time": pid[1],
                "record": pid[2],
                "text": tran_content,
                "file": file_name,
                "audio": audio_file[0],
                "noise": indicator}
            )
        else:
            pass
    with open(out_file, "w") as json_file:
        for item in tran_list:
            json.dump(item, json_file)
            json_file.write("\n")


def limit_ema(df):
    """
    limite the data where audio diary and EMA happened within 5 mins

    Args:
        df (pd.DataFrame): the dataframe containing timestamps for audio diaries and EMAs

    Returns:
        pd.DataFrame: _description_
    """
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d')
    df['stamp'] = pd.to_timedelta(df['stamp'])
    df['ema_stamp'] = pd.to_timedelta(df['ema_stamp'])
    df['time_diff'] = df['ema_stamp'] - df['stamp']
    # within 5 mins
    df = df[df['time_diff'].abs() <= timedelta(minutes=5)]
    df['ema_timing'] = np.where(df['time_diff'] < timedelta(0), 'before', 'after')
    return df

if __name__ == "__main__":
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    ts_df = get_audio_timestamp()
    get_trans(
        config['DATA']['transcripts'],
        config['DATA']['audio'],
        "../data/trans.jsonl")
    # get transcripts
    trans_df = pd.read_json("../data/trans.jsonl", lines=True)
    # add audio diary timestamp
    trans_df = trans_df.merge(ts_df, on=['file', 'pid'])
    trans_df.rename(columns={"time_x": "time"}, inplace=True)
    trans_df = trans_df[['pid', 'time', 'text', 'file', 'stamp', 'audio']]
    # baseline
    baseline = pd.read_csv("../data/baseline.csv")
    baseline = baseline[['account_id', "age", "gender", 
                         "race", "hpsvq-total-score"]]
    baseline['label'] = np.where(baseline['hpsvq-total-score'] <=25, 0, 1)
    baseline.rename(columns={"account_id": "pid"}, inplace=True)
    total_df = trans_df.merge(baseline, on="pid")
    # coherence score
    coherence_df = pd.read_pickle("../data/joined300.pkl")
    coherence_df = coherence_df[['file', 'numscores', 'n_numscores']]
    coherence_df['file'] = coherence_df['file'].str.split(".").str[0]
    total_df = total_df.merge(coherence_df, on='file')
    print(total_df.shape)
    print(len(total_df['pid'].unique()))
    total_df.to_json("../data/trans_baseline.jsonl", lines=True, orient='records')
    if pargs.ema:
        # get ema
        ema_df = check_ema()
        # merged with data
        total_df = total_df.merge(ema_df, on=['pid', 'time'])
        # data entries within 5 min
        total_df = limit_ema(total_df)
        print(len(total_df['pid'].unique()))
        print(total_df.shape)
        total_df.to_json("../data/manual_merged_ema.jsonl", lines=True, orient='records')
    else:
        pass
