'''
Preprocess the AVH dataset to include participants' data that:
    - has manual transcripts for their audio diary
    - has EMA taken before or after the audio diary within 5 mins
'''
import configparser
import re
import os
import json
from glob import glob
from tqdm import tqdm
import pandas as pd


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


def process_pid(series):
    """
    Extract unique pid from file names in a pandas Series

    Args:
        series (pd.Series): Series containing file names of the transcripts

    Returns:
        pd.Series: Series with processed PIDs as integers
    """
    # Split the strings and get the first element (pid part)
    pids = series.str.split('-').str[0]
    
    # Clean up the pid strings
    pids = (pids.str.rstrip('@avh')
                .str.lstrip('u')
                .str.lstrip('0')
                .astype(int))
    
    return pids


def process_pid_for_file(file_name):
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
        pid = process_pid_for_file(file_name)
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


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    trans_file = "../data/trans.jsonl"
    if os.path.exists(trans_file):
        trans_df = pd.read_json(trans_file, lines=True)
    else:
        get_trans(
            config['DATA']['transcripts'],
            config['DATA']['audio'],
            trans_file)
        trans_df = pd.read_json(trans_file, lines=True)
    trans_df = trans_df[['pid', 'text', 'file']]
    coherence_df = pd.read_csv("/edata/george/310_avh.csv")
    coherence_df = coherence_df[['file', 'numscores']]
    coherence_df['file'] = coherence_df['file'].str.split(".").str[0]
    coherence_df['pid'] = process_pid(coherence_df['file'])
    total_df = pd.merge(trans_df, coherence_df, on=['pid', 'file'])
    total_df.to_json("../data/avh_tald.jsonl", lines=True, orient="records")