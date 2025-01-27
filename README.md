# small-lm-sliding-windows

This repository contains code for the manuscript (UNDER PEER REVIEW)

## Setup

Our code is built with Python 3.10.x and CUDA 12.x. We recommend to create a conda virtual environment using the `environment.yaml' provided in this repository.

Please contact the dataset owner for data access.

## Folders

The structure of this repo is listed as follows.

```
├── data
│   ├── no-ema
│      ├── sliding_windows
│          ├── 8
│          ├── 16
│          ├── 32
│          ├── 64
│          ├── 128
│      ├── global_ppl_dataset_1.jsonl
│      ├── global_ppl_dataset_2.jsonl
│   ├── other_data.csv
│   ├── baseline_data.csv
├── notebooks
│   ├── global_ppl_dataset_1.ipynb
│   ├── global_ppl_dataset_2.ipynb
│   ├── sliding_windows_ppl_dataset_1.ipynb
│   ├── sliding_windows_ppl_dataset_2.ipynb
├── figs
│   ├── plot_1.pdf
│   ├── plot_2.pdf
├── scripts
│   ├── eval_ppl.py
│   ├── eval_sliding_ppl.py
│   ├── preprocess.py
```

The resulting ppl files from `eval_ppl.py` and `eval_sliding_ppl.py` are saved under `/data/no-ema/global_dataset_x.jsonl` and `/data/no-ema/sliding_windows/`, respectively. The notebooks for generating the plots can be found under `/noteboos/`.