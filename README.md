# Deep-fake-Forensics-Challenge
Shallow- and Deep- fake Image Manipulation 
Localization Using Deep Learning .
![](./images/network-Recovered.png)

# Shallow- and Deep-Fake Image Manipulation Localization

## Overview

This repository contains the code and documentation for the CS Challenge on Shallow- and Deep-Fake Image Manipulation Localization Using Deep Learning. The challenge involves replicating and enhancing the results of a research paper, drafting a scientific paper, and formulating a startup idea.

You can read our paper from here: [link](https://drive.google.com/file/d/1J-SbeU_iujO9tdgYenbovVpZIYdduaPe/view?usp=sharing).

## Table of Contents

- [Getting Started](##Getting-Started)
- [Usage](##Usage)
- [Dataset](##Dataset)
- [Train/Val/Test](##Train/Val/Test)

## Getting Started

### Follow these steps to get started with the project:
To begin, you will need to clone this GitHub repository :
```bash
git clone https://github.com/Goodnight77/Deep-fake-Forensics-Challenge.git
```
Navigate to the project directory:
```bash 
cd Deep-fake-Forensics-Challenge
```
Install the required packages:
```bash
pip install -r requirements.txt
```
## Usage
### To run the project, use the following command:
```bash
python main.py --n_epoch 50 --n_bs 32 --train_dataframe /path/to/train_data.csv --val_dataframe /path/to/val_data.csv
```
## Dataset
The whole dataset we used to train our model can be downloaded from kaggle using this [link](https://www.kaggle.com/datasets/mohamedbenticha/tsyp-cs-challenge).

## Train/Val/Test
The way (file paths) we split the datasets into train/val/test subsets can be downloaded [link](https://www.dropbox.com/s/opjpz9hoy5xm4um/paths.zip?dl=0).
