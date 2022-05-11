This is homework of capstone project RS School Machine Learning course.

This project uses  Forest train dataset. To download follow the link https://www.kaggle.com/competitions/forest-cover-type-prediction

## Usage
This package allows you to train model of Logistic Regression.

Clone this repository to your machine.
Download Forest train dataset, save csv locally (default path is data/train.csv in repository's root).
Make sure Python 3.9 and Poetry are installed on your machine.
Install the project dependencies (run this and following commands in a terminal, from the root of a cloned repository):
```
poetry install
```
Run train with the following command:
```
poetry run train -d <path to csv with data> -s <path to save trained model>
```
To get a full list of options, use help:

```
poetry run train --help
```
