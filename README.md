# EnsembleLearning4RatingPrediction

Forked from [https://github.com/wangyurzee7/sklearn_worker](https://github.com/wangyurzee7/sklearn_worker)

**(This is my project! I'm not copy cat!)**

## Requirements

```
sklearn
jieba
pandas
```

## How to run?

```
$ python3 main.py --help
usage: main.py [-h] --config CONFIG [--train] [--test]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Config file (Json format)
  --train               Train?
  --test                Test?
```

Noting that if you use neither `--train` nor `--test`, the default behavior of the script is to run both of them.

## How to reproduce the experiment

Ensure that the following files exists:

```
../data/train.csv
../data/valid.csv
../data/test.csv
```

Then, run:

```bash
./run_all.sh
```

PS: It will take a very very very very long time to run all the experiments (about 30 hours on my Ali Cloud Service). Please be patient.

After that, if you want to see all the experimental results, please:

```bash
./cat_all.sh
```