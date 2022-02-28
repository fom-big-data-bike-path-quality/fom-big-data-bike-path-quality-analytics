[![Train model (Dry-Run)](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics/actions/workflows/train-model-dry-run-workflow.yaml/badge.svg)](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics/actions/workflows/train-model-dry-run-workflow.yaml)
[![Issues](https://img.shields.io/github/issues/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics)](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics/issues)

<br />
<p align="center">
  <a href="https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics">
    <img src="./logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Bike Path Quality (Altiplano)</h1>

  <p align="center">
    PyTorch based model that learns from <a href="https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-data">
    bike activity time series data</a>
  </p>
</p>

## About The Project

tbd

### Built With

* [PyTorch](https://pytorch.org/)

## Installation

Initialize the submodules of this repository by running the following commands.

```shell script
git submodule init
git submodule update
```

Install the following dependencies to fulfill the requirements for this project to run.

```shell script
python -m pip install --upgrade pip
pip install flake8 pytest
pip install pandas
pip install matplotlib
pip install sklearn
pip install torch
pip install tqdm
pip install seaborn
pip install telegram-send
pip install gcloud
pip install google-api-core
pip install google-api-tools
pip install google-auth
pip install google-cloud-core
pip install google-cloud-storage
pip install torchviz
```

## Usage

Run this command to start the main script.

```shell script
python main.py [OPTION]...

-h, --help                                         show this help
-c, --clean                                        clean intermediate results before start
-q, --quiet                                        do not log outputs
-t, --transient                                    do not store results
-d, --dry-run                                      only run a limited training to make sure syntax is correct

--skip-data-understanding                          skip data understanding
--skip-validation                                  skip validation

-s, --slice-width <slice-width>                    number of measurements per slice
-w, --window-step <window-step>                    step size used for sliding window data splitter
--down-sampling-factor <down-sampling-factor>      factor by which target classes are capped in comparison to smallest class
-m, --model <model>                                name of the model to use for training
-f, --k-folds <k-folds>                            number of k-folds

-k, --k-nearest-neighbors <k-nearest-neighbors>    number of nearest neighbors to consider in kNN approach
--dtw-subsample-step <dtw-subsample-step>          subsample steps for DTW
--dtw-max-warping-window <dtw-max-warping-window>  max warping window for DTW

-e, --epochs <epochs>                              number of epochs
-p, --patience <patience>                          number of epochs to wait for improvements before finishing training
-l, --learning-rate <learning-rate>                learning rate
--dropout <dropout>                                dropout percentage
--lstm-hidden-dimension <lstm-hidden-dimension>    hidden dimensions in LSTM
--lstm-layer-dimension <lstm-layer-dimension>      layer dimensions in LSTM

Examples:
  python main.py -c -m knn-dtw -k 10 --dtw-subsample-step=1 --dtw-max-warping-window=500
  python main.py -c -m lstm -s 500 -w 500 --lstm-hidden-dimension 128 --lstm-layer-dimension 3
  python main.py -c -m cnn -s 500 -w 500
```

## Roadmap

See the [open issues](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics/issues) for a list of proposed features (and
 known issues).
 
## Metrics

### Configuration 1

* with slice width 500
* step size 500
* overlap 0%

<img src="https://raw.githubusercontent.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-results/main/results/cnn/2022-02-24-12:07:23/05-evaluation/confusion_matrix.png" alt="Confusion Matrix" width="300">  |  <img src="https://raw.githubusercontent.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-results/main/results/lstm/2022-02-25-08:43:39/05-evaluation/confusion_matrix.png" alt="Confusion Matrix" width="300">  |  <img src="https://raw.githubusercontent.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-results/main/results/knn-dtw/2022-02-27-22:02:55/05-evaluation/confusion_matrix_k1.png" alt="Confusion Matrix" width="300">
:-------------------------:|:-------------------------:|:-------------------------:
Confusion matrix CNN | Confusion matrix LSTM | Confusion matrix kNN-DTW with k=1

## Contributing

Since this project is part of an ongoing Master's thesis contributions are not possible as for now.

## License

Distributed under the GPLv3 License. See [LICENSE.md](./LICENSE.md) for more information.

## Contact

Florian Schwanz - florian.schwanz@gmail.com

## Acknowledgements

Icon made by Freepik from www.flaticon.com
