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
pip install google-cloud-storage
pip install gcloud
pip install gsutils
```

## Usage

Run this command to start the main script.

```shell script
python main.py [OPTION]...

  -h, --help                           show this help
  -c, --clean                          clean intermediate results before start
  -q, --quiet                          do not log outputs
  -t, --transient                      do not store results
  -d, --dry-run                        only run a limited training to make sure syntax is correct
  -k, --k-folds <kfolds>               number of k-folds
  -e, --epochs <epochs>                number of epochs
  -l, --learning-rate <learning-rate>  learning rate
  -p, --patience <patience>            number of epochs to wait for improvements before finishing training
  -s, --slice-width <slice-width>      number of measurements per slice
  -w, --window-step <window-step>      step size used for sliding window data splitter

Examples:
  python main.py -c -e 3000 -l 0.001
```

## Roadmap

See the [open issues](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics/issues) for a list of proposed features (and
 known issues).
 
## Metrics

<p>
    Confusion matrix<br/>
    <img src="https://raw.githubusercontent.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-model/main/models/latest/05-evaluation/plots/confusion_matrix.png" alt="Confusion Matrix">
</p>

## Contributing

Since this project is part of an ongoing Master's thesis contributions are not possible as for now.

## License

Distributed under the GPLv3 License. See [LICENSE.md](./LICENSE.md) for more information.

## Contact

Florian Schwanz - florian.schwanz@gmail.com

## Acknowledgements

Icon made by Freepik from www.flaticon.com
