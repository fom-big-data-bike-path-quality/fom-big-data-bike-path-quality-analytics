[![Train model](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics/actions/workflows/train-model-workflow.yaml/badge.svg)](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics/actions/workflows/train-model-workflow.yaml)
[![Issues](https://img.shields.io/github/issues/florianschwanz/fom-big-data-bike-path-quality-analytics)](https://github.com/florianschwanz/fom-big-data-bike-path-quality-analytics/issues)

<br />
<p align="center">
  <a href="https://github.com/florianschwanz/fom-big-data-bike-path-quality-analytics">
    <img src="./logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Bike Path Quality (Altiplano)</h1>

  <p align="center">
    PyTorch based model that learns from bike activity time series data
  </p>
</p>

## About The Project

tbd

### Built With

* [PyTorch](https://pytorch.org/)

## Installation

Initialize the submodules of this repository by running the following command.

```shell script
git pull --recurse-submodules
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
pip install telegram_send
```

## Usage

Run this command to start the main script.

```shell script
python main.py [OPTION]...

  -h, --help                          show help for this script
  -c, --clean                         clean all results before generating new ones
  -d, --dry-run                       perform a dry-run to verify everything is set up correctly
  -e, --epochs <epochs>               number of epochs in the training loop
  -l, --learningrate <learningrate>   learning rate used for training

Examples:
  python main.py -c -e 3000 -l 0.001
```

## Roadmap

See the [open issues](https://github.com/florianschwanz/fom-big-data-bike-path-quality-analytics/issues) for a list of proposed features (and
 known issues).

## Contributing

Since this project is part of an ongoing Master's thesis contributions are not possible as for now.

## License

Distributed under the GPLv3 License. See [LICENSE.md](./LICENSE.md) for more information.

## Contact

Florian Schwanz - florian.schwanz@gmail.com

## Acknowledgements

Icon made by Freepik from www.flaticon.com
