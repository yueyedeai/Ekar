# Ekar

This is a pytorch implement of Ekar for Knowledge Graph-based Recommendation.

## Requirements

* python3
* pytorch 1.0.1
* CUDA 8.0+ (for GPU)

## Usage

### Process data
For the details of data format, see `data/README.md`. 

run the following command to preprocess the data
```bash
./run.sh <dataset> process <gpu-ID> 
```

`<dataset>` is the name of any dataset folder in the `./data` directory.

### Train models


* Train embedding-based models.
```bash
./run.sh <dataset> <embedding models> <gpu-ID>
```
`<embedding models>` include `conve`, `complex` and `distmult`. 

* Train naive RL model. (policy gradient)
```bash
./run.sh <dataset> rl <gpu-ID>
```

* Train Ekar. (policy gradient + reward shaping)
```bash
./run.sh <dataset> rl.rs <gpu-ID>
```

* Note

    1. you can also directly use `experiment.sh`/`experiment-conve.sh`/`experiment-emb.sh`/`experiment-rs.sh`
    and change some flags to train the models.
    2. To train the RL models using reward shaping, make sure 1) you have pre-trained the embedding-based models and 2) set the file path pointers to the pre-trained embedding-based models correctly (example configuration file).

## Change the hyperparameters

To change the hyperparameters and other experiment set up, start from the configuration files. You can also change the flags on the command of training.

## Acknowledgement

This implementation is based on implementation from salesforce's [MultiHopKG](https://github.com/salesforce/MultiHopKG).