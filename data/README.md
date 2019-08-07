# Datasets

## Format

Each line in a data file represents a knowledge graph fact. The representation format is

    entity1 \tab entity2 \tab relation

In general, each dataset folder consists of the following files:
  
    1. train.triples 
        - facts used for training 
    2. dev.triples 
        - facts used for validation
    3. test.triples 
        - facts used for testing
    4. raw.kb 
        - the entire backbone KG.
    5. raw.pgrk
        - undirected, weightless PageRank scores of nodes in the backbone KG calculated using the tool (https://github.com/timothyasp/PageRank) 
    6. raw.csv
        - An undirected representation of raw.kb (extended with reversed edges) which is used as the input file to the PageRank tool. 

* Notes: 

    1. the facts in train.triples/dev.triples/test.triples should be the interactions between user and items.
    2. we should use `raw.kb` to train the knowledge graph embedding method. (Add the `--train_raw_graph` flag. See the commands in `run.sh`)
    3. train.triples/dev.triples/test.triples do not overlap with each other.
