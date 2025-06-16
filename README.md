# CIMPLE-Project Converter

This repository contains the scripts required to convert CIMPLE data into RDF to be deployed into the [Knowledge Base](https://github.com/CIMPLE-project/knowledge-base).

## Requirements

- Python 3.10

## Installation

```bash
pip install -r requirements.txt
```

## How to run the converter

```bash
python update_KG.py -i ../cr_data -o claimreview-kg.ttl -c ./cache
```

### Parameters

- `-i --input`: Input folder
- `-o --output`: Output RDF file for the new triples
- `-g --graph`: Path to the existing RDF graph file
- `-f --format`: Output RDF format (default: nt)
- `-m --models`: Models folder (default: /data/cimple-factors-models)
- `--device`: Device for PyTorch models (e.g., 'cpu', 'cuda', 'cuda:0') (default: auto)
- `--batch_size`: Batch size for factor computation
- `--no-progress`: Hide progress bars

## How to compute DBpedia labels

1. Download DBpedia labels:

   ```bash
   wget -O en_labels.ttl https://downloads.dbpedia.org/repo/dbpedia/generic/labels/2022.12.01/labels_lang=en.ttl.bz2
   ```

1. Run the processing script:

   ```bash
   python dbpedia_labels.py -i /path/to/dumps -l ./en_labels.ttl -o ./dbpedia_labels.ttl
   ```

   **Parameters:**

   - `-i --input`: Input path to the folder containing converted dumps
   - `-o --output`: Output path to store the Turtle file with labels
   - `-l --labels`: Path to where the DBpedia labels are stored
