# CIMPLE-Project Converter

This repository contains the scripts required to convert CIMPLE data into RDF to be deployed into the [Knowledge Base](https://github.com/CIMPLE-project/knowledge-base).

## How to run the converter

```bash
python update_KG.py -i ../cr_data -o claimreview-kg.ttl -c ./cache
```

### Parameters

* `-q --quiet`: Disable progress bar display
* `-i --input`: Input directory path
* `-o --output`: Output turtle file
* `-c --cache`: Cache directory path

## How to compute DBpedia labels

1. Download DBpedia labels:

    ```bash
    wget -O all_labels.ttl https://downloads.dbpedia.org/repo/dbpedia/generic/labels/2022.12.01/labels_lang=en.ttl.bz2
    ```

1. Run the processing script:

    ```bash
    python dbpedia_labels.py /path/to/dumps ./all_labels.ttl ./dbpedia_labels.ttl
    ```

    **Parameters:**

    * `-i --input`: Input path to the folder containing converted dumps
    * `-o --output`: Output path to store the Turtle file with labels
    * `-l --labels`: Path to where the DBpedia labels are stored