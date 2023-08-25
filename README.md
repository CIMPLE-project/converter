# CIMPLE-Project Converter

This repository contains the scripts required to convert CIMPLE data into RDF to be deployed into the [Knowledge Base](https://github.com/CIMPLE-project/knowledge-base).

## How to run

```bash
python update_KG.py -i ../cr_data -o claimreview-kg.ttl -c ./cache
```

## Parameters

* `-q --quiet`: Disable progress bar display
* `-i --input`: Input directory path
* `-o --output`: Output turtle file
* `-c --cache`: Cache directory path
