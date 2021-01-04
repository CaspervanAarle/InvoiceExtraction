# InvoiceExtraction
code to extract SROIE invoice field data and custom dataset fields. The code is specifically made for research purposes and is not generalizable to more dataset other than the ones that are mentioned here.

## Installation

- Python 3.6
- Tesseract
- GPU is recommended

### Data
Data from the SROIE dataset is publicly available online.

Data from the CUSTOM dataset is available upon request for research purposes only. This dataset is generated, not collected, consists of 1000 medical invoices, but including only 10 different templates




## Usage
The preprocess.py, train.py, and predict.py are files that can be executed

### preprocess
Before training can commence, the data must be preprocessed:

``` python preprocess.py```

to preprocess all data, or define the dataset:

``` python preprocess.py -d CUSTOM```
``` python preprocess.py -d SROIE```

### train

``` python train.py```


### predict

## Contributing
Not meant to contribute

## License
[MIT](https://choosealicense.com/licenses/mit/)
