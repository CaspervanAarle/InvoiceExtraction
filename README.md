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
The ```preprocess.py```, ```train.py```, and ```predict.py``` are files that can be executed

### preprocess
Before training can commence, the data must be preprocessed:

``` python preprocess.py```

to preprocess all data, or define the dataset:

``` python preprocess.py -d CUSTOM```
``` python preprocess.py -d SROIE```

### train
After preprocessing, you can train a model over the data. A dataset directory must be given (from the path: \data\preprocessed\):

For example:
``` python train.py CUSTOM_Exact```

You can add a seed to keep the train/test splits constant when training and predicting:

``` python train.py CUSTOM_Exact -s 17071```

When no seed is given, the default seed is chosen.

You can exclude certain templates of the CUSTOM dataset

``` python train.py CUSTOM_Exact -e [t1, t19]```

You may decide to use weights from a previous iteration to start training:

``` python train.py CUSTOM_Exact -cp CUSTOM_Exact\20201222-093425 ```


### predict
After training, a prediction score can be made on the test set to gain insight in the results. Scores are given in Average Precision (AP) and soft Average Precision (softAP), from reference ([Zhao et al.][i33]) and also explained in my report.

``` python predict.py CUSTOM_Exact -cp CUSTOM_Exact\20201222-093425 ```

The excluded templates and the seed must be constant for meaningful results.

A single prediction can also be made:

``` python predict.py t1_9_pdf.pdf -cp CUSTOM_Exact\20201222-093425 ```





## Contributing
Not meant to contribute.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## References
[i33]: https://arxiv.org/abs/1903.12363
