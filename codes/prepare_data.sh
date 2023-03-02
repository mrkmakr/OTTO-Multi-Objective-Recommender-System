rm -r ./otto/inputs/train_valid
rm -r ./otto_submit/inputs/train_valid

kaggle datasets download -d cdeotte/otto-validation -p ./otto/inputs
unzip ./otto/inputs/otto-validation.zip -d ./otto/inputs/train_valid
kaggle datasets download -d columbia2131/otto-chunk-data-inparquet-format -p ./otto_submit/inputs
unzip ./otto_submit/inputs/otto-chunk-data-inparquet-format.zip -d ./otto_submit/inputs/train_valid
