- Preprocess train and test:
python scripts/pre_process.py --train_origin_csv ${train csv file} --train_origin ${folder file train} --test_origin ${folder file test}
- train:
python scripts/train.py
- submission:
python scripts/inference.py
- serve:
python scripts/inference.py python scripts/demo --list_file ${folder test file}


