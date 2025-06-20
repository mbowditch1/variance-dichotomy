Install the requirements with `pip install -r requirements.txt`
Download aligned LFW dataset from https://figshare.com/articles/dataset/lfw-aligned-112x112/27073438?file=49308103 and place in lfw/aligned

## Usage
To run the standard benign accuracy experiments, use the following command:
python main.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--random_seed RANDOM_SEED]

For the modified method run:
python main.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--random_seed RANDOM_SEED] --normalise

To get eps_delta values, run:
python main.py [--model_name MODEL_NAME]  --eps_delta
or
python main.py [--model_name MODEL_NAME]  --eps_delta --normalise

To get PCA eigenvalues, run:
python main.py [--model_name MODEL_NAME]  --get_eigenvalues

To create csv files for the results, run:
python create_ba_csv.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--normalise]
python create_asr_csv.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--normalise]

For synthetic data, run:
python synthetic.py

To run results with adaface, download "adaface_ir101_ms1mv2.ckpt" from https://github.com/mk-minchul/AdaFace to "pretrained" folder.
