# Variance Dichotomy in Feature Spaces of Facial Recognition Systems is a Weak Defense against Simple Weight Manipulation Attacks

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the **aligned LFW dataset** from [Figshare](https://figshare.com/articles/dataset/lfw-aligned-112x112/27073438?file=49308103) and place it in the following directory:

```
lfw/aligned
```

---

## Usage

### Run Standard Benign Accuracy Experiments

```bash
python main.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--random_seed RANDOM_SEED]
```

### Run Revised Backdoor Method

```bash
python main.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--random_seed RANDOM_SEED] --normalise
```

### Get `eps_delta` Values

```bash
python main.py [--model_name MODEL_NAME] --eps_delta
# Or with normalisation
python main.py [--model_name MODEL_NAME] --eps_delta --normalise
```

### Get PCA Eigenvalues

```bash
python main.py [--model_name MODEL_NAME] --get_eigenvalues
```

### Create CSV Files for with Results

```bash
python create_ba_csv.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--normalise]
python create_asr_csv.py [--model_name MODEL_NAME] [--backdoor_type BACKDOOR_TYPE] [--normalise]
```

### Run Synthetic Data Experiments

```bash
python synthetic.py
```

---

## AdaFace Integration

To use the AdaFace backbone, download the pretrained checkpoint:

- `adaface_ir101_ms1mv2.ckpt` from the [AdaFace GitHub repository](https://github.com/mk-minchul/AdaFace)

Place it in the `pretrained/` directory:

```
pretrained/adaface_ir101_ms1mv2.ckpt
```

---

## Contact

For questions or collaborations, feel free to contact matthew.bowditch@warwick.ac.uk

---
