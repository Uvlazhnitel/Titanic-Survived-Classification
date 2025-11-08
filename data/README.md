# Data Directory

Place Titanic dataset files here. Recommended layout:

- `data/train.csv` — training data from Kaggle Titanic competition
- `data/test.csv` — test data from Kaggle Titanic competition (optional)

Large raw data should not be committed. You can fetch via Kaggle:

1. Install Kaggle CLI and place your `kaggle.json` credentials in `~/.kaggle/`.
2. Run `kaggle competitions download -c titanic` and unzip into `data/`.
