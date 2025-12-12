### Install requirements:
```bash
pip install -r requirements.txt
```

### Install the meda package:
```bash
pip install -e .
``` 
from the root directory.

### Provide HF token:
add your Hugging Face API token in the .env file:

```bash
HUGGINGFACEHUB_API_TOKEN=your_hugging```
```
Then visit the corresponding [link](https://huggingface.co/google/medgemma-4b-it) to accept the terms of service for the model.
(https://huggingface.co/google/medgemma-4b-it)

### Set up kaggle authentication:
1. Go to your Kaggle account settings and create a new API token. This will download a file named `kaggle.json`.
2. Place the `kaggle.json` file in the `~/.kaggle/` directory (create the directory if it doesn't exist).
3. Make sure the file has the correct permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
``` 

### Download dataset from PhysioNet:
Run the following command to download the dataset:
```bash
kaggle python data_utils/download_ptbxl.py
```
This will download the PTB-XL dataset from PhysioNet and save it in the `data/ptbxl` directory.

### Transform dataset:
Run the following command to transform the dataset:
```bash
python data_utils/load_and_transform_ptbxl.py
```
This will process the raw dataset and save the transformed data in the `data/ptb-xl-processed` directory

### Make the embeddings:
Run the following command to create the embeddings:
```bash
python embeddings/hubert_emb.py
```
This will generate the embeddings and save them in the `data/embeddings` directory.

### Train baseline/model:
Run the following command to train the model:
```bash
python run_baselines.py
```
This will start the training process using the config.py .