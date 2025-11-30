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

