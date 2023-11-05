# Text De-Toxifier

### Author:
##### Yaroslav Sokolov
##### Group: BS21-AI

### [GitHub Repository Link](https://github.com/BuiniyYarik/Text_De-Toxification)

### GitHub Repository Structure:
The repository has the following structure:
```
Text-De-Toxifier
├── README.md               # The top-level README
│
├── data 
│   └── raw                 # The original data
│       └── filtered.tsv    # The dataset (this file is archived)
│
├── models                                  # Trained and serialized models
│   └── bert_for_sequence_classification    # The BERT model
│       ├── config.json                     # The BERT model configuration
│       └── model.safestones                # The BERT model weights (this file is not uploaded to GitHub)
│
├── notebooks                               #  Jupyter notebooks
│   ├── 1.0-initial-data-exporation.ipynb   # The initial data exploration
│   ├── 2.0-model-training.ipynb            # The model training
│   └── 3.0-model-evaluation.ipynb          # The model evaluation          
│
├── reports                                 # Generated analysis in PDF format
│   ├── Solution_Building_Report.pdf        # The solution building report
│   └── Final_Solution_Report.pdf           # The final solution report
│
├── requirements.txt    # The requirements file for reproducing the analysis environment
│                      
└── src                        # Source code for use
    │                 
    ├── data                    
    │   └── make_dataset.py    # The script to download and preprocess the data
    │
    └── models          
        ├── predict_model.py   # The script to make predictions using Text De-Toxifier
        └── train_model.py     # The script to train the model
```

### How to use the repository:
1. Clone the repository:
```
git clone
```
2. Install the requirements:
```
pip install -r requirements.txt
```
3. Download the "bert_for_sequence_classification" model from the [Google Drive](https://drive.google.com/drive/folders/1UOBYQffMZuUY5SEJ2EtbLyvVi3_ZrOJA?usp=sharing) and put it in the "models" directory.

4. Unpack the "filtered.tsv.zip" file in the "data/raw" directory. You can also download the dataset manually from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip).

5. Run the "make_dataset.py" script to preprocess the data:
```
python src/data/make_dataset.py
```
6. Run the "train_model.py" script to train the model:
```
python src/models/train_model.py
```
7. Run the "predict_model.py" script to make predictions using Text De-Toxifier:
```
python src/models/predict_model.py
```
