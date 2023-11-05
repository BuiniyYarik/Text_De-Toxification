from transformers import BertTokenizer, BertForSequenceClassification
import torch


class TextDetoxifier:
    def __init__(self, model_path, tokenizer='bert-base-uncased'):
        """
        Initialize the TextDetoxifier class
        :param model_path: Path to the model
        :param tokenizer: Tokenizer to use
        """
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, texts):
        """
        Predict the toxicity of the input text
        :param texts: Text to predict
        :return: Predicted toxicity of the text
        """
        # Ensure that texts is a list of strings
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Move tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        # Move logits to CPU and convert to numpy for easy handling
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions = ["toxic" if pred == 1 else "non-toxic" for pred in predictions]
        return predictions
