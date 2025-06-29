from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils_tool import get_folder_path
from utils_tool import timer
import pandas as pd
import torch

data_path = get_folder_path('.', 'data')
labels_file_name = 'emotion_labels_en.csv'
labels_file_path = f'{data_path}/{labels_file_name}'

df = pd.read_csv(labels_file_path)
df = df.dropna()
label2id = {label: i for i, label in enumerate(df['label'].unique())}
df['label_id'] = df['label'].map(label2id)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['sentence'].tolist(),
    df['label_id'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label_id']
)

saved_model_name = 'emotion_model'
saved_model_path = f'{data_path}/emotion_model'

report_path = f'{data_path}/bert_emotion_report.txt'

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

@timer
def model_train():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    max_epoch = 3
    for epoch in range(max_epoch):
        for batch_idx, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{max_epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    model.save_pretrained(saved_model_path)
    tokenizer.save_pretrained(saved_model_path)
    print(f'Model and tokenizer saved to {saved_model_path}')

@timer
def model_eval():
    model, tokenizer, device = load_model()
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=32)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=label2id.keys()))

    report = classification_report(all_labels, all_preds, target_names=label2id.keys())
    print(report)

    with open(report_path, 'w') as f:
        f.write(report)

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(saved_model_path)
    model = BertForSequenceClassification.from_pretrained(saved_model_path)
    model.to(device)
    model.eval()
    print(f'Model and tokenizer loaded from {saved_model_path}')
    return model, tokenizer, device
        
        
if __name__ == '__main__':
    # model_train()
    model_eval()