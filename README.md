# Emotion Classification with BERT

This project demonstrates fine-tuning a BERT model to classify English sentences into one of four emotional categories: **confidence**, **excite**, **inspire**, and **pleasure**.

## Features

- Uses Hugging Face's `transformers` and PyTorch
- Custom `EmotionDataset` class for handling text-label pairs
- Supports training, saving, loading, and evaluation
- Automatically saves a classification report to `.txt`

## Dataset Format

The dataset should be a CSV file (`emotion_labels_en.csv`) with the following format:

```csv
sentence,label
"I believe in myself and work hard to achieve success.",confidence
"Let's go! We can do it!",excite
...

```bash
pip install torch transformers scikit-learn pandas
```

Train and Save Model

```bash
python main.py
```

The classification report will be printed and saved to:
```bash
./data/bert_emotion_report.txt
```

## 📊 Example Classification Report

| Label      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| confidence | 0.96      | 0.98   | 0.97     | 104     |
| excite     | 0.99      | 1.00   | 1.00     | 102     |
| inspire    | 0.99      | 0.97   | 0.98     | 104     |
| pleasure   | 0.98      | 0.97   | 0.98     | 104     |
| **Accuracy** |           |        | **0.98** | 414     |
| **Macro avg** | 0.98   | 0.98   | 0.98     | 414     |
| **Weighted avg** | 0.98 | 0.98 | 0.98     | 414     |


MIT License

Copyright (c) 2025 [WANG HSIN HAO]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      
copies of the Software, and to permit persons to whom the Software is         
furnished to do so, subject to the following conditions:                       

The above copyright notice and this permission notice shall be included in    
all copies or substantial portions of the Software.                           

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN     
THE SOFTWARE.
