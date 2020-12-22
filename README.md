# Movie Review
這是一個Kaggle的練習題 , 分辨電影的頻論是好是壞 </br>
Url : https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/submissions  </br>

## 電影的頻論範例 : 
"There's a thin line between likably old-fashioned and fuddy-duddy, and The Count of Monte Cristo ... never quite settles on either side."

0 - negative </br>
1 - somewhat negative </br>
2 - neutral </br>
3 - somewhat positive </br>
4 - positive </br>

## Relation Work
- 理解Bert是什麼 ? [學習網址↗](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)
- 擁有一個 Google 帳號
- 學會使用 Colab [學習網址↗](https://medium.com/@ericsk/%E9%80%8F%E9%81%8E-google-colaboratory-%E5%AD%B8%E7%BF%92%E4%BD%BF%E7%94%A8-python-%E5%81%9A%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%89%E7%A7%91%E5%AD%B8%E8%A8%88%E7%AE%97-9f92c7bb1f50)
- 懂Python 

## Method 

### 開啟Colab並使用GPU

![OpebGPU](https://github.com/4JasonChou/BertTranning-Kaggle-MovieReview/blob/master/ReadmeData/UseGPU.PNG "This is a sample image.")

### 安裝與引用

安裝Pytorch & 確認版本
```
!pip install torch torchvision

import torch
print(torch.__version__)
torch.cuda.is_available()
```

安裝 Bert Transformers
```
!pip install transformers
```

引用
```
import os
import pickle
import torch
import pandas as pd
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F # 激勵函數
```

先做簡單測試,看看Bert Token是否可以正常轉換成Token ( 使用 英文Bert )
```
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokensForScentence = tokenizer.tokenize('There have been 68 recent deaths of residents and nurses from the facility in a small New Jersey town.')
tokenizer.convert_tokens_to_ids(tokensForScentence)

>>>
[2045, 2031, 2042, 6273, 3522, 6677, 1997, 3901, 1998, 11500, 2013, 1996, 4322, 1999, 1037, 2235, 2047, 3933,2237, 1012]

```

### 訓練任務
> 主要分三個步驟 , 資料處理 , 訓練 , 驗證
#### 資料處理 - 打亂 - 切割 ( TrainningSet & TestingSet )
train Data 總共有 156060 </br>
分成 trainingSet : 109242 / testingSet : ~

```
trainPd = pd.read_csv('train.tsv', sep='\t', header=0)
trainPd=trainPd.sample(frac=1.0) 
trainPd=trainPd.reset_index(drop=True)
mTrainingSet = trainPd.iloc[:109242]
mTestingSet = trainPd.iloc[109243:]
```
![table](https://github.com/4JasonChou/BertTranning-Kaggle-MovieReview/blob/master/ReadmeData/table01.PNG "This is a sample image.")



#### 在把資料處理成Bert能接受的格式
- input_ids : [CLS]句子轉換成Tokens[SEP]
- input_segment_ids : segment_ids儲存的是句子的id，id為0就是第一句，id為1就是第二句
- input_masks :  有值的地方都是1
- answer_lables : 答案

在把這些訓練資料做成DataSet 
```
def makeDataset(data_feature):

    input_ids = data_feature['input_ids']
    input_segment_ids = data_feature['input_segment_ids']
    input_masks = data_feature['input_masks']
    answer_lables = data_feature['answer_lables']

    all_input_ids = torch.tensor([input_id for input_id in input_ids], dtype=torch.long)
    all_input_segment_ids = torch.tensor([input_segment_id for input_segment_id in input_segment_ids], dtype=torch.long)
    all_input_masks = torch.tensor([input_mask for input_mask in input_masks], dtype=torch.long)
    all_answer_lables = torch.tensor([answer_lable for answer_lable in answer_lables], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_segment_ids, all_input_masks, all_answer_lables)

    return dataset
```

#### 訓練
這時候經過剛剛的資料整理 , 會有兩個資料集 ( train_dataset & test_dataset )
進行Bert的分類任務Fine-Turn
```
bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
# Set use gpu
device = torch.device("cuda")
train_dataset = makeDataset(qaData_features)
test_dataset = makeDataset(testQaData_features)
train_dataloader = DataLoader(train_dataset ,batch_size=8 ,shuffle=True)
test_dataloader = DataLoader(test_dataset ,batch_size=8 ,shuffle=True)

config = bert_config.from_pretrained('bert-base-uncased',num_labels = 5)
model = bert_class.from_pretrained('bert-base-uncased', from_tf=bool('.ckpt' in 'bert-base-uncased'), config=config)
model.to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
Learning_rate = 5e-6       # 學習率
optimizer = AdamW(optimizer_grouped_parameters, lr=Learning_rate, eps=1e-8)
for epoch in range(2):
    # 訓練模式
    model.train()
    All_train_correct = 0.0
    AllTrainLoss = 0.0
    count = 0
    for batch_dict in train_dataloader:
        batch_dict = tuple(t.to(device) for t in batch_dict)
        
        # From makeDataset : TensorDataset(all_input_ids, all_input_segment_ids, all_input_masks, all_answer_lables)
        # [0] : id
        # [1] : segment id
        # [2] : mask
        # [3] : answer label

        outputs = model(
            batch_dict[0],
            token_type_ids = batch_dict[1],
            attention_mask = batch_dict[2],
            labels = batch_dict[3]
            )
        loss, logits = outputs[:2]
        
        train_correct = compute_accuracy(logits, batch_dict[3])       # 計算正確率
        All_train_correct += train_correct
        AllTrainLoss += loss.item()
        count += 1

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    Average_train_correct = round(All_train_correct/count, 3)
    Average_train_loss = round(AllTrainLoss/count, 3)

    # 測試模式
    model.eval()
    All_test_correct = 0.0
    AllTestLoss = 0.0
    count = 0
    for batch_dict in test_dataloader:
        batch_dict = tuple(t.to(device) for t in batch_dict)

        outputs = model(
            batch_dict[0],
            token_type_ids = batch_dict[1],
            attention_mask = batch_dict[2],
            labels = batch_dict[3]
            )
        loss, logits = outputs[:2]

        test_correct = compute_accuracy(logits, batch_dict[3])       # 計算正確率
        All_test_correct += test_correct
        AllTestLoss += loss.item()

        count += 1
    
    Average_test_correct = round(All_test_correct/count, 3)
    Average_test_loss = round(AllTestLoss/count, 3)

    print('第' + str(epoch+1) + '次' + '訓練模式，loss為:' + str(Average_train_loss) + ' 正確率為' + str(Average_train_correct)+ '，測試模式，loss為:' + str(Average_test_loss) + ' 正確率為' + str(Average_test_correct))

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained('trained_model')
```
![resoftrain](https://github.com/4JasonChou/BertTranning-Kaggle-MovieReview/blob/master/ReadmeData/trainRes.PNG "This is a sample image.")

#### 預測任務

#### Handler Test.tsv
```
testPd = pd.read_csv('test.tsv', sep='\t', header=0)
testPd.head(5)
```

PhraseId | SentenceId | Phrase
-------- | ---------- | ------
1056061  | 8545       | An intermittently pleasing but mostly routine ...
1056062  | 8545       | An intermittently pleasing but mostly 
1056063  | 8545       | An intermittently pleasing but 

#### 轉換成 Bert input embeding
同訓練任務 , 取得一個Phrase
- input_ids
- input_segment_ids
- input_masks

#### 批次預測
```
submitData_dataset = makeDatasetForPredict(submitData_features)
submitData_dataloader = DataLoader(submitData_dataset ,batch_size=16 ,shuffle=False)

submitQuestionNumber = 0;

for batch_dict in submitData_dataloader:
  batch_dict = tuple(t.to(device) for t in batch_dict)
  outputs = model(
      batch_dict[0],
      token_type_ids = batch_dict[1],
      attention_mask = batch_dict[2]
      )
    
  logits = outputs[0]
  for predicts in logits :
    max_val = torch.max(predicts)
    label = (predicts == max_val).nonzero()[0][0]
    ans_label = answer_dic.to_text(label)
    #print(ans_label) #預測答案
    finalQuestionsId.append(questionsId[submitQuestionNumber])
    finalAnswers.append(ans_label)
    submitQuestionNumber = submitQuestionNumber+1 ;

    if( submitQuestionNumber%1000 == 0 ) :
      print("已預測" + str(submitQuestionNumber-1) +  "題,Ans:" + str(ans_label) )   

# Write CSV test
dataframeWriter = pd.DataFrame({'PhraseId':finalQuestionsId,'Ans':finalAnswers})
#將DataFrame儲存為csv,index表示是否顯示行名，default=True
dataframeWriter.to_csv("submitV1.csv",index=False,sep=',')
```

#### 提交Kaggle結果

![submit](https://github.com/4JasonChou/BertTranning-Kaggle-MovieReview/blob/master/ReadmeData/KaggleRes.PNG "This is a sample image.")

#### 檔案說明 : 
 - BertMovieReview : 訓練任務 Code
 - Predcit : 預測任務 Code
 - train.tsv : Kaggle提供的訓練資料
 - test.tsv : Kaggle提供的測驗提交資料
 

