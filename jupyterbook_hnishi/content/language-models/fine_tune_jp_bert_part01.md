<a href="https://colab.research.google.com/github/hnishi/handson-language-models/blob/main/fine_tune_jp_bert_part01.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# huggingface transformers を使って日本語 BERT モデルをファインチューニングして感情分析 (with google colab) part01

本記事では、日本語 BERT モデルをファインチューニングして感情分析する方法を解説します。

BERT の詳細な解説は、この記事のスコープ外とします。

この記事は、part01 です。

[part02](https://github.com/hnishi/handson-language-models/blob/main/fine_tune_jp_bert_part02.ipynb) では、まとまったデータセットを使って実際に学習と評価を行っています。

すべての記事の目次は以下をご参照ください。

https://github.com/hnishi/handson-language-models/blob/main/README.md

## 参考

- [huggingface transformers ドキュメント](https://huggingface.co/transformers/)
- [BERT 論文](https://arxiv.org/abs/1810.04805)
- [Fine-tuning a BERT model with transformers](https://towardsdatascience.com/fine-tuning-a-bert-model-with-transformers-c8e49c4e008b)

## 必要なライブラリのインストール


```python
!pip install -q transformers
```

    [K     |████████████████████████████████| 1.9MB 5.4MB/s 
    [K     |████████████████████████████████| 890kB 23.9MB/s 
    [K     |████████████████████████████████| 3.2MB 25.4MB/s 
    [?25h  Building wheel for sacremoses (setup.py) ... [?25l[?25hdone



```python
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import AdamW
```

## 日本語 BERT の簡単なチュートリアル

最初に、huggingface transformers を使った日本語 BERT pre-trained model の使い方や fine tuning の方法を、簡単に見ていくことにします。

今回試す事前学習済みモデルとしては、 bert-large-japanese　を利用してみたいと思います。

---

**補足**

最近 (2021-03-05)、東北大学のグループから BERT の日本語 pre-trained models の [version 2](https://github.com/cl-tohoku/bert-japanese/releases/tag/v2.0) が公開されています。

このリリースでは、より大きなアーキテクチャの bert-large-japanese モデルも公開されています。

- cl-tohoku/bert-base-japanese-v2
- cl-tohoku/bert-large-japanese

```
BERT-base models consist of 12 layers, 768 dimensions of hidden states, and 12 attention heads.
BERT-large models consist of 24 layers, 1024 dimensions of hidden states, and 16 attention heads.
```

- https://huggingface.co/cl-tohoku
- https://github.com/cl-tohoku/bert-japanese

### Pre-trained Model を使って推論

BERT なので、mask された token を予測するように学習されています。

したがって、pre-trained model を使って、文章中の穴埋め (文章中の欠損箇所の予測) を行うことができます。

以下の２種類のモデルを使って推論してみました。

- cl-tohoku/bert-large-japanese
- bert-base-multilingual-uncased (BERT の多言語モデル)

結果、 `bert-base-multilingual-uncased` のほうが自然な文章となりました。

この違いは、学習に使用されたデータセットに依存しているのかと思いましたが、どちらのモデルも wikipedia をデータセットとして利用しているので、謎です。


```python
model_name = "cl-tohoku/bert-large-japanese"

unmasker = pipeline('fill-mask', model=model_name)
unmasker("こんにちは、私は[MASK]モデルです。")
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=336.0, style=ProgressStyle(description_…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1354281605.0, style=ProgressStyle(descr…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=236001.0, style=ProgressStyle(descripti…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=174.0, style=ProgressStyle(description_…


    


    Some weights of the model checkpoint at cl-tohoku/bert-large-japanese were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).





    [{'score': 0.13797800242900848,
      'sequence': 'こんにちは 、 私 は お モデルです 。',
      'token': 860,
      'token_str': 'お'},
     {'score': 0.09143581241369247,
      'sequence': 'こんにちは 、 私 は う モデルです 。',
      'token': 856,
      'token_str': 'う'},
     {'score': 0.03621169179677963,
      'sequence': 'こんにちは 、 私 は こう モデルです 。',
      'token': 11668,
      'token_str': 'こう'},
     {'score': 0.028521882370114326,
      'sequence': 'こんにちは 、 私 は 、 モデルです 。',
      'token': 828,
      'token_str': '、'},
     {'score': 0.02647402137517929,
      'sequence': 'こんにちは 、 私 は いら モデルです 。',
      'token': 24523,
      'token_str': 'いら'}]




```python
model_name = "bert-base-multilingual-uncased"

unmasker = pipeline('fill-mask', model=model_name)
unmasker("こんにちは、私は[MASK]モデルです。")
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=625.0, style=ProgressStyle(description_…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=672271273.0, style=ProgressStyle(descri…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=871891.0, style=ProgressStyle(descripti…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1715180.0, style=ProgressStyle(descript…


    


    Some weights of the model checkpoint at bert-base-multilingual-uncased were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).





    [{'score': 0.21041248738765717,
      'sequence': 'こんにちは 、 私 は 、 モテルてす 。',
      'token': 1482,
      'token_str': '、'},
     {'score': 0.12128563225269318,
      'sequence': 'こんにちは 、 私 は 元 モテルてす 。',
      'token': 2051,
      'token_str': '元'},
     {'score': 0.033908627927303314,
      'sequence': 'こんにちは 、 私 は 初 モテルてす 。',
      'token': 2178,
      'token_str': '初'},
     {'score': 0.029899440705776215,
      'sequence': 'こんにちは 、 私 は 女 モテルてす 。',
      'token': 3014,
      'token_str': '女'},
     {'score': 0.021887250244617462,
      'sequence': 'こんにちは 、 私 は 男 モテルてす 。',
      'token': 5846,
      'token_str': '男'}]



### テキスト分類のための Fine Tuning の手順

以下、簡易的に 3 種類のラベル (positive: 2, neutral: 1, negative: 0) のデータを使って fine tuning を行います。


```python
# 確認用のデータセット
df = pd.DataFrame([{"text": "私はこの映画をみることができて、とても嬉しい。", "label": 2},
                              {"text": "今日の晩御飯は何だろう。", "label": 1},
                              {"text": "猫に足を噛まれて痛い。", "label": 0}])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>私はこの映画をみることができて、とても嬉しい。</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>今日の晩御飯は何だろう。</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>猫に足を噛まれて痛い。</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_docs = df["text"].tolist()
train_labels = df["label"].tolist()
```

### モデルのダウンロード (キャッシュがない場合) と読み込み

同時にダウンロードされるトークナイザーを利用して、データセットの text の encoding を行います。


```python
# Ref: https://huggingface.co/transformers/training.html#pytorch

model_name = "cl-tohoku/bert-large-japanese"

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_name)
```

    Some weights of the model checkpoint at cl-tohoku/bert-large-japanese were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at cl-tohoku/bert-large-japanese and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
encodings = tokenizer(train_docs, return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
```


```python
# Fine-tuning in native PyTorch

# > the AdamW() optimizer which implements gradient bias correction as well as weight decay.
optimizer = AdamW(model.parameters(), lr=1e-5)

labels = torch.tensor(train_labels).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
```

### Fine Tune したモデルで推論

学習に使ったデータを入力して推論してみます。


```python
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=model_name)
```


```python
sentiment_analyzer("これは、テストのための文章です")
```




    [{'label': 'LABEL_0', 'score': 0.3931503891944885}]




```python
_ = list(map(lambda x: print(f"{x}: {sentiment_analyzer(x)}"), train_docs))
```

    私はこの映画をみることができて、とても嬉しい。: [{'label': 'LABEL_2', 'score': 0.3935401141643524}]
    今日の晩御飯は何だろう。: [{'label': 'LABEL_1', 'score': 0.5711930990219116}]
    猫に足を噛まれて痛い。: [{'label': 'LABEL_0', 'score': 0.5832840204238892}]


スコアは低いですが、学習に使ったデータセットに関して、正しいラベルが予測できていることを確認できました。

## まとめ

簡単な文章とラベルを用意して fine tuning する方法を記載しました。

[次の記事](https://github.com/hnishi/handson-language-models/blob/main/fine_tune_jp_bert_part02.ipynb) では、より大きなデータセットを使って、より時間のかかる学習を試してみたいと思います。
