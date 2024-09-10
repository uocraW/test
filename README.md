# test

## 背景
NER就是抽取句子中的实体，比如人名、地名、机构名等。最简单的方式就是将NER建模成序列标注任务，对每一个词打上一个标签，然后根据标签的含义抽取实体。

```
输入:  我            爱        天               安               门               。
输出:  B-Person      O        B-Location       I-Location       I-Location      O
```
B: begin I: inside  O: other，根据这些标签的含义，就可以抽取到“天安门”这个地名实体和“我”这个人名实体了。


实现两个NER模型，一个是bert_tagger，将BERT编码出来的word表示直接过一个线性分类层。另一个是bert_lstm_crf_tagger，其实就是在上个模型基础上多一个LSTM层和CRF层。

## Step 0
目标：熟悉目录结构，基本的代码逻辑，以及debug文件的配置项(.vscode/launch.json)。
设置好python环境，直接用我的tomm环境即可，不需要自己安装。

步骤：
1. 进入main.py文件，点击右下角3.8.10 64-bit选项，然后在众多python环境中找到tomm，点击即可。此时main文件头部的所有引入都变绿证明操作正确。


## Step 1
目标：将原始数据集转化为所需要的格式。

步骤：
1. 将数据集放入./data/conll03/origin/目录下  
2. 使用debug模式中的Module(选第二个)直接运行dataprocessing/data_processing文件中的代码即可，代码和debug的配置是写好的。
3. 输入的文件在./data/conll03/目录下

具体逻辑可以看代码。

## Step 2
目标：实现bert_tagger模型，只需要补全./model/bert_tagger.py中缺失的代码片段即可。其余训练部分的代码已经写好。

步骤：
1. 在./model/bert_tagger.py中补全代码，debug能够正常通过，使用debug模式中Current File(选第一个)来运行代码即可。
2. 使用vscode的git提交代码，并push到github上。数据集文件不提交，有数据协议限制，且文件较大，没必要。

## Step 3
目标：实现bert_lstm_crf_tagger模型，只需要补全./model/bert_lstm_crf_tagger.py中缺失的代码片段即可，就是在Step 2的基础上再添加LSTM层即可，CRF代码已经写好。其余训练部分的代码已经写好。

步骤：
1. 在./model/bert_lstm_crf_tagger.py中补全代码，debug能够正常通过，使用debug模式中Current File(选第一个)来运行代码即可。
2. 使用vscode的git提交代码，并push到github上。数据集文件不提交，有数据协议限制，且文件较大，没必要。
