import numpy as np
import torch 
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
import torch.nn.functional as F

from utils.nn import Linear
import json
from nltk.tokenize import word_tokenize,sent_tokenize
import time
import os
#
class TextNet(torch.nn.Module):
    def __init__(self): #code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('local-bert-base-uncased')
        #加载bert
        self.textExtractor = BertModel.from_pretrained(
            'local-bert-base-uncased', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        #定义其他网络
        #self.fc = torch.nn.Linear(embedding_dim, code_length)
        #self.tanh = torch.nn.Tanh()
        # self.bilstm = torch.nn.LSTM(embedding_dim,512,num_layers=1,bidirectional=True,batch_first=True)

    def forward(self, tokens, segments, input_masks):
        output=self.textExtractor(tokens, token_type_ids=segments,attention_mask=input_masks)

        #取每个句子的第一个token，即cls，即可代表整个句子的语义
        #text_embeddings = output[0][:, 0, :]
        #output[0](batch size, sequence length, model hidden dimension)

        # features = self.bilstm(output[0])
        # print("features:")
        # print(features[0][0][0])
        # print("len features:")
        # print(len(features))
        # print("len features0")
        # print(len(features[0]))
        # print("len features00")
        # print(len(features[0][0]))
        # print("len features000")
        # print(len(features[0][0][0]))

        return output[0]



#——————构造模型——————
class BiNet(torch.nn.Module):
    def __init__(self,bert_hidden_size=768,embedding_dim=1024,hidden_size=512,lstm_embedding_dim=4096):
        super(BiNet, self).__init__()

        #
        self.input_bilstm = torch.nn.LSTM(bert_hidden_size, 512,num_layers=1,bidirectional=True,batch_first=True)

        # 
        self.att_weight_c = Linear(hidden_size * 2, 1)
        self.att_weight_q = Linear(hidden_size * 2, 1)
        self.att_weight_cq = Linear(hidden_size * 2, 1)

        #bilstm
        self.bilstm = torch.nn.LSTM(lstm_embedding_dim,hidden_size,num_layers=1,bidirectional=True,batch_first=True)

        #maxpool
        self.maxpool = torch.nn.MaxPool2d(2,stride=2)

        #fc
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(7680, 2048),  # 输入层与第一隐层结点数设置，全连接结构
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            # torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            torch.nn.Linear(2048,5),  # 第一隐层与第二隐层结点数设置，全连接结构
            # torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            # torch.nn.Linear(5,5),  # 第二隐层与输出层层结点数设置，全连接结构
            torch.nn.Softmax(dim=1) # 由于有两个概率输出，因此对其使用Softmax进行概率归一化
        )

        # self.fc1 = torch.nn.Linear(7680, 5)   #1024*5

        # self.s1 = torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
        # self.fc2 = torch.nn.Linear(5,5),  # 第一隐层与第二隐层结点数设置，全连接结构

    def forward(self, batch_inputs):
        # batch_inputs = [batch_qa0_features.cuda(), batch_qa1_features.cuda(), batch_qa2_features.cuda(), \
        # batch_qa3_features.cuda(),batch_qa4_features.cuda(),batch_text_features.cuda(),batch_visual_features.cuda()]
        
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            #句子长度，128
            c_len = c.size(1)
            q_len = q.size(1)
            # print(c_len,q_len) 

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)  #select取每个二维张量的第i行
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq1 = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq1

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # print(c2q_att)
            # print(c2q_att.size(2))

            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c)
            # print(q2c_att)
            if c.size(0) != 1: 
                q2c_att = q2c_att.squeeze() #当batch=1时，这里会出问题，因为会直接去掉两层括号，而下面只加上一层，维数就不对了
            # print(q2c_att)
            # (batch, c_len, hidden_size * 2) (tiled)
            print(c.size(0))
            if c.size(0) != 1: 
                q2c_att = q2c_att.unsqueeze(1)   #当batch=1时，这里会出问题，因为会直接去掉两层括号，而下面只加上一层，维数就不对了
            # print(q2c_att)
            q2c_att1 = q2c_att.expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att1], dim=-1)
            return x

        # input format!!!!
        # batch_inputs = [batch_qa0_features.cuda(), batch_qa1_features.cuda(), batch_qa2_features.cuda(), \
        # batch_qa3_features.cuda(),batch_qa4_features.cuda(),batch_text_features.cuda(),batch_visual_features.cuda()]
        print(len(batch_inputs[0]))

        # batch_qa0_features = batch_inputs[0]
        # batch_qa1_features = batch_inputs[1]
        # batch_qa2_features = batch_inputs[2]
        # batch_qa3_features = batch_inputs[3]
        # batch_qa4_features = batch_inputs[4]
        # batch_text_features = batch_inputs[5]
        # batch_visual_features = batch_inputs[6]

        # print("qa4")
        # print(batch_qa4_features)

        # print(batch_qa4_features.size(0),batch_qa4_features.size(1),batch_qa4_features.size(2))

        batch_outputs = []
        for bf in batch_inputs:
            batch_outputs.append(self.input_bilstm(bf)[0])
        print("BiLSTM: ")
        print(batch_outputs[4])

        # batch_qa0_features = self.input_bilstm(batch_qa0_features)[0]
        # batch_qa1_features = self.input_bilstm(batch_qa1_features)[0]
        # batch_qa2_features = self.input_bilstm(batch_qa2_features)[0]
        # batch_qa3_features = self.input_bilstm(batch_qa3_features)[0]
        # batch_qa4_features = self.input_bilstm(batch_qa4_features)[0]
        # batch_text_features = self.input_bilstm(batch_text_features)[0]
        # batch_visual_features = self.input_bilstm(batch_visual_features)[0]

        # print(batch_qa4_features)
        # print(batch_qa4_features.size(0),batch_qa4_features.size(1),batch_qa4_features.size(2))

        vq_bi_attention = []
        tq_bi_attention = []
        for qa in batch_outputs[:5]:
            vq_bi_attention.append(att_flow_layer(batch_outputs[6], qa))
            tq_bi_attention.append(att_flow_layer(batch_outputs[5], qa))
        print("Attention: ")
        print(vq_bi_attention[4])

        vt_bi_attention = att_flow_layer(batch_outputs[5], batch_outputs[6])


        vq_features = [self.bilstm(vq)[0] for vq in vq_bi_attention]
        tq_features = [self.bilstm(tq)[0] for tq in tq_bi_attention]
        vt_features = self.bilstm(vt_bi_attention)[0]
        print("BiLSTM2:")
        print(vq_features)


        vq_features = [self.maxpool(vq) for vq in vq_features]
        tq_features = [self.maxpool(tq) for tq in tq_features]
        vt_features = self.maxpool(vt_features)
        print("Maxpool:")


        outputs = []
        for i in range(len(vq_features)):
            outputs.append(torch.cat([vq_features[i][:, 0, :], tq_features[i][:, 0, :], vt_features[:, 0, :]], dim=1))

        print(len(outputs))
        print(outputs)
        # #将res里的元素拼接起来，相当于在第一维度里进行list.append
        outputs = torch.cat(outputs,1)
        print("outputs: ", outputs)
        
        # #全连接层 输出维度1
        outputs = self.fc(outputs)
        print("outputs0: ", outputs)
       
        return outputs

'''
        outputs = []
        for data_features in batch:

            allqa_features = []
            for f in data_features.allqa_features:
                qaf = self.input_bilstm(f)[0]
                allqa_features.append(qaf)

            visual_features = self.input_bilstm(data_features.visual_features)[0]
            text_features = self.input_bilstm(data_features.text_features)[0]


            # 输入参数为data_features: allqa_features,text_features,visual_fetures
            allqa_bi_features = []
            for _qaf in allqa_features:
                vq_bi_attention = att_flow_layer(visual_features, _qaf)
                tq_bi_attention = att_flow_layer(text_features, _qaf)
                vt_bi_attention = att_flow_layer(visual_features, text_features)

                vq_lstm_features = self.bilstm(vq_bi_attention)
                tq_lstm_features = self.bilstm(tq_bi_attention)
                vt_lstm_features = self.bilstm(vt_bi_attention)

                vq_maxpool_features = self.maxpool(vq_lstm_features[0])
                tq_maxpool_features = self.maxpool(tq_lstm_features[0])
                vt_maxpool_features = self.maxpool(vt_lstm_features[0])


                a_features = torch.cat([vq_maxpool_features[:, 0, :], tq_maxpool_features[:, 0, :], vt_maxpool_features[:, 0, :]], dim=1)
                # a_features = torch.add(vq_lstm_features[0][:, 0, :],tq_lstm_features[0][:, 0, :])
                # b_features = torch.add(a_features,vt_lstm_features[0][:, 0, :])

                allqa_bi_features.append(a_features)

            res = tuple(allqa_bi_features)
            #将res里的元素拼接起来，相当于在第一维度里进行list.append
            last = torch.cat(res,1)
            print("last: ", last)
            #全连接层 输出维度1
            # fc1 = torch.nn.Linear(7680, 5)   #1024*5
            # s1 = torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            # fc2 = torch.nn.Linear(5,5),  # 第一隐层与第二隐层结点数设置，全连接结构
            # llast = self.fc1(last)
            # print("llast: ", llast)


            # llast0 = llast.squeeze(1)
            # print(llast0)
            # llast1 = llast0.unsqueeze(0)
            # print(llast1)

            # lllast = F.softmax(llast,dim=1)

            last = self.fc(last)
            print("lllast: ", last)


            #maxpool_features = self.maxpool(lstm_features[0])

            #features = self.fc(text_embeddings)
            #features=self.tanh(features)
            print("hahahhahahah")
            outputs.append(last)
        print(outputs)
        print("hihihihihi")

        return outputs
'''
        
#——————输入处理——————
def text_to_tensor(texts):
    tokenizer = BertTokenizer.from_pretrained('local-bert-base-uncased')

    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text) #用tokenizer对句子分词
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)#索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    # max_len = max([len(single) for single in tokens]) #最大的句子长度
    # max_len = 0

    # for j in range(len(tokens)):
    #     padding = [0] * (max_len - len(tokens[j]))
    #     tokens[j] = tokens[j] + padding
    #     segments[j] = segments[j] + padding
    #     input_masks[j] = input_masks[j] + padding


    #segments列表全0，因为只有一个句子1，没有句子2
    #input_masks列表1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
    #相当于告诉BertModel不要利用后面0的部分

    if len(tokens[0])>512:
        for i in range(len(tokens)):
            tokens[i] = tokens[i][:128]+tokens[i][-382:]
            segments[i] = segments[i][:128]+segments[i][-382:]
            input_masks[i] = input_masks[i][:128]+input_masks[i][-382:]

    # max_len = max([len(single) for single in tokens]) #最大的句子长度
    # max_len = 0

    # for j in range(len(tokens)):
    #     padding = [0] * (max_len - len(tokens[j]))
    #     tokens[j] = tokens[j] + padding
    #     segments[j] = segments[j] + padding
    #     input_masks[j] = input_masks[j] + padding

        
    #转换成PyTorch tensors
    tokens_tensor = torch.tensor(tokens)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(input_masks)
    # print("tokens: ")
    #print(tokens_tensor)

    # print("segments:")
    #print(segments_tensors)

    # print("input_masks_tensors: ")
    #print(input_masks_tensors)
    print("Succeed in getting tensor!")
    # print(tokens_tensor, segments_tensors, input_masks_tensors)
    return tokens_tensor, segments_tensors, input_masks_tensors

#
def read_jsonfile(json_path):
    with open(json_path,'r', encoding='utf-8') as json_file:
        content = json.load(json_file)
        return content

def get_tvqa_plus_subtitle(json_path,vid_name):
    subtitle_file = read_jsonfile(json_path)
    if vid_name in subtitle_file:
        sub_text = subtitle_file[vid_name]["sub_text"]
        return sub_text
    else:
        print("Vid_name not found!")

def clean_subtitle(sub_text):
    subtitle = sub_text.replace("<eos>", " ")
    return subtitle

#处理text，将text前后分别加上CLS和SEP，以输入BERT
def add_cls_and_sep(text):
    sen = "[CLS] " + text + " [SEP]"
    return sen

class DataNode():
    def __init__(self,video="",question="",candidate_answers=[],answer_index=0,subtitle="",visual_text="",keywords="",captions=""):
        self.video = video
        self.question = question
        self.candidate_answers = candidate_answers
        self.answer_index = answer_index
        self.subtitle = subtitle
        self.visual_text = visual_text
        self.keywords = keywords
        self.captions = captions  #仅lifeQA数据集

class FeatureNode():
    def __init__(self,allqa_features,ks_features,visual_features):
        self.allqa_features = allqa_features
        self.text_features = ks_features   #kc_featrues
        self.visual_features = visual_features

def get_features(data):
    textNet = TextNet()

    qas = [[add_cls_and_sep(data.question+ai)] for ai in data.candidate_answers]
    print(qas)
    allqa_features = []
    for _qa in qas:
        qa_tokens_tensor, qa_segments_tensors, qa_input_masks_tensors = text_to_tensor(_qa)
        qa_features = textNet(qa_tokens_tensor, qa_segments_tensors, qa_input_masks_tensors)
        # print(qa_features)
        # print(len(qa_features[0]),len(qa_features[0][0]),len(qa_features[0][0][0]))  #1,128,1024
        allqa_features.append(qa_features)
    print("Succeed in getting QA features!!")

    # keywords+subtitle   
    text = data.subtitle+" "+data.keywords
    # if get_length(text)>510:
    #     words = word_tokenize(text)
    #     text = " ".join(words[:510])
    ks_tokens_tensor, ks_segments_tensors, ks_input_masks_tensors = text_to_tensor([add_cls_and_sep(text)])
    ks_features = textNet(ks_tokens_tensor, ks_segments_tensors, ks_input_masks_tensors)
    print("Succeed in getting ks features!!")

    #visual text
    # visual_text = data.visual_text
    # if get_length(visual_text)>254:
    #     visual_words = word_tokenize(visual_text)
    #     visual_text = " ".join(visual_words[:254])
    v_tokens_tensor, v_segments_tensors, v_input_masks_tensors = text_to_tensor([add_cls_and_sep(data.visual_text)])
    visual_features = textNet(v_tokens_tensor, v_segments_tensors, v_input_masks_tensors)
    print("Succeed in getting visual features!!")

    features = FeatureNode(allqa_features,ks_features,visual_features)
    return features

def get_targets(dataset):
    targets = []
    for data in dataset:
        targets.append([int(data.answer_index)])
    return targets

#计算单词个数
def get_length(text):
    length = len(word_tokenize(text))
    return length


#########获取数据集
def get_tvqa(dataset_path,start,end):
    tvqa_dataset = []
    tvqa_plus_annotations = read_jsonfile(dataset_path)
    # tvqa_plus_subtitles = read_jsonfile("TVQA/tvqa_plus_subtitles/tvqa_plus_subtitles.json")
    for i, data in enumerate(tvqa_plus_annotations[start:end]):
        video = data["vid_name"]
        question = data["q"]
        a0 = data["a0"]
        a1 = data["a1"]
        a2 = data["a2"]
        a3 = data["a3"]
        a4 = data["a4"]
        candidate_answers = [a0,a1,a2,a3,a4]
        answer_index = data["answer_idx"]
        subtitle = clean_subtitle(get_tvqa_plus_subtitle("TVQA/tvqa_plus_subtitles/tvqa_plus_subtitles.json",video))
        # if get_length(subtitle)>510:
        #     words = word_tokenize(subtitle)
        #     subtitle = " ".join(words[:510])
        
        # subtitle = clean_subtitle(tvqa_plus_subtitles[video]["sub_text"])
        # subtitle = " Rose"
        # print("subtitle:  "+subtitle)
        visual_text_list =[]
        bbox = data["bbox"]
        for key in bbox:
            for box in bbox[key]:
                if box["label"] not in visual_text_list:
                    visual_text_list.append(box["label"])
        visual_text = ", ".join(visual_text_list)
        # visual_text = "Rose"
        # print("visual:   "+visual_text)
        keywords = " "
        datanode = DataNode(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
        # print(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
        tvqa_dataset.append(datanode)

    return tvqa_dataset

def divide_dataset(dataset, batch_size):
    res_dataset = []
    start = 0
    end = start + batch_size
    dataset_length = len(dataset)
    while end<=dataset_length:
        batch = dataset[start:end]
        res_dataset.append(batch)
        start = end
        end = start + batch_size
    
    if end>dataset_length:
        end=dataset_length
        if start!=end:
            batch = dataset[start:end]
            res_dataset.append(batch)
    return res_dataset

# def divide_targets(targets, batch_size):
#     res_targets = []
#     start = 0
#     end = start + batch_size
#     targets_length = len(targets)
#     while end<=dataset_length:
#         batch = dataset[start:end]
#         res_dataset.append(batch)
#         start = end
#         end = start + batch_size
    
#     if end>dataset_length:
#         end=dataset_length
#         batch = dataset[start:end]
#         res_dataset.append(batch)
#     return res_dataset


def get_lifeqa(json_path):
    # lifeqa = Dataset()
    data = read_jsonfile(json_path)
    for key in data:
        video = key
        for q in data[key]["questions"]:
            question = q["question"]
            answer_index = q["correct_index"]
            candidate_answers = q["answers"]

def feature_zero_padding(feature, maxlen):
    # [1,x,768] -> [1,maxlen,768]
    print(feature)
    print(feature.size(0),feature.size(1), feature.size(2))
    zero = torch.zeros((feature.size(0),maxlen-feature.size(1), feature.size(2)))
    new_f = torch.cat([feature,zero], dim=1)
    print(new_f)
    return new_f

def batch_zero_padding(batch_features):
    # if type == "text":
    maxlen = max([v.size(1) for v in batch_features])
    print("max: ",maxlen)
    new_batch = []
    for f in batch_features:
        newf = feature_zero_padding(f, maxlen)
        new_batch.append(newf)
    return new_batch
    # else:
    #     maxlen = max([qa.size(1) for qas in batch_features for qa in qas])
    #     print("max: ",maxlen)
    #     new_batch = []

    #     for qas in batch_features:
    #         new_qas = []
    #         for f in qas:
    #             newf = feature_zero_padding(f, maxlen)
    #             new_qas.append(newf)
    #         new_batch.append(new_qas)
    #     return new_batch

def get_tvqa_batch_features(start=0, end=None):
    directory = "TVQA/train_features_filter/"
    batch_visual_features = []
    batch_text_features = []
    batch_qa0_features = []
    batch_qa1_features = []
    batch_qa2_features = []
    batch_qa3_features = []
    batch_qa4_features = []
    for j in range(end-start):
        index = start + j
        fname = str(index) + ".pth"
        path = directory + fname
        df = torch.load(path)
        batch_visual_features.append(df.visual_features)
        batch_text_features.append(df.text_features)
        batch_qa0_features.append(df.allqa_features[0])
        batch_qa1_features.append(df.allqa_features[1])
        batch_qa2_features.append(df.allqa_features[2])
        batch_qa3_features.append(df.allqa_features[3])
        batch_qa4_features.append(df.allqa_features[4])

    return batch_qa0_features, batch_qa1_features, batch_qa2_features,\
            batch_qa3_features, batch_qa4_features, batch_text_features, batch_visual_features

def process_tvqa_batch_features(batch_qa0_features, batch_qa1_features, batch_qa2_features,
            batch_qa3_features, batch_qa4_features, batch_text_features, batch_visual_features):

    batch_visual_features = batch_zero_padding(batch_visual_features)
    print("padding_visual_features: ")
    print(batch_visual_features)
    batch_visual_features = torch.cat(batch_visual_features, dim=0)
    print("cat_visual_features: ")
    print(batch_visual_features)

    batch_text_features = batch_zero_padding(batch_text_features)
    print("padding_text_features: ")
    print(batch_text_features)
    batch_text_features = torch.cat(batch_text_features, dim=0)
    print("cat_text_features: ")
    print(batch_text_features)

    batch_qa0_features = batch_zero_padding(batch_qa0_features)
    print("padding_qa0_features: ")
    print(batch_qa0_features)
    batch_qa0_features = torch.cat(batch_qa0_features, dim=0)
    print("cat_qa0_features: ")
    print(batch_qa0_features)

    batch_qa1_features = batch_zero_padding(batch_qa1_features)
    print("padding_qa1_features: ")
    print(batch_qa1_features)
    batch_qa1_features = torch.cat(batch_qa1_features, dim=0)
    print("cat_qa1_features: ")
    print(batch_qa1_features)

    batch_qa2_features = batch_zero_padding(batch_qa2_features)
    print("padding_qa2_features: ")
    print(batch_qa2_features)
    batch_qa2_features = torch.cat(batch_qa2_features, dim=0)
    print("cat_qa2_features: ")
    print(batch_qa2_features)

    batch_qa3_features = batch_zero_padding(batch_qa3_features)
    print("padding_qa3_features: ")
    print(batch_qa3_features)
    batch_qa3_features = torch.cat(batch_qa3_features, dim=0)
    print("cat_qa3_features: ")
    print(batch_qa3_features)

    batch_qa4_features = batch_zero_padding(batch_qa4_features)
    print("padding_qa4_features: ")
    print(batch_qa4_features)
    batch_qa4_features = torch.cat(batch_qa4_features, dim=0)
    print("cat_qa4_features: ")
    print(batch_qa4_features)

    # batch_allqa_features = batch_zero_padding(batch_allqa_features, "qa")
    # print("padding_allqa_features: ")
    # print(batch_allqa_features)
    # batch_allqa_features = torch.cat(batch_allqa_features, dim=0)
    # print("cat_allqa_features: ")
    # print(batch_allqa_features)
    return batch_qa0_features, batch_qa1_features, batch_qa2_features,\
            batch_qa3_features, batch_qa4_features, batch_text_features, batch_visual_features


# #-------------------------------------------------------

def train():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BiNet(bert_hidden_size=768,embedding_dim=1024,hidden_size=512,lstm_embedding_dim=4096)
    # binet = torch.load("BiNET12-0001.pkl")
    model0 = model.cuda()

    binet = torch.nn.DataParallel(model0, device_ids=[0,1])

    #train
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(binet.module.parameters(), lr=0.0001)

    # tvqa_dataset = get_tvqa("TVQA/tvqa_plus_annotations/tvqa_plus_train.json")
    print("Succeed in loading dataset!!!!!")


    ##target要改
    # targets = torch.tensor(get_targets(tvqa_dataset))

    # batch_size = 8
    # train_dataset = divide_dataset(tvqa_dataset,batch_size)
    # train_targets = divide_dataset(targets,batch_size)

    for epoch in range(3):  # loop over the dataset multiple times
        start = 0
        step = 12   #batch_size
        end = start+step
        num = 23544

        # running_loss = 0.0
        # for i, data in enumerate(train_dataset):
        while start < num:
            if end <= num:
                data_set =  get_tvqa("TVQA/tvqa_plus_annotations/tvqa_plus_train.json", start, end)
            else:
                end = num
                data_set =  get_tvqa("TVQA/tvqa_plus_annotations/tvqa_plus_train.json", start, end)

            train_targets = torch.tensor(get_targets(data_set))
            train_targets = train_targets.cuda()
            print(train_targets)


            print("start: ", start, "end: ", end)

            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data

            binet.train()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = net(inputs)

            # data_features1 = []
            # batch = []
            # for d in data:

            # extract features
            # for d in data_set:
                # data_features1.append(get_features(d))

            # load features
            # batch_visual_features = []
            # for j in range((end-start)):
            #     index = start + j
            #     path = "TVQA/" + "train_features_filter/" + str(index) + ".pth"
            #     df = torch.load(path)
            #     print("df.visual_features:")
            #     print(df.visual_features)
            #     batch_visual_features.append(df.visual_features)

            batch_qa0_features, batch_qa1_features, batch_qa2_features,batch_qa3_features,\
            batch_qa4_features, batch_text_features, batch_visual_features = get_tvqa_batch_features(start, end)
            
            batch_qa0_features, batch_qa1_features, batch_qa2_features,batch_qa3_features,batch_qa4_features, \
            batch_text_features, batch_visual_features = \
            process_tvqa_batch_features(batch_qa0_features, batch_qa1_features, batch_qa2_features,\
            batch_qa3_features, batch_qa4_features, batch_text_features, batch_visual_features)

            batch_inputs = [batch_qa0_features.cuda(), batch_qa1_features.cuda(), batch_qa2_features.cuda(), \
            batch_qa3_features.cuda(), batch_qa4_features.cuda(), \
            batch_text_features.cuda(), batch_visual_features.cuda()]

                # allqa_features = []
                # for qaf in df.allqa_features:
                #     allqa_features.append(qaf.cuda())

                # visual_features = df.visual_features.cuda()
                # text_features = df.text_features.cuda()

                # fn = FeatureNode(allqa_features,text_features,visual_features)
                # batch.append(fn)


            print("Succeed in getting features!!!")

            # outputs = []
            # for _df in batch:
            #     outputs.append(binet(_df))
            outputs = binet(batch_inputs)

            print("Succeed in getting outputs!!!!")
            print(outputs)


            # print(targets)
            # print(train_targets)
            # print(train_targets[i])
            # target = train_targets[i].squeeze(1)
            target = train_targets.squeeze(1)
            print(target)

            loss = criterion(outputs, target)
            #loss.requires_grad = True
            print(loss)
            print("Succeed in getting loss")
            print("data start: ",start,"  times: ",epoch)
            print("time:  ",time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            #loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()

            start = end
            end = start + step

            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

        torch.save(binet, "BiNET.pkl")
    print('Finished Training')


def test():
    ###加载模型测试预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tvqa_dataset = get_tvqa("TVQA/tvqa_plus_annotations/tvqa_plus_val.json",0,None)
    targets = torch.tensor(get_targets(tvqa_dataset))
    targets = targets.cuda()
    the_model = torch.load("BiNET-soft-tvqa-0001-4-00001-2.pkl").cuda()
    # the_model = BiNet()
    torch.no_grad()
    the_model.eval()
    allcount = 0
    truecount = 0

    for i, data in enumerate(tvqa_dataset):

        # data_features = get_features(data)

        # load features
        # batch_visual_features = []
        # for j in range((end-start)):
        #     index = start + j
        
        path = "TVQA/" + "val_features_filter/" + str(i) + ".pth"
        data_features = torch.load(path)

        print("Succeed in getting features!!!")

        # allqa_features = []
        # for qaf in data_features.allqa_features:
        #     allqa_features.append(qaf)

        # visual_features = data_features.visual_features
        # text_features = data_features.text_features
        # fn = FeatureNode(allqa_features,text_features,visual_features)

        batch_inputs = [data_features.allqa_features[0].cuda(), data_features.allqa_features[1].cuda(), \
            data_features.allqa_features[2].cuda(), \
            data_features.allqa_features[3].cuda(), data_features.allqa_features[4].cuda(), \
            data_features.text_features.cuda(), data_features.visual_features.cuda()]

        outputs = the_model(batch_inputs)
        print("Succeed in getting outputs")
        print(outputs)
        outputs = F.softmax(outputs,dim=1)
        print(outputs)
        b = torch.argmax(outputs,dim=1).cuda()
        print(targets[i])
        print(b)
        if targets[i][0]==b[0]:
            truecount = truecount + 1
            print(True)
        allcount = allcount + 1

        print(allcount, truecount, truecount/allcount)

if __name__ == '__main__':
    # test hard final
    # train()
    test()