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
        self.att_bilstm = torch.nn.LSTM(lstm_embedding_dim,hidden_size,num_layers=1,bidirectional=True,batch_first=True)
        self.raw_bilstm = torch.nn.LSTM(1024, hidden_size,num_layers=1,bidirectional=True,batch_first=True)

        #maxpool
        self.maxpool = torch.nn.MaxPool2d(2,stride=2)

        #fc
        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(20480, 2048),  # PVT输入层与第一隐层结点数设置，全连接结构
            torch.nn.Linear(12288, 2048),  # PV/no-self
            # torch.nn.Linear(16384, 2048),  # PT,no-bi
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.5),
            # torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            torch.nn.Linear(2048,4),  # 第一隐层与第二隐层结点数设置，全连接结构
            # torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
            # torch.nn.Linear(5,5),  # 第二隐层与输出层层结点数设置，全连接结构
        )

        # soft attention
        self.softmax = torch.nn.Softmax(1)
        self.dropout = torch.nn.Dropout(0.1)

        # self attention
        self.selfweight = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 1)
        )



    def forward(self, batch_inputs):
        # batch_inputs = [batch_qa0_features.cuda(), batch_qa1_features.cuda(), batch_qa2_features.cuda(), \
        # batch_qa3_features.cuda(),batch_qa4_features.cuda(),batch_text_features.cuda(),batch_visual_features.cuda()]
        def soft_att(q, k, v):
            # print(q.size(), k.size())
            attn = torch.bmm(q, k.transpose(1,2))
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            output = torch.bmm(attn, v)
            return output

        def self_att(inputs):
            # (B, L, H) -> (B , L, 1)
            energy = self.selfweight(inputs)
            weights = F.softmax(energy.squeeze(-1), dim=1)
            # (B, L, H) * (B, L, 1) -> (B, H)
            outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
            return outputs


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
            # print(c.size(0))
            if c.size(0) != 1: 
                q2c_att = q2c_att.unsqueeze(1)   #当batch=1时，这里会出问题，因为会直接去掉两层括号，而下面只加上一层，维数就不对了
            # print(q2c_att)
            q2c_att1 = q2c_att.expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att1], dim=-1)
            return x

        # batch_qa0_features = batch_inputs[0]
        # batch_qa1_features = batch_inputs[1]
        # batch_qa2_features = batch_inputs[2]
        # batch_qa3_features = batch_inputs[3]
        
        # batch_text_features = batch_inputs[5]
        # batch_visual_features = batch_inputs[6]
        # batch_keywords_features = batch_inputs[7]


        batch_outputs = []
        for bf in batch_inputs:
            batch_outputs.append(self.input_bilstm(bf)[0])
        print("BiLSTM: ")
        # print(batch_outputs[5])

        keytext_features = soft_att(batch_outputs[6], batch_outputs[4], batch_outputs[4])
        print("keytext: ")
        # print(keytext_features)
        # print(keytext_features.size())

        # get soft-att
        self_text_features = soft_att(batch_outputs[4], batch_outputs[4], batch_outputs[4])
        self_visual_features = soft_att(batch_outputs[5], batch_outputs[5], batch_outputs[5])
        self_qa_features = []
        for qa in batch_outputs[:4]:
            self_qa_features.append(soft_att(qa, qa, qa))
        print("Self-Att: ")
        # print(self_text_features.size())

        # self_text_features = self_att(batch_outputs[5])
        # self_visual_features = self_att(batch_outputs[6])
        # self_qa_features = []
        # for qa in batch_outputs[:5]:
        #     self_qa_features.append(self_att(qa))
        # print("Self-Att: ")
        # print(self_text_features.size())


        vq_features = []
        tq_features = []
        for qa in batch_outputs[:4]:
            tq_features.append(att_flow_layer(keytext_features, qa))
            vq_features.append(att_flow_layer(batch_outputs[5], qa))
            
        print("Bi-Attention: ")
        # print(vq_features[4])

        vt_features = att_flow_layer(keytext_features, batch_outputs[5])


        vq_features = [self.att_bilstm(vq)[0] for vq in vq_features]
        tq_features = [self.att_bilstm(tq)[0] for tq in tq_features]
        vt_features = self.att_bilstm(vt_features)[0]
        keytext_features = self.raw_bilstm(keytext_features)[0]
        print("BiLSTM2:")
        # print(keytext_features[:, 0, :].size())


        # vq_features = [self.maxpool(vq) for vq in vq_features]
        # tq_features = [self.maxpool(tq) for tq in tq_features]
        # vt_features = self.maxpool(vt_features)
        # keytext_features = self.maxpool(keytext_features)
        # print("Maxpool:")

        
        # a = keytext_features[:, 0, :] *  vt_features[:, 0, :]
        # print(" dot")

        # print(vq_features[0][:, 0, :].size(), tq_features[0][:, 0, :].size(), vt_features[:, 0, :].size())


        outputs = []
        for i in range(len(vq_features)):
            # P+V+T
            # outputs.append(torch.cat([self_text_features[:, 0, :], 
            #     self_visual_features[:, 0, :], self_qa_features[i][:, 0, :],
            #     vq_features[i][:, 0, :], tq_features[i][:, 0, :]], dim=1))
            # no bi-att
            # outputs.append(torch.cat([keytext_features[:, 0, :],self_text_features[:, 0, :], 
            #     self_visual_features[:, 0, :], self_qa_features[i][:, 0, :]], dim=1))
            # no self-att
            outputs.append(torch.cat([ keytext_features[:, 0, :], vq_features[i][:, 0, :], 
                tq_features[i][:, 0, :]], dim=1))
            # P + V
            # outputs.append(torch.cat([self_visual_features[:, 0, :], self_qa_features[i][:, 0, :],
            #     vq_features[i][:, 0, :]], dim=1))
            # P+T
            # outputs.append(torch.cat([keytext_features[:, 0, :], self_text_features[:, 0, :], 
            #     self_qa_features[i][:, 0, :], tq_features[i][:, 0, :]], dim=1))

        # print(len(outputs))
        # #将res里的元素拼接起来，相当于在第一维度里进行list.append
        outputs = torch.cat(outputs,1)
        # print("outputs: ", outputs)
        # #全连接层 输出维度1
        print(outputs.size())

        outputs = self.fc(outputs)
        # print("outputs0: ", outputs)
       
        return outputs


class FeatureNode():
    def __init__(self,allqa_features,text_features,visual_features,keywords_features,answer_index,question_type):
        self.allqa_features = allqa_features
        self.text_features = text_features   #kc_featrues
        self.visual_features = visual_features
        self.keywords_features = keywords_features
        self.answer_index = answer_index
        self.question_type = question_type

# #-------------------------------------------------------

def test():
    ###加载模型测试预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # lqa_dataset = get_tvqa("LifeQA/tvqa_plus_annotations/tvqa_plus_val.json",0,None)
    # targets = torch.tensor(get_targets(tvqa_dataset))
    # targets = targets.cuda()
    the_model = torch.load("BiNET-l-noself-12-0001-8-2.pkl").cuda()
    # the_model = BiNet()
    torch.no_grad()
    the_model.eval()
    allcount = 0
    truecount = 0
    question_all = [0,0,0,0,0,0,0]  #What、How、Why、Where、Who/when/which
    question_true = [0,0,0,0,0,0,0]
    others = []

    for i in range(408):

        # data_features = get_features(data)

        # load features
        # batch_visual_features = []
        # for j in range((end-start)):
        #     index = start + j
        
        path = "LifeQA/" + "test_features/" + str(i) + ".pth"
        data_features = torch.load(path)

        print("Succeed in getting features!!!")

        # allqa_features = []
        # for qaf in data_features.allqa_features:
        #     allqa_features.append(qaf)

        # visual_features = data_features.visual_features
        # text_features = data_features.text_features
        # fn = FeatureNode(allqa_features,text_features,visual_features)

        batch_inputs = [data_features.allqa_features[0].cuda(), data_features.allqa_features[1].cuda(), \
            data_features.allqa_features[2].cuda(), data_features.allqa_features[3].cuda(),  \
            data_features.text_features.cuda(), data_features.visual_features.cuda(), \
            data_features.keywords_features.cuda()]

        outputs = the_model(batch_inputs)
        print("Succeed in getting outputs")
        print(outputs)
        outputs = F.softmax(outputs,dim=1)
        print(outputs)
        b = torch.argmax(outputs,dim=1).cuda()

        target = data_features.answer_index.long().cuda()
        print(target)
        print(b)
        flag = False
        if target[0]==b[0]:
            truecount = truecount + 1
            flag = True
            print(flag)
        allcount = allcount + 1

        print(allcount, truecount, truecount/allcount)

        # get question type: What、How、Why、Where、Who/When/Which

        question_type = int(data_features.question_type.tolist()[0])


        if question_type == 0:
            question_all[0] = question_all[0]+1
            if flag == True:
                question_true[0] = question_true[0]+1
        elif question_type == 1:
            question_all[1] = question_all[1]+1
            if flag == True:
                question_true[1] = question_true[1]+1
        elif question_type == 2:
            question_all[2] = question_all[2]+1
            if flag == True:
                question_true[2] = question_true[2]+1
        elif question_type == 3:
            question_all[3] = question_all[3]+1
            if flag == True:
                question_true[3] = question_true[3]+1
        elif question_type == 4:
            question_all[4] = question_all[4]+1
            if flag == True:
                question_true[4] = question_true[4]+1
        elif question_type == 5:
            question_all[5] = question_all[5]+1
            if flag == True:
                question_true[5] = question_true[5]+1
        elif question_type == 6:
            question_all[6] = question_all[6]+1
            if flag == True:
                question_true[6] = question_true[6]+1
        else:
            print("question_type fail")
            others.append(str(question_type)+str(flag))

        print("What、How、Why、Where、Who")
        print(question_all)
        print(question_true)
        print(others)

if __name__ == '__main__':
    # train()
    test()