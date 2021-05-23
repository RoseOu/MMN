import json
import jsonlines

import numpy as np
import torch 
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
import torch.nn.functional as F

from utils.nn import Linear
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet
import time
# from pywsd.lesk import simple_lesk
import networkx as nx



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
        # print(len(tokens[i]))
        
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

def get_old_tvqa_subtitle(dir_path,vid_name):
    file_path = dir_path + vid_name + ".srt"
    with open(file_path,'r', encoding='utf-8') as file:
        content = file.readlines()

    bat = []
    subtitle = ""
    for item in content:
        if item == "\n":
            subtitle = subtitle + "".join(bat[2:])
            bat = []
        else:
            bat.append(item.strip())

    return subtitle

def get_old_tvqa_gt_subtitle(dir_path,vid_name,ts):
    # change 00:00:31,845 to ..
    def time_format(time):
        hour = time.split(":")[0]
        minute = time.split(":")[1]
        sec = time.split(":")[2]
        sec = sec.replace(",",".")
        res = int(hour) * 60 * 60 + int(minute) * 60 + float(sec)
        return res


    file_path = dir_path + vid_name + ".srt"
    with open(file_path,'r', encoding='utf-8') as file:
        content = file.readlines()

    bat = []
    subtitle = ""

    for item in content:
        if item == "\n":
            start = time_format(bat[1].split("-->")[0].strip())
            end = time_format(bat[1].split("-->")[1].strip())
            ts_start = float(ts.split("-")[0])
            ts_end = float(ts.split("-")[1])
            sentence = "".join(bat[2:])
            if start<ts_end and end>ts_start:
                subtitle = subtitle + sentence
            bat = []
        else:
            bat.append(item.strip())

    return subtitle

def clean_subtitle(sub_text):
    subtitle = sub_text.replace("<eos>", " ")
    return subtitle

#处理text，将text前后分别加上CLS和SEP，以输入BERT
def add_cls_and_sep(text):
    sen = "[CLS] " + text + " [SEP]"
    return sen

class DataNode():
    def __init__(self,video="",question="",candidate_answers=[],answer_index=0,subtitle="",visual_text="",keywords=""):
        self.video = video
        self.question = question
        self.candidate_answers = candidate_answers
        self.answer_index = answer_index
        self.subtitle = subtitle
        self.visual_text = visual_text
        self.keywords = keywords


class FeatureNode():
    def __init__(self,allqa_features,text_features,visual_features,keywords_features,answer_index):
        self.allqa_features = allqa_features
        self.text_features = text_features   #kc_featrues
        self.visual_features = visual_features
        self.keywords_features = keywords_features
        self.answer_index = answer_index



def get_features(data):
    textNet = TextNet()

    qas = [[add_cls_and_sep(data.question+ " [SEP] " +ai)] for ai in data.candidate_answers]
    # qas = [['[CLS] Who are you? Jack. [SEP]'], 
    # ['[CLS] Who are you? Mark . [SEP]'], 
    # ['[CLS] Who are you? Mike . [SEP]'], 
    # ['[CLS] Who are you? Rose . [SEP]'], 
    # ['[CLS] Who are you? Peter . [SEP]']]
    print(qas)
    allqa_features = []
    for _qa in qas:
        qa_tokens_tensor, qa_segments_tensors, qa_input_masks_tensors = text_to_tensor(_qa)
        qa_features = textNet(qa_tokens_tensor, qa_segments_tensors, qa_input_masks_tensors)
        # print(qa_features)
        # print(len(qa_features[0]),len(qa_features[0][0]),len(qa_features[0][0][0]))  #1,128,1024
        allqa_features.append(qa_features)
    print("Succeed in getting QA features!!")

    # #keyword
    print(data.keywords)
    k_tokens_tensor, k_segments_tensors, k_input_masks_tensors = text_to_tensor([add_cls_and_sep(data.keywords)])
    keywords_features = textNet(k_tokens_tensor, k_segments_tensors, k_input_masks_tensors)
    print("Succeed in getting keywords features!!")

    # subtitle text
    text_tokens_tensor, text_segments_tensors, text_input_masks_tensors = text_to_tensor([add_cls_and_sep(data.subtitle)])
    text_features = textNet(text_tokens_tensor, text_segments_tensors, text_input_masks_tensors)
    print("Succeed in getting text features!!")

    #visual text
    v_tokens_tensor, v_segments_tensors, v_input_masks_tensors = text_to_tensor([add_cls_and_sep(data.visual_text)])
    visual_features = textNet(v_tokens_tensor, v_segments_tensors, v_input_masks_tensors)
    print("Succeed in getting visual features!!")

    # answer_index
    answer_index = torch.Tensor([data.answer_index])


    features = FeatureNode(allqa_features,text_features,visual_features,keywords_features, answer_index)
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
        
        visual_text_list =[]
        bbox = data["bbox"]
        for key in bbox:
            for box in bbox[key]:
                if box["label"] not in visual_text_list:
                    visual_text_list.append(box["label"])
        visual_text = ". ".join(visual_text_list)
        
        
        stopwords_path = "keyword_extraction/english.txt"
        stopwords = stopword_list(stopwords_path)
        char_path = "keyword_extraction/remove.txt"
        charwords = stopword_list(char_path)


        sentences = sent_tokenize(subtitle)
        subtitle_wordslist = [word_tokenize(sen) for sen in sentences]
        filter_subtitle_words = [remove_stopwords(word_tokenize(sen), charwords) for sen in sentences]

        # print(filter_subtitle_words)

        filter_question_words = remove_stopwords(word_tokenize(question),charwords)
        print(filter_question_words)

        keywords = []
        keywords = keywords + filter_question_words

        #-----------------问题相关性计算---------------------------------------
        # #计算问题相关性
        # #将词语列表转成语义列表
        # senseslist = words_to_senses(subtitle_wordslist,sentences)

        # #将问题词语列表转成语义列表
        # question_senses = title_words_to_senses(filter_question_words, data.question)

        # #计算问题相关性得分
        # question_map = title_sim(subtitle_wordslist, senseslist, question_senses)
        # print(question_map)
        # print("标题相关性计算完毕")

        #-----------------视觉相关性计算---------------------------------------
        # 计算视觉相关性
        visual_words = visual_text.split(". ")
        visual_map = visual_sim(filter_subtitle_words, visual_words)
        print("视觉相关性计算完毕")
        print(visual_words)
        print(visual_map)

        #-----------------语义重要性计算---------------------------------------
        #计算语义重要性得分
        G = get_graph(filter_subtitle_words)
        semantic_map = semantic_score(G, stopwords)
        print("语义重要性计算完毕")
        print(semantic_map)

        #-----------------TF-IDF值计算---------------------------------------
        # tf_map = get_tf(subtitle_wordslist,txt_path)
        # corpus_txt_path = 'newstxt'
        # idf_map = get_idf(wordslist,corpus_txt_path)
        # print("TF-IDF值计算完毕")

        #-----------------取词取句连成篇---------------------------------------
        
        total_map = compute({},visual_map,semantic_map)
        total_list = sorted(total_map.items(), key=lambda item:item[1], reverse=True)
        print(total_map)
        print(total_list)

        all_count = count_words(filter_subtitle_words)
        now_count = len(keywords)
 
        print(all_count)
        print(now_count)
        k = 0
        while now_count < all_count * 0.2 and k < len(total_list):
            word = total_list[k][0]
            print(word)
            if word not in keywords:
                keywords.append(word)
            now_count = len(keywords)
            print(now_count)
            k = k + 1

        keywords = ". ".join(keywords)
        print(keywords)

        datanode = DataNode(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
        print(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
        tvqa_dataset.append(datanode)

    return tvqa_dataset


def get_lqa(dataset_path):
    lqa_dataset = []
    lqa_json = read_jsonfile(dataset_path)
    for key in lqa_json:
        video = key
        print(video)
        data = lqa_json[key]
        subtitle = ""
        if "manual_captions" in data: 
            for c in data["manual_captions"]:
                subtitle = subtitle + " " + c["transcript"]
        else:
            subtitle = " "
        print(subtitle)

        lqa_objects = read_jsonfile("LifeQA/lqa_objects/"+key+".json")
        visual_text_list =[]
        for lo in lqa_objects:
            olist = lo[1]
            for obj in olist:
                if obj not in visual_text_list:
                    visual_text_list.append(obj)
        visual_text = ". ".join(visual_text_list)
        print(visual_text)

        stopwords_path = "keyword_extraction/english.txt"
        stopwords = stopword_list(stopwords_path)
        char_path = "keyword_extraction/remove.txt"
        charwords = stopword_list(char_path)


        sentences = sent_tokenize(subtitle)
        subtitle_wordslist = [word_tokenize(sen) for sen in sentences]
        filter_subtitle_words = [remove_stopwords(word_tokenize(sen), charwords) for sen in sentences]

        # print(filter_subtitle_words)

        for q in data["questions"]:
            question = q["question"]
            candidate_answers = q["answers"]
            answer_index = q["correct_index"]

            filter_question_words = remove_stopwords(word_tokenize(question),charwords)
            print(filter_question_words)

            keywords = []
            keywords = keywords + filter_question_words

            #-----------------问题相关性计算---------------------------------------
            # #计算问题相关性
            # #将词语列表转成语义列表
            # senseslist = words_to_senses(subtitle_wordslist,sentences)

            # #将问题词语列表转成语义列表
            # question_senses = title_words_to_senses(filter_question_words, data.question)

            # #计算问题相关性得分
            # question_map = title_sim(subtitle_wordslist, senseslist, question_senses)
            # print(question_map)
            # print("标题相关性计算完毕")

            #-----------------视觉相关性计算---------------------------------------
            # 计算视觉相关性
            visual_words = visual_text.split(". ")
            visual_map = visual_sim(filter_subtitle_words, visual_words)
            print("视觉相关性计算完毕")
            print(visual_words)
            print(visual_map)

            #-----------------语义重要性计算---------------------------------------
            #计算语义重要性得分
            G = get_graph(filter_subtitle_words)
            semantic_map = semantic_score(G, stopwords)
            print("语义重要性计算完毕")
            print(semantic_map)

            #-----------------TF-IDF值计算---------------------------------------
            # tf_map = get_tf(subtitle_wordslist,txt_path)
            # corpus_txt_path = 'newstxt'
            # idf_map = get_idf(wordslist,corpus_txt_path)
            # print("TF-IDF值计算完毕")

            #-----------------取词取句连成篇---------------------------------------
            
            total_map = compute({},visual_map,semantic_map)
            total_list = sorted(total_map.items(), key=lambda item:item[1], reverse=True)
            print(total_map)
            print(total_list)

            all_count = count_words(filter_subtitle_words)
            now_count = len(keywords)
     
            print(all_count)
            print(now_count)
            k = 0
            while now_count < all_count * 0.2 and k < len(total_list):
                word = total_list[k][0]
                print(word)
                if word not in keywords:
                    keywords.append(word)
                now_count = len(keywords)
                print(now_count)
                k = k + 1

            keywords = ". ".join(keywords)
            print(keywords)
                

            datanode = DataNode(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
            # print(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
            lqa_dataset.append(datanode)
            print("________________")

    return lqa_dataset


def get_old_tvqa(dataset_path,start,end):
    tvqa_dataset = []
    with open(dataset_path, "r", encoding="utf-8") as tvqa_qa_file:
        i=0
        for data in jsonlines.Reader(tvqa_qa_file):
            if i>=start and i<end:
                video = data["vid_name"]
                question = data["q"]
                a0 = data["a0"]
                a1 = data["a1"]
                a2 = data["a2"]
                a3 = data["a3"]
                a4 = data["a4"]
                candidate_answers = [a0,a1,a2,a3,a4]
                answer_index = data["answer_idx"]
                ts = data["ts"]
                # subtitle = get_old_tvqa_subtitle("TVQA-/tvqa_subtitles/", video)
                #gt_subtitle
                subtitle = get_old_tvqa_gt_subtitle("TVQA-/tvqa_subtitles/", video, ts)

                # print(question)
                # print(candidate_answers)
                # print(answer_index)
                # print(subtitle)
                # print("---")
                # print(gt_subtitle)
                # print("\n")
                
                # visual_text_list =[]
                # bbox = data["bbox"]
                # for key in bbox:
                #     for box in bbox[key]:
                #         if box["label"] not in visual_text_list:
                #             visual_text_list.append(box["label"])
                # visual_text = ". ".join(visual_text_list)
                visual_text = ""

                
                stopwords_path = "keyword_extraction/english.txt"
                stopwords = stopword_list(stopwords_path)
                char_path = "keyword_extraction/remove.txt"
                charwords = stopword_list(char_path)

                
                sentences = sent_tokenize(subtitle)
                subtitle_wordslist = [word_tokenize(sen) for sen in sentences]
                filter_subtitle_words = [remove_stopwords(word_tokenize(sen), charwords) for sen in sentences]

                # print(filter_subtitle_words)

                filter_question_words = remove_stopwords(word_tokenize(question),charwords)
                print(filter_question_words)

                keywords = []
                keywords = keywords + filter_question_words

                #-----------------问题相关性计算---------------------------------------
                # #计算问题相关性
                # #将词语列表转成语义列表
                # senseslist = words_to_senses(subtitle_wordslist,sentences)

                # #将问题词语列表转成语义列表
                # question_senses = title_words_to_senses(filter_question_words, data.question)

                # #计算问题相关性得分
                # question_map = title_sim(subtitle_wordslist, senseslist, question_senses)
                # print(question_map)
                # print("标题相关性计算完毕")

                #-----------------视觉相关性计算---------------------------------------
                # 计算视觉相关性
                visual_words = visual_text.split(". ")
                visual_map = visual_sim(filter_subtitle_words, visual_words)
                print("视觉相关性计算完毕")
                print(visual_words)
                print(visual_map)

                #-----------------语义重要性计算---------------------------------------
                #计算语义重要性得分
                G = get_graph(filter_subtitle_words)
                semantic_map = semantic_score(G, stopwords)
                print("语义重要性计算完毕")
                print(semantic_map)

                #-----------------TF-IDF值计算---------------------------------------
                # tf_map = get_tf(subtitle_wordslist,txt_path)
                # corpus_txt_path = 'newstxt'
                # idf_map = get_idf(wordslist,corpus_txt_path)
                # print("TF-IDF值计算完毕")

                #-----------------取词取句连成篇---------------------------------------
                
                total_map = compute({},visual_map,semantic_map)
                total_list = sorted(total_map.items(), key=lambda item:item[1], reverse=True)
                print(total_map)
                print(total_list)

                all_count = count_words(filter_subtitle_words)
                now_count = len(keywords)
         
                print(all_count)
                print(now_count)
                k = 0
                while now_count < all_count * 0.2 and k < len(total_list):
                    word = total_list[k][0]
                    print(word)
                    if word not in keywords:
                        keywords.append(word)
                    now_count = len(keywords)
                    print(now_count)
                    k = k + 1

                keywords = ". ".join(keywords)
                print(keywords)

                datanode = DataNode(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
                print(video,question,candidate_answers,answer_index,subtitle,visual_text,keywords)
                tvqa_dataset.append(datanode)

            i=i+1

    return tvqa_dataset


def stopword_list(stopwords_path):
    with open (stopwords_path,'r',encoding='utf-8') as file:
        lines = file.readlines()
        stopwords_list = [line.strip() for line in lines]
    return stopwords_list

#格式为[a,b,c]
def remove_stopwords(wordlist,stopwords):
    result_words = []
    for word in wordlist:
        if word.lower() not in stopwords and word != ' ' and word != '\u3000':
            result_words.append(word)
    return result_words

def question_extract(question):
    #分词和去停用词
    stopwords_path = "keyword_extraction/english.txt"
    stopwords = stopword_list(stopwords_path)

    question_words = word_tokenize(question)
    filter_question_words = remove_stopwords(question_words,stopwords)
    return filter_question_words

def subtitle_extract(subtitle, question):
    question_words = question_extract(question)
    subtitle_words = [remove_stopwords(word_tokenize(sen),stopwords) for sen in sentences]

def extend_index(i,step,length):
    index_list = [i]
    for j in range(1,step):
        if i+j<length:
            index_list.append(i+j)
        if i-j>=0:
            index_list.append(i-j)
    return index_list

#计算wordslist总词数
def count_words(wordslist):
    count = 0
    for sen in wordslist:
        for w in sen:
            count = count + 1
    return count

def get_wordslist(subtitle_wordslist, index_list):
    now_wordslist = []
    for i, sen in enumerate(subtitle_wordslist):
        if i in index_list:
            now_wordslist.append(sen)
    return now_wordslist


#----------------------------计算问题相关性---------------------
#获得关键词语义列表（词语列表转成语义列表
def words_to_senses(wordslist,sentences):
    senseslist = []
    for i in range(len(wordslist)):
        words = wordslist[i]
        sentence = sentences[i]
        senses = []
        for w in words:
            sense = simple_lesk(sentence, w) #可指定pos='n'
            if not sense:
                sense = None
            else:
                senses.append(sense)
        senseslist.append(senses)
    return senseslist

#将标题词语列表转成语义列表
def title_words_to_senses(title_words,title):
    title_senses = []
    for tw in title_words:
        tw_sense = simple_lesk(title, tw)
        if not tw_sense:
            continue
        else:
            title_senses.append(tw_sense)
    return title_senses

#计算问题相关性(取最大值)
def title_sim(wordslist, senseslist, title_senses):
    title_map = {}
    for words in wordslist:
        for word in words:
            title_map[word]=0
    for i in range(len(senseslist)):
        senses = senseslist[i]
        for j in range(len(senses)):
            s = senses[j]
            if s:
                s_n = wordnet.synset(s.name())
                max_sim = 0
                for ts in title_senses:
                    ts_n = wordnet.synset(ts.name())
                    sim = s_n.path_similarity(ts_n)
                    #dist = sense_n.shortest_path_distance(tw_sense_n)
                    #sim = normalization(dist)
                    #print(s_n,ts_n,sim)
                    if not sim:
                        sim=0
                    if sim>max_sim:
                        max_sim=sim

                if title_map[wordslist[i][j]]<max_sim:
                    title_map[wordslist[i][j]] = max_sim

            # if s not in title_map:
            #     title_map[s] = max_sim
    return title_map

#----------------------------视觉信息相关性---------------------

#计算视觉相关性
def visual_sim(wordslist, filter_visual_words):
    visual_map = {}
    for i in range(len(wordslist)):
        words = wordslist[i]
        for w in words:
            if w not in visual_map:
                if w in filter_visual_words:
                    visual_map[w]=1
                else:
                    visual_map[w]=0

    return visual_map

#----------------------------语义重要性------------------------------------
#输入关键词列表（格式为[[a,b],[c,d],[e]]），根据共现关系，输出一个图
def get_graph(wordslist):
    words = []
    G = nx.Graph()
    for i in wordslist:
        for j in i:
            if j not in words:
                words.append(j)
                G.add_node(j)
    for wl in wordslist:
        for x in range(len(wl)):
            for y in range(x+1,len(wl)):
                G.add_edge(wl[x],wl[y])
    return G

#输入图，计算度中心性，介数中心性，相乘
def semantic_score(G, stopwords):
    semantic_map = {}
    degree_tuple = G.degree()
    bet_map = nx.betweenness_centrality(G)
    for t in degree_tuple:
        word = t[0]
        degree = t[1]
        bet = bet_map[word]
        if len(degree_tuple)>1:
            score = degree/(len(degree_tuple)-1) * bet
        else:
            score = 0
        semantic_map[word]=score * 100
        if word in stopwords:
            semantic_map[word]=0
    return semantic_map

#----------------------------计算总分值，并排序------------------------------------
def compute(question_map,visual_map,semantic_map):
    total_map = {}
    for word in visual_map:
        score1 = 0.3 * visual_map[word] + 0.7 * semantic_map[word]
        total_map[word] = score1

    return total_map

def extract_tvqa_features():
    start = 0
    end = None
    tvqa_dataset = get_tvqa("TVQA/tvqa_plus_annotations/tvqa_plus_val.json", start, end)
    print("Succeed in loading dataset!!!!!")


    for i, data in enumerate(tvqa_dataset):

        data_features = get_features(data)

        print(data_features.keywords_features)
        print("Succeed in getting features!!!")

        path = "TVQA/" + "val_features_soft/" + str(start+i) + ".pth"

        torch.save(data_features, path)


def extract_lqa_features():
    start = 0
    end = None
    lqa_dataset = get_lqa("LifeQA/lqa_test.json")
    print("Succeed in loading dataset!!!!!")

    for i, data in enumerate(lqa_dataset):

        data_features = get_features(data)

        print(data_features.question_type)
        print("Succeed in getting features!!!")

        path = "LifeQA/" + "test_features/" + str(start+i) + ".pth"

        torch.save(data_features, path)

def extract_old_tvqa_features():
    # start = 9677
    # end = 15253
    # tvqa_dataset = get_old_tvqa("TVQA-/tvqa_qa_release/tvqa_val.jsonl",start,end)
    # print("Succeed in loading dataset!!!!!")

    # for i, data in enumerate(tvqa_dataset):

    #     data_features = get_features(data)

    #     print(data_features.keywords_features)
    #     print("Succeed in getting features!!!")

    #     path = "TVQA-/" + "val_features_gt/" + str(start+i) + ".pth"

    #     torch.save(data_features, path)

if __name__ == '__main__':
    extract_lqa_features()
    # extract_tvqa_features()

    
    ###add keyword extraction
    ###chage qa to [cls] q [sep] a [sep]
    #### add lqa  answer index
    # extract features final soft