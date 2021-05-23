#coding: utf-8
from nltk.tokenize import word_tokenize,sent_tokenize
import cv2 as cv
import os
from subprocess import call

#处理vtt字幕文件，去除时间戳，拼接成文本，转换成txt文件
def vtt_to_txt(vtt_path,txt_path):
    with open(vtt_path,'r',encoding='utf-8') as file:
        lines = file.readlines()
        segments = []
        seg=[]
        for line in lines:
            if line=="\n":
                segments.append(seg[1:])
                seg = []
            else:
                seg.append(line.strip())
    textlist = []
    segments=segments[1:]
    for seg in segments:
        textlist = textlist+seg
    text = " ".join(textlist)
    with open(txt_path,'w',encoding='utf-8') as file:
        file.write(text)
    print(text)

#处理txt文件，将文本分句，返回句子的list
def txt_to_list(txt_path, type="subtitle"):
    sentences=[]
    #处理字幕文件
    if type=="subtitle":
        with open(txt_path,'r',encoding='utf-8') as file:
            txt = file.read()
            sentences = sent_tokenize(txt)
    #处理yolov3得到的视觉文件
    else:
        with open(txt_path,'r',encoding='utf-8') as file:
            lines = file.readlines()
            for word in lines:
                sentences.append(word.strip())
            
    print(sentences)
    return sentences

#处理句子list，将所有句子前后分别加上CLS和SEP，以输入BERT
def add_cls_and_sep(sentences):
    bert_sentences = []
    for sen in sentences:
        sen = "[CLS] " + sen + " [SEP]"
        bert_sentences.append(sen)
    print(bert_sentences)
    return bert_sentences

#采样提取视频帧
def get_frames(video_path,img_dir):
    video = cv.VideoCapture(video_path)
    fps=int(video.get(5)) #帧速率
    print("FPS: " + str(fps))
    success, frame = video.read()  #VideoCapture得到的图片是RGB空间
    img_id=1
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    while success:
        if img_id%fps==0:
            img_outpath = img_dir + "\\img" + str(img_id) + ".png"
            cv.imwrite(img_outpath, frame)
        img_id=img_id+1    
        success, frame = video.read()    #VideoCapture得到的图片是RGB空间

#使用yolov3处理图片,输出label.txt
def yolov3(img_dir):
    call(["python", "PyTorch-YOLOv3\\detect.py", "--image_folder", img_dir, 
        "--model_def", "PyTorch-YOLOv3\\config/yolov3.cfg","--weights_path", "PyTorch-YOLOv3\\weights\\yolov3.weights", 
        "--class_path", "PyTorch-YOLOv3\\data\\coco.names"])

def c3d():

if __name__ == "__main__":
    vtt_to_txt("testcut.vtt","testcut.txt")
    sentences = txt_to_list("testcut.txt")
    bert_sentences = add_cls_and_sep(sentences)

    get_frames("testcut.mp4","img")
    yolov3("img")
    #在yolov3的detect.py定义了输出文件为labels.txt
    labels = txt_to_list("labels.txt",type="visual")


