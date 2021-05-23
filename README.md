# MMN

## 0.预处理
preprocess.py: 本来是打算用来预处理视频，用yolo提取视觉对象以及用c3d动作识别的。但是因为数据集有提供视觉对象，所以这个实际上没有用到。   

## 1.特征提取
（1）real_extract_features_att.py: 提取特征     
（2）extract_tvqa_gt_features.py：提取使用了GT时间的TVQA+的特征      
（3）new_extract_features_q.py: 也是提取特征，后来想添加一个字段所以重新提取了一遍，所以这个和（1）有一点点不同。    
  
* 代码里的old_tvqa实际上就表示TVQA数据集，关于这一部分的方法可以全部忽略。因为最后并没有用到。。  

## 2.模型训练
main1.py：训练的代码参考这个文件train()吧。。这部分最新版的代码没了orz。。   

## 3.预测
test_lqa.py：在lifeqa上的预测。    
test_tvqa.py：在TVQA+上的预测。模型结构的代码也看这个。  

* 实际上最重要的就是模型结构的代码，直接看test_tvqa.py里的BiNet就可以。
