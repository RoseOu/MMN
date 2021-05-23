# MMN

## 0.预处理
preprocess.py: 本来是打算用来预处理视频，用yolo提取视觉对象以及用c3d动作识别的。但是因为数据集有提供视觉对象，所以这个实际上没有用到。

## 1.特征提取
（1）real_extract_features_att.py: 提取特征
（2）extract_tvqa_gt_features.py：提取使用了GT时间的TVQA+的特征
（3）new_extract_features_q.py: 也是提取特征，后来想添加一个字段所以重新提取了一遍，所以这个和（1）有一点点不同。

## 2.训练模型
main1.py：参考这个吧。。这部分最新版的代码没了orz。。

## 3.预测
test_lqa.py：在lifeqa上的预测。
test_tvqa.py：在TVQA+上的预测。