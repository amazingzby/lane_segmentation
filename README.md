# 2020中国华录杯·数据湖算法大赛—定向算法赛（车道线识别）解决方案
本项目为2020中国华录杯·数据湖算法大赛车道线赛道解决方案。项目使用ocrnet网络结构，并采用数据增强，多分辨率融合，后处理等策略提升模型准确率。方案在复赛B榜中最终得分为43.60，取得复赛第1和决赛第3的成绩。
## 一.提交结果复现
**1.将B榜数据拷贝至 dataset/testB文件夹下，将初赛和复赛训练的图片和label分别拷贝至 dataset/trainval_pic 和 dataset/trainval_tag 文件夹下可运行本项目的训练，验证和测试代码</br>
2.运行python infer_submit.py && postProcess.py 复现B榜提交结果</br>
3.复赛B榜提交结果保存在 ./result/result_submit文件夹下,复现结果保存在 ./result/result 文件夹下，运行python check.py 检查每个图片复现结果和提交结果不同像素的数量</br></br>**
备注：提交结果和复现结果生成的差异如下：（1）.在本地生成提交结果时使用多卡（8卡）并行生成不同分辨率的预测结果，提交代码采用aistudio的运行环境（单卡）；（2）.多分辨率融合提交结果使用fp16计算，复现结果使用fp32计算；（3）.由于上述原因，使用check.py比较提交结果和复现结果，图片会有少量像素具有不同结果，但占比较少。提交结果和复现结果的少量差异理论上对最终分数影响较小，如果经测试差异较大，请联系本人进行代码复核。</br>
## 二.训练
修改config.py，设置配置参数，在config_factory中给出了本项目所采用的配置参数。运行train.py进行训练。运行infer.py获得单个模型的预测结果,运行infer_submit.py获得多模型融合的预测结果，运行postProcess.py对预测结果后处理。
## 三.竞赛思路
本次竞赛主要面临三个问题：模型准确性，数据不平衡，过拟合，解决以上三个问题对最终结果提高至关重要。</br>
1.模型准确性</br>
通过大量实验，对比deeplab3pro,hrnet,ocrnet在多个分辨率下的性能对比，发现ocrnet在相同分辨率下比其他模型最终得分高2分左右，因此项目最终使用ocrnet；采用多分辨率融合可以显著提高模型准确性，当融合3种分辨率时比单个模型有3-5分的提高，进一步增加融合数量时，对性能提升有限（0.5分左右提升）；使用膨胀腐蚀操作，消除分割图像中的孤立点,可提升0.5分左右。</br>
2.数据不平衡</br>
数据不平衡主要体现在两个方面：（1）.前景（车道线）数据和背景（非车道线）数据不平衡，改善前景数据和背景数据不平衡的方式为：使用weighted softmax 损失函数，背景类别（0类别）权重为0.5，其余类别权重为5，增大前景类别权重，减小背景类别权重；（2）.前景类别间数据不平衡，通过在训练集中复制多份少类别样本改善前景类别间数据不平衡,使用Lovasz-softmax 损失函数进一步改善数据不平衡问题。</br>
3.过拟合 </br>
多次训练调参提交结果，可以逐步提升算法在当前榜单的成绩，但当榜单数据切换（A,B榜切换，初赛复赛数据切换），同一参数在不同榜单的分数，名次都会发生一定改变，这是因为多次在同一榜单提交，模型超参数会在当前榜单过拟合，因此模型的泛化能力也是项目的重点考虑指标。为防止过拟合，训练原始学习率为0.0002，最终学习率为0.0001-0.00007，更小的学习率在验证集上具有更高的mIoU，但对测试集数据性能提升有限，甚至降低测试集mIoU;多分辨率模型融合，每个模型配置（最终学习率，数据增强策略）具有少量差异，通过加权多个模型结果可进一步提升算法的泛化能力。
## 四.代码模块介绍
1.data文件夹下定义了数据加载和数据预处理方法。数据预处理主要采用了颜色变换，图片缩放和宽高比缩放等数据增强操作，项目没采用目标分割常用的上下翻转和镜像翻转操作，原因：（1）.图片上下目标分布差异较大，上下翻转对模型的准确性提升有限；（2）.镜像翻转可能改变部分目标的类别（如类别9的左虚右实和10的左实右虚）。</br>
2.models 文件夹下定义了模型网络结构和损失函数，模型有deeplab3pro,hrnet和ocrnet。通过大量实验，ocrnet在不同分辨率下相比于其他模型均取得了最高的mIoU,因此在最终提交结果时使用ocrnet。</br>
3.dataset 文件夹存放数据集和数据集列表。训练和验证的A,B,C分别是对训练数据的三次随机切分，其中A,B,C三部分数据的验证集没用相互重复样本，以此保证所有数据均能被训练到；通过训练集和验证集在本地测试，并寻找最佳模型。竞赛后期，使用全部数据进行训练（名字包含trainval的txt文件）以进一步增加模型准确率，并对小样本（当前类别具有的数量小于500张）图片进行复制操作，以改善类别不均衡问题。</br>
4.tools 文件夹下包含了丰富的分析工具。md5用于分析数据重复问题，包含训练数据和测试数据重复（经分析发现A榜和训练集有少量重复，B榜和训练数据无重复），训练数据间重复（发现部分训练数据重复，对于多张重复图片，只保留1张作为训练样本）；img_class统计各类别（除去0类）包含的图片数量，以支持解决数据不平衡问题，show_images对各个类别数据进行可视化，以分析各类别数据的标注质量，分析数据分布等。</br>
更多功能详见源码
