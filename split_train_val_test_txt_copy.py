# coding: utf-8
# author: YYH
# 2022-2-28

"""
该脚本用于根据数据集的xml格式的标签生成对应的图像名称
"""

import os
import random
random.seed(0)

# xml文件存放地址,在训练自己数据集的时候，改成自己的数据路径
xmlfilepath = '/home/yyh/Desktop/VisDrone2019/VisDrone2019-DET-val/VisDrone2019-DET-val/annotations_xml'
# 存放test.txt，train.txt，trainval.txt，val.txt文件路径
saveBasePath = "/home/yyh/Desktop/VisDrone2019/VisDrone2019-DET-val/VisDrone2019-DET-val"

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)
list = range(num)

fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    fval.write(name)

fval.close()
