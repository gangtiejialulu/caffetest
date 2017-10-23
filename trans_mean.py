#coding=utf-8
#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('/new/xjl/caffe/python')
import caffe

root=''
binary_path = root + 'mean.binaryproto' #binaryproto文件路径
npy_path = root + 'mean.npy' #转化后保存路径

blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open(binary_path , 'rb' ).read()
blob.ParseFromString(bin_mean)
arr = np.array( caffe.io.blobproto_to_array(blob) )
npy_mean = arr[0]
#如果图像输入数据，归一化到0-1，要除以255，如果没有则不需要
# npy_mean = npy_mean/255
np.save(npy_path , npy_mean )

