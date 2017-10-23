#coding=utf-8
import	os
import	sys
sys.path.append('/new/xjl/caffe/python')
import	caffe
import	numpy	as	np
from	collections	import	Counter

deploy	=	'deploy.prototxt'	#	deploy文件
caffe_model	=	'/new/xjl/DIGITS-6/digits/jobs/20171021-154458-597e/snapshot_iter_62220.caffemodel'	#	训练好的	caffemodel

dir	=	'/new/xjl/work/caffe_test/alexnet_fcn/input/'
filelist	=	[]
filenames	=	os.listdir(dir)
for	fn	in	filenames:
    fullfilename	=	os.path.join(dir,	fn)
    filelist.append(fullfilename)

labels_filename	=	'labels.txt'	#	类别名称文件，将数字标签转换回类别名称
mean_file	=	'mean.npy'

def	Test(img):
    net	=	caffe.Net(deploy,	caffe_model,	caffe.TEST)	#	加载model和network

    im	=	caffe.io.load_image(img)	#	加载图片
    net.blobs['data'].reshape(1,im.shape[2],im.shape[0],im.shape[1])
#	图片预处理设置
    transformer	=	caffe.io.Transformer({'data':	net.blobs['data'].data.shape})	#	设定图片的shape格式(1,3,28,28)
#	im=im.transpose((2,0,1))
#	batch	=	im.reshape(-1,im.shape[0],im.shape[1],im.shape[2])
#	transformer	=	caffe.io.Transformer({'data':	batch.shape})	#	设定图片的shape格式
    transformer.set_transpose('data',	(2,	0,	1))	#	改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)

    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中

    #执行测试
    out = net.forward()

    labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
    softmax = net.blobs['softmax'].data[0] #.flatten() #取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称

    #print(softmax)
    print(softmax.shape)

    prob_map = np.amax(softmax,axis=0)       #nclass层中最大值（概率）
    order_map = np.argmax(softmax,axis=0)    #nclass层中最大值所在的序号

    print(prob_map.shape)
    print(order_map.shape)
    print(prob_map)
    print(order_map)

    # set thresold
    order_map[prob_map<0.95]=-1          #最大概率小于0.99的位置的序号变为-1
    print order_map

    key_value = Counter(order_map.flatten().tolist())      #返回键值对：order_map中 相同元素:出现次数
    key_value = sorted(key_value.iteritems(),key=lambda item:item[1],reverse=True)   #键值对按值降序

    # 38 x 26
    heatmap= np.zeros((38,26))
    for k,v in key_value:
        if k!=-1:
            k_label = labels[k]
            print('%s --> %d' % (k_label,v))
            point = k_label.split('x')      #序号转换为label
            heatmap[int(point[0])-1][int(point[1])-1]=v
    print(heatmap)
    np.savetxt('result.txt',heatmap,fmt='%d')

if __name__=='__main__':
    for path in filelist:
        Test(path)
