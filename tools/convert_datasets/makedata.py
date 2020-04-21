import os
import shutil
path = 'D:\cvdata\MOT\DETRAC-train-data\Insight-MVT_Annotation_Train'
def walk(path):
    fileall = []
    if not os.path.exists(path):
        return -1
    for root,dirs,names in os.walk(path):
        for filename in names:
            #print(os.path.join(root,filename)) #路径和文件名连接构成完整路径
            fileall.append(os.path.join(root,filename))
    return fileall
if __name__=='__main__':
    file = walk(path)
    print(file)
    print(len(file))

#
# import os
# #第一部分，准备工作，拼接出要存放的文件夹的路径
# file = 'E:/测试/1.jpg'
# #current_foder是‘模拟’文件夹下所有子文件名组成的一个列表
    current_folder = file#current_foder是‘模拟’文件夹下所有子文件名组成的一个列表
    print(str(len(current_folder))+'个文件')
    # # 第二部分，将名称为file的文件复制到名为file_dir的文件夹中
    for x in current_folder:
        #拼接出要存放的文件夹的路径
        ls = x.split('\\')
        print(ls)
        file_dir = 'D:\cvdata\MOT\DETRAC_VOC\JPEGImages'+'\\'+str(ls[-2])+'__'+str(ls[-1])
        #将指定的文件file复制到file_dir的文件夹里面
        print('copy'+ x +' to ' + file_dir)
        shutil.copy(x,file_dir)