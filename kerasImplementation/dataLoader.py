# @Author: Luc Blassel <zlanderous>
# @Date:   2018-01-18T22:47:47+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-18T23:15:14+01:00



import download
import shutil
import pickle
import os.path
import numpy as np
from scipy.misc import imsave

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
path = "data/CIFAR-10/"
batchesLoc = "cifar-10-batches-py"
batches = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5","test_batch"]
validation = "data_batch_5"
test = "test_batch"

meta = 'batches.meta'
img_size = 32
num_channels = 3


def convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images

def getLabels(meta):
    meta = os.path.join(path,batchesLoc,meta)
    labelsRaw = unpickle(meta)[b'label_names']
    labels = [x.decode('utf-8') for x in labelsRaw]
    return labels

def isLabel(element,wantedLabel):
    labels = getLabels(meta)
    ind = labels.index(wantedLabel)
    if element == ind:
        return True
    else:
        return False

def unpickle(filename):
    with open(filename, mode='rb') as toOpen:
        data = pickle.load(toOpen,encoding='bytes')
    return data


def dirCreator(meta,path):

    labels = getLabels(meta)
    for dataset in ['train','validation','test']:
        for label in labels:
            dataPath = os.path.join(path,dataset,label)
            if not os.path.exists(dataPath):
                print("creating" ,dataPath, "directory")
                os.makedirs(dataPath)


def batchLoader(batch):
    batchPath = os.path.join(path,batchesLoc,batch)
    raw = unpickle(batchPath)
    imagesRaw = raw[b'data']
    classes = raw[b'labels']
    sep = {}

    for i in set(classes):
        sep[i] =  []

    for i in range(len(classes)):
        sep[classes[i]].append(imagesRaw[i])

    for key in sep:
        sep[key] = convert_images(sep[key])

    return sep

def sepSaver(sep,setType,batch):
    labels = getLabels(meta)
    name = 1
    for key in sep:
        for img in sep[key]:
            classPath = os.path.join(path,setType,labels[key],batch+str(name))
            imsave(classPath+".png",img)
            name+=1


def downloader(url,path):
    download.maybe_download_and_extract(url,path)
    dirCreator(meta,path)

    for batch in batches:
        print(batch)
        sep = batchLoader(batch)
        if batch == validation:
            sepSaver(sep,'validation',batch)
        elif batch == test:
            sepSaver(sep,'test',batch)
        else:
            sepSaver(sep,'train',batch)

def main():
    downloader(url,path)

if __name__ == '__main__':
    main()
