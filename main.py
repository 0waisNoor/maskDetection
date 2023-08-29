'''
load_mask() returns just one value

I have not defined image_reference function
'''

from xml.etree import ElementTree
import os
import numpy as np
import cv2
from matplotlib import pyplot
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes


class KangarooDataset:

    def __init__(self,dataset):
        self.dataset=dataset
        self.image_info={}


    def add_image(self,image_id,path,annotation):
        self.image_info[image_id] = (annotation,path)

    def load_dataset(self,dataset_dir,is_train=True):
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        
        for filename in os.listdir(images_dir):
            image_id = filename[:-4]
            
            if int(image_id)>=150 and is_train:
                continue

            if int(image_id)<150 and not is_train:
                continue


            if image_id in ['00090']:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image(image_id=image_id, path=img_path, annotation=ann_path)

    def getMask(self,image_id):
        #loads the bounding mask of an image
        boxes,width,height = self.extractBoxes(self.image_info[image_id][0])

        masks = np.zeros([height,width,len(boxes)],dtype='uint8')

        #creates the mask
        for i in range(0,len(boxes)):
            box = boxes[i]
            masks[box[1]:box[3],box[0]:box[2],i]=1

        return masks
    
    def extractBoxes(self,id):

        tree = ElementTree.parse(id)
        root = tree.getroot()
        boxes = root.findall('.//bndbox')

        xmin=0
        ymin=0
        xmax=0
        ymax=0
        bounding_boxes=[]
        for box in boxes:
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = (xmin,ymin,xmax,ymax)
            bounding_boxes.append(coors)


        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)

        return bounding_boxes,width,height


        




ds = KangarooDataset("kangaroo")
ds.load_dataset("kangaroo")
image_url = ds.image_info['00001'][1]
img  = cv2.imread(image_url)

mask = ds.getMask('00001')
box = extract_bboxes(mask)

print(mask)
print(box)

class_names=[]
class_ids=[]

masks=[]
boxes=[]

# for i in  ds.image_info:
#     class_names.append(i)
#     class_ids.append(i)
#     masks.append(ds.getMask(i))
#     boxes.append(extract_bboxes(ds.getMask(i)))



# class_names = np.array(class_names)
# class_ids = np.array(class_ids)
# boxes = np.array(boxes)



display_instances(img, box, mask, np.array(class_ids), np.array(class_names))