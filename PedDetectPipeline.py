from ImageSlicer import ImageSlicer
from keras.models import model_from_json
import cv2
from PIL import Image
import numpy as np
import os
from skimage import color, exposure, transform


class PedDetPipeline:
    
    modified_image = None
    positive_slides = None
    
    def __init__(self, img, stride = 10):
         
         #give me the image and say no more! I'll do the rest ;)
         
         #Step1: get the slides
         slides = self.get_slides(img, stride = stride)
         
         #Step2: predict in each slides whether there is a pedestrian
         preds = self.predictall(slides) 
         
         #step3: Collect the slides in which there is a pedestrian
         positive_slides = self.analyse_predictions(slides, preds)
         #Ignore the next two lines. In order to prepare the original image, I have used 
         #the object of ImageSlicer class
         dummy= ImageSlicer(img)
         img = dummy.original_img
         
         #Step4: Draw rectangles around the slides that contain pedestrians, in the 'img'
         self.modified_image = self.prepare_final_image(positive_slides, img)
         
         
         
         
    def get_slides(self, img, stride = 10):
        im_slicer = ImageSlicer(img)
        slides=im_slicer.get_all_slides( stride )
        print(len(slides))
        return slides
    
    def load_model(self):
        if True:
            model_file = open('model2.json', 'r')
            loaded_model = model_file.read()
            model_file.close()
            model = model_from_json(loaded_model)
            #load weights into new model
            model.load_weights("weights2.hdf5")
            print("Model loaded successfully")
            return model
            
        else:
            print("Error importing the model. Make sure the model has\
            been trained\nand the model.json and weights.hdf5 files are in place, and try again.")
            os._exit(0)
    
    def predict(self, model, slide):
        
        #Flatten it
        image = np.array(slide).flatten()
      
        
        # float32
        image = image.astype('float32') 
        
        # normalize it
        image = image / 255
        
        # reshape for NN
        rimage = image.reshape(1, 36, 18,1)
        
        # Now feed it to the model, to fetch the predictions
        #clas = model.predict_proba(rimage)[0]
        prob_array = model.predict(rimage)
       
        if prob_array[0][0]>.99:
            print(prob_array)
            return prob_array[0][0]
        else:
            return 0
        
      
            
    def predictall(self, slides):
        
        #predict for all slides whether it contains a pedestrian
        model = self.load_model()
        preds=[]
        for slide in slides:
            slide = cv2.resize(slide[0], (18 , 36), interpolation = cv2.INTER_AREA)
            preds.append(self.predict(model, slide))
        
        return preds
    
    def analyse_predictions(self, slides, preds):
        
        #To take out the positive slides and abandon the rest to save memory
        positive_slides = []
        self.positive_slides = positive_slides
        for _ in range(len(slides)):
            if preds[_] != 0:
                positive_slides.append((slides[_][1], preds[_]))
                
        slides.clear(); slides = None
        preds.clear(); preds = None
        return positive_slides
    
        
    def non_max_suppression2(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []
        
        def get_overlap_area(r1 , r2 , area1, area2):
            xx1 = max(r1[0], r2[0])
            yy1 = max(r1[1], r2[1])
            xx2 = min(r1[2], r2[2])
            yy2 = min(r1[3], r2[3])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h)
            diff= abs(area1 -area2) +1
            return overlap/ diff
            
        boxes=np.array(boxes)
        preds=boxes[:,1]
        boxes=np.array(list(boxes[:,0]))
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        boxes=boxes
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        indexes = [_ for _ in range(len(boxes))]
        selected=[]; rejected=[]
        while len(indexes)>0:
            element=indexes[0]
            indexes.remove(indexes[0])
            for index in range(len(indexes)):
                o_ratio=get_overlap_area( (x1[element], y1[element], x2[element], y2[element]) ,
                                    (x1[indexes[index]], y1[indexes[index]], x2[indexes[index]], y2[indexes[index]]), 
                                    areas[element], areas[indexes[index]])
                if o_ratio > overlapThresh:
                    
                    if preds[element] >= preds[indexes[index]]:
                        selected.append(element)
                        rejected.append(indexes[index])
                    else:
                        selected.append(indexes[index])
                        rejected.append(element)                
                        break
                

        print(set(rejected))
        boxes= np.delete(boxes, list(rejected), axis=0)

        return boxes
        
    
        
    def non_max_suppression_slow(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # initialize the list of picked indexes
        pick = [];boxes=np.array(boxes);
        preds=boxes[:,1];boxes=np.array(list(boxes[:,0]))
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 1:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes, then initialize
            # the suppression list (i.e. indexes that will be deleted)
            # using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i);suppress=[]
            #suppress = [last]
            # loop over all indexes in the indexes list
            for pos in range(0, last):
                # grab the current index
                j = idxs[pos]
                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]
                
                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap > overlapThresh and preds[i] > preds[j]:
                    suppress.append(pos)
                elif overlap > overlapThresh and preds[i] < preds[j]:
                    suppress.append(last)
                else:
                    suppress.append(last)
            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)
        # return only the bounding boxes that were picked
        return boxes[pick]
        
        
    def prepare_final_image(self, slides, img):
        
        #mark the positive slides in the image
        slides2 = self.non_max_suppression2(slides,0.2)
        print(slides2)
        #for slide in slides:
         #   cv2.rectangle(img ,(slide[1][0],slide[1][1]),(slide[1][2],slide[1][3]),(0,0,255),1)
        for slide in slides2:
            cv2.rectangle(img ,(slide[0],slide[1]),(slide[2],slide[3]),(0,255,0),1)
        return img
        
    def get_result(self):
        
        #get the final result as an image
        self.modified_image = cv2.resize(self.modified_image,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        return self.modified_image
