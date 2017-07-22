import numpy as np
import cv2

#This class is used to get various patches from the image of different sizes
class ImageSlicer:
    
    original_img = None
    new_img = None
    
    def __init__(self,img):
        
        #making two copies of the same image
        original_img = np.array(img)
        new_img = np.array(img)
        
        #resizing keeping the aspect ratio constant
        a_ratio = new_img.shape[0]/new_img.shape[1]
        #new_row=int(new_img.shape[0])
        new_row = 128
        new_colm = int(new_row/a_ratio)
        new_img = cv2.resize(new_img, (new_colm,new_row), interpolation = cv2.INTER_AREA)
        original_img = cv2.resize(original_img, (new_colm,new_row), interpolation = cv2.INTER_AREA)
        #convert new_one to grayscale
        new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
        
        
        self.original_img = original_img
        self.new_img = new_img
        
        
        
    def display(self):
        #display the images
        cv2.imshow("jj",self.original_img)
        cv2.waitKey(0)
        
        cv2.imshow("jj",self.new_img)
        cv2.waitKey(0)
    
    def sliding_windows(self,shape=(36,18),stride_horizontal=10,
                        stride_vertical=10):
        #this function cuts all slices from the image of a given size
        start = 0;
        end = shape[1]
        patches = []
        
        while True:
            
            if end>self.new_img.shape[1]:
                end=self.new_img.shape[1]

            start2 = 0
            end2 = shape[0]

            while True:
                if end2>self.new_img.shape[0]:
                    end2=self.new_img.shape[0]

                
                patches.append([self.new_img[start2:end2,start:end] , [start, start2, end, end2]])                          
                start2 += stride_vertical 
                end2 += stride_vertical
                
                if end2 > self.new_img.shape[0]:
                    break
                
            
            
            start += stride_horizontal
            end += stride_horizontal
            
            if end > self.new_img.shape[1]:
                break
            
        return patches #patches is a list containing different slides
        
    def get_all_slides(self, stride=10):
     
       #a function to get all slides with different shapes
       all_slides = []
       varying_shape = [36,18]
       for _ in range(8):
           all_slides.extend(self.sliding_windows(shape = tuple(varying_shape), stride_horizontal=stride,
                        stride_vertical = stride))
           varying_shape[0] = int(varying_shape[0]*1.2)
           varying_shape[1] = int(varying_shape[1]*1.2)
       
       return all_slides
        
    def display_slides(self):
        #display all slides when needed
        for _ in self.get_all_slides():
            cv2.imshow("patch",_[0])
            cv2.waitKey(0)
