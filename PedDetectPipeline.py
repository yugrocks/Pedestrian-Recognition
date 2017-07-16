from ImageSlicer import ImageSlicer
from keras.models import model_from_json
import cv2
from PIL import Image
import numpy as np

class PedDetPipeline:
    
    modified_image = None
    positive_slides = None
    
    def __init__(self, img):
         
         #give me the image and say no more! I'll do the rest ;)
         
         #Step1: get the slides
         slides = self.get_slides(img, stride = 10)
         
         #Step2: predict in each slides whether there is a pedestrian
         preds = self.predictall(slides) 
         
         #step3: Collect the slides in which there is a pedestrian
         positive_slides = self.analyse_predictions(slides, preds)
         
         #Ignore the next two lines. In order to prepare the original image, I have used 
         #the object of ImageSlicer class
         dummy= ImageSlicer(img)
         img = dummy.new_img
         
         #Step4: Draw rectangles around the slides that contain pedestrians, in the 'img'
         self.modified_image = self.prepare_final_image(positive_slides, img)
         
         
         
         
    def get_slides(self, img, stride = 10):
        im_slicer = ImageSlicer(img)
        slides=im_slicer.get_all_slides( stride )
        print(len(slides))
        return slides
    
    def load_model(self):
        try:
            model_file = open('model.json', 'r')
            loaded_model = model_file.read()
            model_file.close()
            model = model_from_json(loaded_model)
            #load weights into new model
            model.load_weights("weights.hdf5")
            print("Model loaded successfully")
            return model
            
        except:
            print("Error importing the model. Make sure the model has\
            been trained\nand the model.json and weights.hdf5 files are in place, and try again.")
            exit()
    
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
       
        if prob_array[0][0]>.995:
            print(prob_array)
            return 1
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
            if preds[_] == 1:
                positive_slides.append(slides[_])
                
        slides.clear(); slides = None
        preds.clear(); preds = None
        return positive_slides
    
        
        
    def prepare_final_image(self, slides, img):
        
        #mark the positive slides in the image
        for slide in slides:
            cv2.rectangle(img ,(slide[1][0],slide[1][1]),(slide[1][2],slide[1][3]),(0,0,255),1)
        return img
        
    def get_result(self):
        
        #get the final result as an image
        self.modified_image = cv2.resize(self.modified_image,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        return self.modified_image
            
            
            
            
            
            
            
