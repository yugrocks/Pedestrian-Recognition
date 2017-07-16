from PIL import Image
from sklearn.cross_validation import train_test_split
import os
from sklearn.utils import shuffle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, Flatten, MaxPooling2D
from keras import regularizers
import cv2

img_rows=36
img_colms=18
img_channels=1



def init_dataset():
    X=[];y=[]
    
    #positive examples
    for _ in os.listdir("Dataset/training_set/ped"):
        X.append(np.array(Image.open("Dataset/training_set/ped/"+_)).flatten())
        y.append(1)
        
    #for _ in os.listdir("Dataset/training_set2/ped"):
     #   X.append(np.array(Image.open("Dataset/training_set2/ped/"+_)).flatten())
      #  y.append(1)
        
    for _ in os.listdir(r"C:\Users\Yugal\Anaconda3\Projects\The pedestrians dataset128x64"):
        
        img=cv2.cvtColor(cv2.resize(np.array(Image.open(r"C:\Users\Yugal\Anaconda3\Projects\The pedestrians dataset128x64/"+_)), (18,36), interpolation = cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        X.append(img.flatten())
        y.append(1)
    
    #negative examples    
    for _ in os.listdir("Dataset/training_set/non_ped"):
        X.append(np.array(Image.open("Dataset/training_set/non_ped/"+_)).flatten())
        y.append(0)
    
    for _ in os.listdir("Dataset/training_set2/non_ped"):
        X.append(np.array(Image.open("Dataset/training_set2/non_ped/"+_)).flatten())
        y.append(0)
        
    
    #shuffle
    X,y = shuffle(X,y,random_state=2)
    
    #split into train and test sets:
    X_train, X_test, y_train, y_test=train_test_split(X,y,
                                                      test_size=0.0,
                                                      random_state=4)
    
    #They were lists, so convert to np arrays
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    
    #reshape every image inside
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_colms, img_channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_colms, img_channels)
    
    #normalize
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    #y_train = np_utils.to_categorical(y_train, 2)
    #y_test = np_utils.to_categorical(y_test, 2)
    
    
    return X_train, X_test , y_train, y_test
    
    

    
def make_model():
    
    model=Sequential()
    
    #add conv layers and pooling layers 
    model.add(Convolution2D(32,3,3, input_shape=(36,18,1),activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    
    #add conv layers and pooling layers 
    model.add(Convolution2D(32,3,3, input_shape=(36,18,1),activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.5)) #to reduce overfitting
    
    model.add(Flatten())
    
    #Now two hidden(dense) layers:
    model.add(Dense(output_dim = 64, activation = 'relu',
                    #kernel_regularizer=regularizers.l2(0.03)
                    ))
    
    model.add(Dropout(0.5))#again for regularization
    
    model.add(Dense(output_dim = 64, activation = 'relu',
                    #kernel_regularizer=regularizers.l2(0.03)
                    ))
    
    
    model.add(Dropout(0.5))#last one lol
    
    #output layer
    model.add(Dense(output_dim = 1, activation = 'sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    model.get_config()
    
    return model
    


def train_CNN():
    
    #get model:
    model=make_model()
    
    #get datasets:
    X_train, X_cv, Y_train, Y_cv = init_dataset()
    
    #start training:
    history = model.fit(X_train, Y_train, batch_size=32, epochs=10,
verbose=1, validation_split=0.2)
    
    return history, model
    
    
    
history, model = train_CNN()

#saving the weights
model.save_weights("weights.hdf5",overwrite=True)

#saving the model itself in json format:
model_json = model.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")


"""after 10 epochs:
    - loss: 0.3406 
    - acc: 0.9525 
    - val_loss: 0.2876 
    - val_acc: 0.9716
"""

    
def predict(model, img):
    
    #Flatten it
    image = np.array(img).flatten()
  
    
    # float32
    image = image.astype('float32') 
    
    # normalize it
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_rows, img_colms,img_channels)
    
    # Now feed it to the NN, to fetch the predictions
    clas = model.predict(rimage)
    #prob_array = model.predict_proba(rimage)

    return  clas

    
y_pred=[]
y=[]
for _ in os.listdir(r"C:\Users\Yugal\Anaconda3\Projects\The pedestrians dataset 2\T1\ped_examples"):
        img=Image.open(r"C:\Users\Yugal\Anaconda3\Projects\The pedestrians dataset 2\T1\ped_examples/"+_)
        
        #img = cv2.resize(np.array(img), (18,36), interpolation = cv2.INTER_AREA)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        y_pred.append(predict(model,img))
        y.append(1)


from sklearn.metrics import confusion_matrix


cm=confusion_matrix(y, y_pred)











