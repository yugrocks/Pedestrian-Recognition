from PIL import Image
from PedDetectPipeline import PedDetPipeline as pdp
import cv2

def main():
    
    #make a pipeline object
    pl = pdp( Image.open(r"ped14.png") )
    img = pl.get_result()
            
    #show the image
    cv2.imshow("",img)
    cv2.waitKey(0)
    
    
main()