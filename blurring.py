import cv2

class Blurring:
    
    def __init__(self, type='gaussian'):
        self.type = type

    def execute(self, image ,value ):
        if(self.type == 'gaussian'):
            return cv2.GaussianBlur(image, (value,value), 0)
        elif(self.type == 'average'):
            return cv2.blur(image, (value,value),)
        elif(self.type == 'median'):
            return cv2.medianBlur(image, value)
        
        return image
