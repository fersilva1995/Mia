from keras_facenet import FaceNet
import numpy as np

class Embbeder:
    def __init__(self,):
        self.embedder = FaceNet()
        self.embeddings = []

    def get_embeddings(self, imgs):
        self.embeddings.clear()
        for face in imgs:
            self.embeddings.append(self.get_embedding(face)[0])


    def get_embedding(self, face_img):
        img = face_img.astype('float32') #3D (160x160x3)
        img = np.expand_dims(img, axis=0)
        yhat = self.embedder.embeddings(img)
        return yhat #512D image (1x1x512)

