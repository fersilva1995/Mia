
import cv2
from detect_image import DetectImage
from embbeder import Embbeder
import numpy as np
embbeder = Embbeder()


class RecognitionController():

    def __init__(self, user_controller, svm_controller):
        self.detector = DetectImage(640)
        self.user_controller = user_controller
        self.svm_controller = svm_controller
        self.detectors = {}

        self.reload()

    def reload(self):
        for svm in self.svm_controller.svms.values():
            self.detectors[svm.id] = self.svm_controller.read_model(svm.id)

    

    def recongnize(self, image):
        if('main' not in self.detectors):
            return
        
        response = []

        faces = self.detector.get_face(image)
        for face in faces:
            face_arr = cv2.resize(face, (160,160))
            rgb_img = cv2.cvtColor(face_arr, cv2.IMREAD_COLOR)
            feature = embbeder.get_embedding(rgb_img)

            face_name = self.detectors['main'].predict(feature)[0]
            if(face_name in self.user_controller.users):
                svm = self.svm_controller.read(face_name)
                result = self.detectors[svm.id].predict(feature)
                if(result[0] != 'negative'):
                    _, byte_array = cv2.imencode('.jpg', face)
                    response_data = {
                        'user_id': face_name,
                        'name': self.user_controller.users[face_name].name,
                        'face': byte_array
                    }
                    response.append(response_data)


        return response
                        


