
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

            user_id = self.detectors['main'].predict(feature)[0]
            if(user_id in self.user_controller.users):
                selected_user = self.user_controller.users[user_id]
                svm = self.svm_controller.read(user_id)
                result = self.detectors[svm.id].predict(feature)[0]
                _, byte_array = cv2.imencode('.jpg', face)
                response_data = {}
                if(result == user_id):
                    response_data = {
                        'user_id': user_id,
                        'name': self.user_controller.users[user_id].name,
                        'face': byte_array
                    }
                else:
                    response_data = {
                        'user_id': user_id,
                        'name': self.user_controller.users[user_id].name + '-Negative',
                        'face': byte_array
                    }
                response.append(response_data)


        return response
                        


