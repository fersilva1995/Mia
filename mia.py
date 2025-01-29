from concurrent import futures
import grpc
import Mia_pb2
import Mia_pb2_grpc
import time
from user import UserController
from svm import SvmController
from recognition import RecognitionController
import threading
import numpy as np
import cv2 

user_controller = UserController()
svm_controller = SvmController(user_controller)
rec_controller = RecognitionController(user_controller, svm_controller)

delay = 1

class TrainThread(threading.Thread):
    def __init__(self, user_id='main'):
        self.user_id = user_id
        threading.Thread.__init__(self)

    def run(self):
        svm_controller.train(self.user_id)
        
class MiaService(Mia_pb2_grpc.MiaService):

    #region User
    def ReadUsers(self, request, context):
        data = [ Mia_pb2.UserData(name= user.name, id= user.id) for user in user_controller.users.values()]
        response = Mia_pb2.GetUsersResponse(data= data)
        return response
    
    def CreateUser(self, request, context):
        name = request.name
        user_id = user_controller.create_user(name)
        response = Mia_pb2.GetUsersResponse()

        svm_controller.update('main', 'main', user_controller.users)
        training = TrainThread()
        training.start()

        svm_controller.create(user_id, [user_id], user_id=user_id)
        training = TrainThread(user_id)
        training.start()

        return response
    
    def UpdateUser(self, request, context):
        user_controller.edit_user(request.id, request.name)
        response = Mia_pb2.GetUsersResponse()
        return response

    def DeleteUser(self, request, context):
        user_id = request.id
        user_controller.delete_user(user_id)
        response = Mia_pb2.GetUsersResponse()

        svm_controller.update('main', 'main', user_controller.users)
        training = TrainThread()
        training.start()

        svm_controller.delete(user_id)

        return response
    

    #endregion

    #region Svm

    def ReadSvms(self, request, context):
        data = [ Mia_pb2.SvmData(name= svm.name, 
                                id= svm.id,
                                users= svm.users, 
                                create_negative = svm.create_negative, 
                                create_unknown = svm.create_unknown) 
                                for svm in svm_controller.svms.values()]
        response = Mia_pb2.GetSvmsResponse(data= data)
        return response
    
    def CreateSvm(self, request, context):
        svm_controller.create(request.name, request.users, request.create_negative, request.create_unknown)
        response = Mia_pb2.GetSvmsResponse()
        return response
    
    def UpdateSvm(self, request, context):
        svm_controller.update(request.id, request.name, request.users, request.create_negative, request.create_unknown)
        response = Mia_pb2.GetSvmsResponse()
        return response

    def DeleteSvm(self, request, context):
        svm_controller.delete(request.id)
        response = Mia_pb2.GetSvmsResponse()
        return response
    
    def TrainSvm(self, request, context):
        response = svm_controller.train(request.id)
        return Mia_pb2.MiaResponse(response=response)

    #endregion

    #region Image
    def ReadIndex(self, request, context):
        user_id = request.user_id
        source = request.source

        response = Mia_pb2.IndexResponse()
        data = []

        if(user_id not in user_controller.users):
            return
        
        user = user_controller.users[user_id]
        values = getattr(user, source).values()

        for value in values:
            data.append(Mia_pb2.ImageData( id = value.id, name = value.name))

        response.data.extend(data)
        return response
    
    def ReadData(self, request, context):
        user_id = request.user_id
        source = request.source
        ids = request.ids

        if(user_id not in user_controller.users):
            return
        
        user = user_controller.users[user_id]

        items = getattr(user, source).items()
        for id, item in items:
            if(id in ids):
                data = Mia_pb2.ImageData()
                data.id = id
                if isinstance(item.value, bytes):
                    data.image_bytes = item.value
                else:
                    data.image_bytes = item.value.tobytes()
                yield data
                time.sleep(delay)

        return
                
    def SetData(self, request_iterator, context):
        response = ''
        for request in request_iterator:
            try:
                user_id = request.id
                source = request.source
                name = request.name
                reference = request.reference
                image = request.image_bytes

                if(source == "images"):
                    user_controller.set_images(user_id, name, image)

                if(source == "aug_images"):
                    user_controller.set_argumented_image(user_id, name, reference)

                if(source == "faces"):
                    user_controller.set_faces(user_id, name, reference)

                if(source == "face_features"):
                    user_controller.set_feature(user_id, name, reference)
        


                response += 'sucess,'
            except Exception as e:
                response += 'fail,'

        if(source == "face_features"):
            svm = svm_controller.read(user_id)
            training = TrainThread(svm.id)
            training.start()

            if(user_id != 'unknown'):
                training = TrainThread('unknown')
                training.start()
        
        return Mia_pb2.MiaResponse(response=response)

    def RemoveData(self, request, context):
        try:
            user_id = request.user_id
            source = request.source
            ids = request.ids
            for id in ids:
                user_controller.remove(user_id, id, source)

            if(source == "face_features"):
                svm = svm_controller.read(user_id)
                training = TrainThread(svm.id)
                training.start()

                if(user_id != 'unknown'):
                    training = TrainThread('unknown')
                    training.start()


            return Mia_pb2.MiaResponse(response='success')
        except Exception as e:
             return Mia_pb2.MiaResponse(response=str(e)) 


    #endregion

    def RecognizeSingle(self, request_iterator, context):
        for request in request_iterator:
            image_bytes = request.image
            image_name = request.image_name
            threshold = request.threshold
            dectector = request.svm_id
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            response = rec_controller.recongnize_single(image, dectector, threshold)

            for message in response:
                server_message = Mia_pb2.RecognitionResponse (
                    user_id = message['user_id'], 
                    name = message['name'],
                    #image = message['face'].tobytes(),
                    image_name = image_name,
                )

                yield server_message
                time.sleep(delay)

    def Recognize(self, request_iterator, context):
        for request in request_iterator:
            image_bytes = request.image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            response = rec_controller.recongnize(image)

            for message in response:
                server_message = Mia_pb2.RecognitionResponse (
                    user_id = message['user_id'], 
                    name = message['name'],
                    image = message['face'].tobytes(),
                )

                yield server_message
                time.sleep(delay)
       

def serve():

    options = [('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MiB
           ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    Mia_pb2_grpc.add_MiaServiceServicer_to_server(MiaService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started on port 50051")
    server.start()
    server.wait_for_termination()

serve()












