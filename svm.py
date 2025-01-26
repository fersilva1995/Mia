from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import uuid;
from pathlib import Path
import os
import numpy as np

import threading
lock = threading.Lock()


class Svm:

    def __init__(self, data, name='', users=[], create_negative=True, user_id=0):
        if(data):
            self.name = data['name']
            self.id = data['id']
            self.users = data['users']
            self.create_negative = data['create_negative']
            self.user_id = data['user_id']
        else:
            if(name != 'main'):
                self.id = str(uuid.uuid4())
            else:
                self.id = 'main'
            self.name = name        
            self.users = list(users)
            self.create_negative = create_negative
            self.user_id = user_id


        

class SvmController:
    def __init__(self, user_controller):
        path = Path()
        base_dir = path.absolute().joinpath('svms')
        folders = [d.name for d in base_dir.iterdir() if d.is_dir()]
        self.user_controller = user_controller
        self.svms = {}

        for folder in folders:
            user_dir = base_dir.joinpath(folder)
            file_path = user_dir.joinpath('data.pickle')
            file = str(file_path)

            if(os.path.exists(file)):
                data = pickle.load(open(file, 'rb'))
                svm = Svm(data)
                self.svms[svm.id] = svm
            else:
                user = Svm(False, folder)
                self.svms[svm.id] = user

        if not 'main' in folders:
            main = self.create('main', user_controller.users)

        main_data_path = base_dir.joinpath('main').joinpath('main.pkl')
        if(not os.path.exists(main_data_path)):
           self.train('main')


    def save(self, id):
        path = Path()
        file = str(path.absolute().joinpath('svms').joinpath(id).joinpath('data.pickle'))
        with open(file, "wb") as f:
            pickle.dump(self.svms[id].__dict__, f)

    def create(self, name, users, create_negative=True, user_id=0):
        svm = Svm(False, name, users, create_negative, user_id)
        path = Path()
        dir_path = path.absolute().joinpath('svms').joinpath(svm.id)
        file_path = dir_path.joinpath('data.pickle')
        dir = str(dir_path)
        if(not os.path.exists(dir)):
            os.makedirs(dir)
            with open(str(file_path), "wb+") as f:
                pickle.dump(svm.__dict__, f)
            
            self.svms[svm.id] = svm

        return svm.id

    def read(self, user_id):
        for svm in self.svms.values():
            if(svm.user_id == user_id):
                return svm
            
    def read_model(self, svm_id):
        if(svm_id in self.svms):
            path = Path()
            base_dir = path.absolute().joinpath('svms')
            data_path = base_dir.joinpath(svm_id).joinpath(svm_id + '.pkl')
            if(os.path.exists(data_path)):
                svm_model = pickle.load(open(data_path, 'rb'))
                return svm_model

        return None



    def update(self, id, name, users, create_negative=True):
        if(id in self.svms):
            svm = self.svms[id]
            svm.name = name
            svm.create_negative = create_negative
            svm.users = list(users)
            self.save(id)

    def delete(self, id):
        if(id in self.svms):
            del self.svms[id]
            path = Path()
            dir_path = path.absolute().joinpath('svms').joinpath(id)
            file_path = dir_path.joinpath('data.pickle')
            if os.path.exists(file_path):
                os.remove(file_path)
        
            os.rmdir(dir_path)
        
            

    def train(self,id):
        if(id not in self.svms):
            return
        
        path = Path()
        user_controller = self.user_controller


        svm = self.svms[id]
        name = svm.name
        faces = []
        labels = []

        for user_id in svm.users:
            if(user_id in user_controller.users):
                user = user_controller.users[user_id]
                for face in user.face_features.values():
                    faces.append(np.array(face.value[0], dtype=float)) 
                    labels.append(user.id)

        if(svm.create_negative):
            for user_id in user_controller.users:
                if(user_id not in svm.users):
                    user = user_controller.users[user_id]
                    for face in user.face_features.values():
                        faces.append(np.array(face.value[0], dtype=float)) 
                        labels.append('negative')
            
        if(len(list(set(labels))) > 1):
            X_train, X_test, Y_train, Y_test = train_test_split(faces, labels, shuffle=True, random_state=17)
            model = SVC(kernel = 'linear', probability=True)
            model.fit(X_train, Y_train)
            ypreds_test = model.predict(X_test)
            ac = accuracy_score(Y_test, ypreds_test)
            print(ac)
            file = str(path.absolute().joinpath('svms').joinpath(id).joinpath(name + '.pkl'))
            with open(file,'wb') as f:
                pickle.dump(model,f)