import json
import uuid;
import os;
import pickle;
import cv2
import datetime
import random
import string
import numpy as np
from pathlib import Path
from face_generator import FaceGenerator
from detect_image import DetectImage
from embbeder import Embbeder

face_generator = FaceGenerator()
embbeder = Embbeder()


class Data: 
    def __init__(self, id, name, value, reference):
        self.id = id
        self.name = name
        self.value = value
        self.reference = reference


class User:

    def __init__(self, data, name = ''):
        
        if(data):
            self.name = data['name']
            self.id = data['id']
            self.images = data['images']
            self.aug_images = data['aug_images']
            self.audios = data['audios']
            self.faces = data['faces']
            self.face_features = data['face_features']
        else:
            self.name = name
            self.id = str(uuid.uuid4())
            self.images = {}
            self.audios = {}
            self.faces = {}
            self.aug_images = {}
            self.face_features = {}
            

class UserController:

    def __init__(self):
        self.detector = DetectImage(640)
        path = Path()
        base_dir = path.absolute().joinpath('user')
        folders = [d.name for d in base_dir.iterdir() if d.is_dir()]
        self.users = {}

        for folder in folders:
            user_dir = base_dir.joinpath(folder)
            file_path = user_dir.joinpath('data.pickle')
            file = str(file_path)

            if(os.path.exists(file)):
                data = pickle.load(open(file, 'rb'))
                user = User(data)
                self.users[user.id] = user
            else:
                user = User(False, folder)
                self.users[user.id] = user



    def generate_random_id(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        random_id = f"{timestamp}{random_string}"
        return random_id

    def save(self, id):
        path = Path()
        file = str(path.absolute().joinpath('user').joinpath(id).joinpath('data.pickle'))
        with open(file, "wb") as f:
            pickle.dump(self.users[id].__dict__, f)

        
    def create_user(self, name):
        user = User(False, name)
        path = Path()
        dir_path = path.absolute().joinpath('user').joinpath(user.id)
        file_path = dir_path.joinpath('data.pickle')
        dir = str(dir_path)
        if(not os.path.exists(dir)):
            os.makedirs(dir)
            with open(str(file_path), "wb+") as f:
                pickle.dump(user.__dict__, f)
            
            self.users[user.id] = user

        return user.id

    def edit_user(self, id, name):
        if(id in self.users):
            self.users[id].name = name
            self.save(id)

    def delete_user(self, id):
        del self.users[id]
        path = Path()
        dir_path = path.absolute().joinpath('user').joinpath(id)
        file_path = dir_path.joinpath('data.pickle')
        if os.path.exists(file_path):
            os.remove(file_path)
        
        os.rmdir(dir_path)

      
    def set_images(self, user_id, name, image):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]
        byte_array = []
        if os.path.isfile(image):
            img = cv2.imread(image)
            _, byte_array = cv2.imencode('.jpg', img)
            byte_array = byte_array.tobytes()
        elif isinstance(image, np.ndarray):
            _, byte_array = cv2.imencode('.jpg', image)
            byte_array = byte_array.tobytes()
        elif isinstance(image, bytes):
            byte_array = bytes(bytearray(image))
    
        image_id = self.generate_random_id()
        user.images[image_id] = Data(image_id, name, byte_array, '')
        self.save(user_id)
       
    def set_argumented_image(self, user_id, name, reference):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]
        
        if(reference not in user.images):
            return
        
        augmentations = face_generator.generate_random_augmentations()
        image = cv2.imdecode(np.frombuffer(user.images[reference].value, np.uint8), cv2.IMREAD_COLOR)
        output_image = face_generator.apply_augmentations(image, augmentations)
        _, byte_array = cv2.imencode('.jpg', output_image)
        image_id = self.generate_random_id()
        
        user.aug_images[image_id] = Data(image_id, name, byte_array, reference)
        self.save(user_id)

    def set_faces(self, user_id, name, reference):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]
        image = ''

        if(reference in user.images):
            image = user.images[reference]
        
        if(reference in user.aug_images):
            image = user.aug_images[reference]

        if(image == ''):
            return
        
        main_image = cv2.imdecode(np.frombuffer(image.value, np.uint8), cv2.IMREAD_COLOR)
        face = self.detector.get_face(main_image)[0]
        face = cv2.resize(face, (160,160))
        _, byte_array = cv2.imencode('.jpg', face)
        image_id = self.generate_random_id()

        user.faces[image_id] = Data(image_id, name, byte_array, reference)
        self.save(user_id)

    def set_feature(self, user_id, name, reference):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]

        if(reference not in user.faces):
            return

        loaded_image_ndarray = cv2.imdecode(user.faces[reference].value, cv2.IMREAD_COLOR)
        feature = embbeder.get_embedding(loaded_image_ndarray)
        image_id = self.generate_random_id()

        user.face_features[image_id] = Data(image_id, name, feature, reference)
        self.save(user_id)

    def remove(self, user_id , id, attr_name):
        if user_id not in self.users:
            return

        user = self.users[user_id]
        if not hasattr(user, attr_name):
            return

        attr = getattr(user, attr_name)
        if id not in attr:
            return

        del attr[id]
        self.save(user_id)

    def remove_image(self, user_id, id):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]
        if(id not in self.users[user_id].images):
            return
        
        del self.users[user_id].images[id]
        self.save(user_id)

    def remove_augmented_image(self, user_id, id):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]
        if(id not in self.users[user_id].aug_images):
            return
        
        del self.users[user_id].aug_images[id]
        self.save(user_id)

    def remove_faces(self, user_id, id):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]
        if(id not in self.users[user_id].faces):
            return
        
        del self.users[user_id].faces[id]
        self.save(user_id)

    def remove_feature(self, user_id, id):
        if(user_id not in self.users):
            return
        
        user = self.users[user_id]
        if(id not in self.users[user_id].face_features):
            return
        
        del self.users[user_id].face_features[id]
        self.save(user_id)


    def save_image(self, id, name, output):
        if(name in self.users[id].images):
            loaded_image_ndarray = cv2.imdecode(np.frombuffer(self.users[id].images[name], np.uint8), cv2.IMREAD_COLOR)
            p = Path(output)
            output_name = str(p.joinpath(name))
            cv2.imwrite(f'{output_name}.png', loaded_image_ndarray)

    def save_aug_images(self, id, name, output):
        if(name in self.users[id].aug_images):
            images = self.users[id].aug_images[name]
            for image in images:
                loaded_image_ndarray = cv2.imdecode(image['value'], cv2.IMREAD_COLOR)
                p = Path(output)
                output_name = str(p.joinpath(image['key'] + '.png'))
                cv2.imwrite(output_name, loaded_image_ndarray)

    def save_face(self, id, name, output):
        if(name in self.users[id].faces):
            for image in self.users[id].faces[name]:
                loaded_image_ndarray = cv2.imdecode(image['value'], cv2.IMREAD_COLOR)
                p = Path(output)
                output_name = str(p.joinpath(image['key'] + '.png'))
                cv2.imwrite(output_name, loaded_image_ndarray)


    

'''

controller = UserController()
path = Path()
local_path = path.absolute()
selected_id = ''


while True:
    print("\n=== Main Menu ===")
    print("1. User")
    print("2. Select")
    print("3. Show selected")
    print("4. Save")
    print("5. Image")
    print("6. Augmentation")
    print("7. Face")
    print("8. Features")
    print("0. Exit")

    choice = input("Enter your choice: ")

    if choice == '1':
        print("\n=== User Menu ===")
        print("1. Create")
        print("2. Update")
        print("3. Delete")
        choice = input("Enter your choice: ")


        if choice == '1':
            name = input("Enter your Name: ")
            controller.create_user(name)

        elif choice == '2':
            name = input("Enter new name to selected user: ")
            controller.users[selected_id].name = name
            controller.save(selected_id) 

        elif choice == '3':
            controller.delete_user(selected_id)

    elif choice == '2':
        counter = 1
        for key, value in controller.users.items():
            print(f"{counter} - Id : {key} - Name: {value.name}")
            counter += 1
        
        id_index = input("Select id by index: ")
        selected_id = list(controller.users)[int(id_index)-1]

    elif choice == '3':
        print(f"Id : {selected_id} - Name: {controller.users[selected_id].name}")

    elif choice == '4':
        controller.save(selected_id) 

    elif choice == '5':
        print("\n=== Image Menu ===")
        print("1. New")
        print("2. Save")
        print("3. Delete")
        print("0. Return")

        choice = input("Enter your choice: ")

        if choice == '1':
            image_path = input('Enter image path: ')
            name = input('Name: ')
            controller.set_images(selected_id, name, image_path)
            controller.save(selected_id)

        elif choice == '2':
            counter = 1
            if(len(controller.users[selected_id].images) > 0):
                for value in controller.users[selected_id].images:
                    print(f"{counter} - Name: {value}")
                    counter += 1
            
                image_index = input("Select image by index: ")
                output_path = input('Enter image path to save: ')
                controller.save_image(selected_id, list(controller.users[selected_id].images)[int(image_index)-1], output_path)

        elif choice == '3':
            counter = 1
            if(len(controller.users[selected_id].images) > 0):
                for value in controller.users[selected_id].images:
                    print(f"{counter} - Name: {value}")
                    counter += 1
            
                image_index = input("Select image by index: ")
                name = list(controller.users[selected_id].images)[int(image_index)-1]
                del controller.users[selected_id].images[name]
            controller.save(selected_id)

    elif choice == '6':
        print("\n=== Aug Menu ===")
        print("1. New")
        print("2. Save")
        print("3. Delete")
        print("0. Return")

        choice = input("Enter your choice: ")

        if choice == '1':
            counter = 1
            if(len(controller.users[selected_id].images) > 0):
                for value in controller.users[selected_id].images:
                    print(f"{counter} - Name: {value}")
                    counter += 1
            
                image_index = input("Select image by index: ")
                images_to_generate = input('Number of images to generate: ')
                name = list(controller.users[selected_id].images)[int(image_index)-1]
                for counter in range(int(images_to_generate)):
                    controller.set_argumented_image(selected_id, name)
             
            controller.save(selected_id)

        elif choice == '2':
            counter = 1
            for value in controller.users[selected_id].aug_images:
                print(f"{counter} - Name: {value}")
                counter += 1
        
            image_index = input("Select image by index: ")
            selected =  list(controller.users[selected_id].images)[int(image_index)-1]

            output_path = input('Enter image path to save: ')
            
            controller.save_aug_images(selected_id, selected, output_path)

        elif choice == '3':
            counter = 1
            for value in controller.users[selected_id].aug_images:
                print(f"{counter} - Name: {value}")
                counter += 1
        
            image_index = input("Select image by index: ")
            selected =  list(controller.users[selected_id].images)[int(image_index)-1]

            del controller.users[selected_id].aug_images[name]
            controller.save(selected_id)

    elif choice == '7':
        print("\n=== Face Menu ===")
        print("1. New")
        print("2. Save")
        print("3. Delete")
        print("0. Return")

        choice = input("Enter your choice: ")

        if choice == '1':
            counter = 1
            if(len(controller.users[selected_id].images) > 0):
                for value in controller.users[selected_id].images:
                    print(f"{counter} - Name: {value}")
                    counter += 1
            
                image_index = input("Select image by index: ")
                name = list(controller.users[selected_id].images)[int(image_index)-1]
                controller.set_faces(selected_id, name)
             
            controller.save(selected_id)

        elif choice == '2':
            counter = 1
            for value in controller.users[selected_id].aug_images:
                print(f"{counter} - Name: {value}")
                counter += 1
        
            image_index = input("Select image by index: ")
            selected =  list(controller.users[selected_id].images)[int(image_index)-1]

            output_path = input('Enter image path to save: ')
            
            controller.save_face(selected_id, selected, output_path)

        elif choice == '3':
            counter = 1
            for value in controller.users[selected_id].aug_images:
                print(f"{counter} - Name: {value}")
                counter += 1
        
            image_index = input("Select image by index: ")
            selected =  list(controller.users[selected_id].images)[int(image_index)-1]

            del controller.users[selected_id].faces[name]
            controller.save(selected_id)

    elif choice == '8':
        print("\n=== Feature Menu ===")
        print("1. New")
        print("2. Delete")
        print("0. Return")

        choice = input("Enter your choice: ")

        if choice == '1':
            counter = 1
            if(len(controller.users[selected_id].images) > 0):
                for value in controller.users[selected_id].images:
                    print(f"{counter} - Name: {value}")
                    counter += 1
            
                image_index = input("Select image by index: ")
                name = list(controller.users[selected_id].images)[int(image_index)-1]
                controller.set_feature(selected_id, name)
             
            controller.save(selected_id)

        elif choice == '2':
            counter = 1
            for value in controller.users[selected_id].aug_images:
                print(f"{counter} - Name: {value}")
                counter += 1
        
            image_index = input("Select image by index: ")
            selected =  list(controller.users[selected_id].images)[int(image_index)-1]

            del controller.users[selected_id].features[name]
            controller.save(selected_id)



    elif choice == '0':
        print("Exiting the program. Goodbye!")
        break
    else:
        print("Invalid choice. Please try again.")



#controller.create_user('Fernando')

'''


'''


###IMPORT IMAGES
file_path = 'faces/fernando'
base_path = local_path.joinpath(file_path).absolute()
for file in os.listdir(str(base_path)):
    controller.set_images(os.path.splitext(file)[0], str(base_path.joinpath(file).absolute()))
controller.save()



#CHECK IMPORTED IMAGES
file_path = 'output_faces'
base_path = local_path.joinpath(file_path).absolute()

for key in Fernando.user.images:
    Fernando.save_image(key, str(base_path))


file_path = 'output_faces'
base_path = local_path.joinpath(file_path).absolute()

for key in Fernando.user.images:
    for counter in range(10):
        Fernando.set_argumented_image(key)
Fernando.save()



###CHECK AUG IMAGES
file_path = 'output_faces'
base_path = local_path.joinpath(file_path).absolute()

for key in Fernando.user.images:
    Fernando.save_aug_images(key, str(base_path))



###CHECK AUG IMAGES
for key in Fernando.user.images:
    Fernando.set_faces(key)
Fernando.save()


###CHECK FACES
file_path = 'output_faces2'
base_path = local_path.joinpath(file_path).absolute()

for key in Fernando.user.images:
    Fernando.save_face(key, str(base_path))



for key in Fernando.user.faces:
    Fernando.set_feature(key)

Fernando.save()
'''








    