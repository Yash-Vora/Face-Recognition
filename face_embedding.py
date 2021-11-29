'''

Go to cmd/terminal and write following commands to run this script.

1. If you want to train model with different class images:
   python face_embedding.py --class_name <Enter unique class name> --id <Enter unique id> --img_folder_path <Enter image folder path>
   Example:
   python face_embedding.py --class_name 'Yash' --id 1 --img_folder_path 'train_images/Yash/'
   python face_embedding.py --class_name 'Barack' --id 2 --img_folder_path 'train_images/Barack/'
   python face_embedding.py --class_name 'Vihan' --id 3 --img_folder_path 'train_images/Vihan/'

'''


# Import Required Libraries
import os
import cv2 
import face_recognition
import pickle
import argparse


class FaceEmbeddings:
    # Constructor - To initialize objects with values
    def __init__(self, class_name, id, image_folder_path):
        self.class_name = class_name
        self.id = id
        self.image_folder_path = image_folder_path

    # It is used to train model with images
    def train_with_image_class(self):
        try:
            name_file = open("ref_name.pkl","rb")
            ref_dictt = pickle.load(name_file)
            name_file.close()
        except:
            ref_dictt = {}

        if self.id not in ref_dictt:
            ref_dictt[self.id] = self.class_name

            name_file = open("ref_name.pkl","wb")
            pickle.dump(ref_dictt,name_file)
            name_file.close()

            try:
                embed_file = open("ref_embed.pkl","rb")
                embed_dictt = pickle.load(embed_file)
                embed_file.close()
            except:
                embed_dictt = {}

            img_list = os.listdir(self.image_folder_path)

            if img_list != []:
                for i in img_list:
                    image = cv2.imread(self.image_folder_path+i)
                    grey_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

                    face_location = face_recognition.face_locations(grey_img)
                    if face_location != []:
                        face_encoding = face_recognition.face_encodings(grey_img)[0]
                        if self.id in embed_dictt:
                            embed_dictt[self.id] += [face_encoding]
                        else:
                            embed_dictt[self.id] = [face_encoding]

                if embed_dictt != {}:
                    embed_file=open("ref_embed.pkl","wb")
                    pickle.dump(embed_dictt,embed_file)
                    embed_file.close()
            else:
                print('The image folder you have passed is empty')
        else:
            print('Please Enter Unique Id')

    # Destructor - To delete all objects before code ends
    def __del__(self):
        print('Objects Deleted')


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument('--class_name', type=str, required=True, help='Enter name')
    ap.add_argument('--id', type=int, required=True, help='Enter id')
    ap.add_argument('--img_folder_path', type=str, required=True, help='Pass path of image folder')

    # Parse the argument
    parsed_args = ap.parse_args()

    # Take parsed arguments and pass to detect_face_from_image() function to make detection
    class_name, id, image_folder_path = parsed_args.class_name, parsed_args.id, parsed_args.img_folder_path
    
    # Create FaceEmbeddings() class object, initialize values and call train_with_image_class() to train face from image
    FaceEmbeddings(class_name, id, image_folder_path).train_with_image_class()


# ----------------------------------------------------------------END--------------------------------------------------------------------------------