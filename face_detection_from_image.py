'''

Go to cmd/terminal and write following commands to run this script.

1. If you want to detect face from image with output path:
   python face_detection_from_image.py --img_path <Pass image path> --out_path <Pass output path where output image is stored>
   Example:
   python face_detection_from_image.py --img_path 'test_images/yash_&_vihan.jpeg' --out_path 'Output_Image/yash_&_vihan_output.jpg'
   python face_detection_from_image.py --img_path 'test_images/barack.jpg' --out_path 'Output_Image/barack_output.jpg'

2. If you want to detect face from image without output path:
   python face_detection_from_image.py --img_path <Pass image path>
   Example:
   python face_detection_from_image.py --img_path 'test_images/yash.jpeg'
   python face_detection_from_image.py --img_path 'test_images/barack.jpg'

'''


# Import Required Libraries
import face_recognition
import cv2
import numpy as np
import pickle
import argparse


class FaceDetectionFromImage:
    # Constructor - To initialize objects with values
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

    # This method is used to detect face from image
    def detect_face_from_image(self):
        # Read ref_name pickle file
        name_file=open("ref_name.pkl","rb")
        ref_dictt=pickle.load(name_file)        
        name_file.close()

        # Read ref_embed pickle file
        embed_file=open("ref_embed.pkl","rb")
        embed_dictt=pickle.load(embed_file)      
        embed_file.close()

        known_face_encodings = []
        known_face_names = []

        # Storing known_face_encodings and known_face_names from readed pickle file
        for id, embed_list in embed_dictt.items():
            for my_embed in embed_list:
                known_face_encodings += [my_embed]
                known_face_names += [id]

        face_locations = []
        face_encodings = []
        face_names = []

        # Read image and convert it to grey image
        image = cv2.imread(self.image_path)
        grey_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # Get location of the face and encoding that face
        face_locations = face_recognition.face_locations(grey_img)
        face_encodings = face_recognition.face_encodings(grey_img, face_locations)

        # Comparing faces with known faces and getting name of that faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

        # Draw bounding box on detected faces and write name of face below it
        for (top_s, right, bottom, left), id in zip(face_locations, face_names):
            cv2.rectangle(image, (left, top_s), (right, bottom), (0, 102, 0), 2)
            cv2.rectangle(image, (left, bottom + 18), (right, bottom), (0, 102, 0), cv2.FILLED)
            print(ref_dictt[id])
            if id == 'Unknown':
                cv2.putText(image, "Unknown", (left, bottom + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1, 3)
            else:
                cv2.putText(image, ref_dictt[id], (left, bottom + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1, 3)

        # Save Image
        if self.output_path != None:
            cv2.imwrite(self.output_path, image)

        # Show Image
        cv2.imshow('Image', image)

        # Stop showing image if any key from keyboard is pressed
        cv2.waitKey(0)

    # Destructor - To delete all objects before code ends
    def __del__(self):
        print('Objects Deleted')


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument('--img_path', type=str, required=True, help='Pass path of image')
    ap.add_argument('--out_path', type=str, default=None, help='Pass path where to store output image') 

    # Parse the argument
    parsed_args = ap.parse_args()

    # Take parsed arguments and pass to detect_face_from_image() function to make detection
    image_path, output_path = parsed_args.img_path, parsed_args.out_path

    # Create FaceDetectionFromImage() class object, initialize values and call detect_face_from_image() to detect face from image
    FaceDetectionFromImage(image_path, output_path).detect_face_from_image()


# ----------------------------------------------------------------------END--------------------------------------------------------------------------