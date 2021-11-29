'''

Go to cmd/terminal and write following commands to run this script.

1. If you want to detect face from web camera with output path:
   python face_detection_from_webcam.py --out_path <Pass output path where output web camera video is stored>
   Example:
   python face_detection_from_webcam.py --out_path 'Output_Webcam/webcam_output_video.mov'

2. If you want to detect face from web camera without output path:
   python face_detection_from_webcam.py
   Example:
   python face_detection_from_webcam.py

'''


# Import Required Libraries
import face_recognition
import cv2
import numpy as np
import pickle
import argparse


class FaceDetectionFromWebCam:
    # Constructor - To initialize objects with values
    def __init__(self, output_path):
        self.output_path = output_path

    # This method is used to detect face from image
    def detect_face_from_web_cam(self):
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

        # Read from WebCam
        cap = cv2.VideoCapture(0)

        # Used for Downloading Video
        if self.output_path != None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, 20.0, (width,height))

        while cap.isOpened():
            # Read from WebCam by frames
            ret,frame = cap.read()

            # Grey image
            grey_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

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
                cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 102, 0), 2)
                cv2.rectangle(frame, (left, bottom + 18), (right, bottom), (0, 102, 0), cv2.FILLED)
                print(ref_dictt[id])
                if id == 'Unknown':
                    cv2.putText(frame, "Unknown", (left, bottom + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1, 3)
                else:
                    cv2.putText(frame, ref_dictt[id], (left, bottom + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1, 3)

            # Download Video
            if self.output_path != None:
                out.write(frame)
                
            # Show WebCam
            cv2.imshow('MoveNet Lightning', frame)
                
            # Stop the WebCam when 'q' is pressed on the keyboard
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release the WebCam
        cap.release()

        # Release Storing Video
        if self.output_path != None:
            out.release()

        # Destroy all windows
        cv2.destroyAllWindows()

    # Destructor - To delete all objects before code ends
    def __del__(self):
        print('Objects Deleted')


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument('--out_path', type=str, default=None, help='Pass path where to store output image') 

    # Parse the argument
    parsed_args = ap.parse_args()

    # Take parsed arguments and pass to detect_face_from_image() function to make detection
    output_path = parsed_args.out_path

    # Create FaceDetectionFromWebCam() class object, initialize values and call detect_face_from_web_cam() to detect face from web camera
    FaceDetectionFromWebCam(output_path).detect_face_from_web_cam()


# ----------------------------------------------------------------------END--------------------------------------------------------------------------