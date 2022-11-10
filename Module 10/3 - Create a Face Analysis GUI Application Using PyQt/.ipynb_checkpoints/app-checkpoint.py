from PyQt5.QtGui import QPixmap, QColor

from PyQt5 import uic, QtCore,Q tGui, QtWidgets


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


import os
import cv2
import sys
import json
import qdarkstyle
import tensorflow as tf
from deepface import DeepFace



from previous_lesson import *
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QApplication, QSplashScreen 
from PyQt5.QtCore import QTimer





# Load the trained age and gender estimation model.
age_gender_model = tf.keras.models.load_model(r'models/EfficientNetB3_224_weights.11-3.44.hdf5')

# Build the deepface Race prediction model.
race_model = DeepFace.build_model('Race')

# Load the hair segmentation model.
hair_model = tf.keras.models.load_model(r'models/hairnet_matting_30.hdf5')

# Initialize the mediapipe face mesh class.
mp_face_mesh = mp.solutions.face_mesh

# Set up the face mesh function with appropriate arguments.
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.8, min_tracking_confidence=0.5)

images = []
eye_color_list = []
hair_color_list = []
gender_list = []
age_list = []
race_list = []

male = []
female = []

average_results_dic = {}



class App(QMainWindow):

    def __init__(self):
        super(App, self).__init__()

        # Load the ui file
        uic.loadUi("app.ui", self)

        self.setWindowTitle("AI Executable GUI App")
        # self.setStyleSheet("background-color: black;")
        # self.tabWidget.setStyleSheet('QTabBar::tab {background-color: black;}')

        self.splash_picture = QPixmap('assets/splash_screen.jpg')

        self.splash = QSplashScreen(self.splash_picture)
        self.splash.show()

        # Close SplashScreen after 3 seconds (3000 ms)
        QTimer.singleShot(3000, self.splash.close)

        self.index = 0

        # Define our widgets
        self.upload_button = self.findChild(QPushButton, "upload_button")
        self.image_label = self.findChild(QLabel, "image_label")  
        self.next_image_button = self.findChild(QPushButton, "next_image_button")

        self.image_eye_color_label = self.findChild(QLabel, "image_eye_color_label")
        self.image_eye_rect_label = self.findChild(QLabel, "image_eye_rect_label")
        self.image_hair_color_label = self.findChild(QLabel, "image_hair_color_label")
        self.image_hair_rect_label = self.findChild(QLabel, "image_hair_rect_label") 


        self.dataset_eye_color_label = self.findChild(QLabel, "dataset_eye_color_label")
        self.dataset_eye_rect_label = self.findChild(QLabel, "dataset_eye_rect_label")
        self.dataset_hair_color_label = self.findChild(QLabel, "dataset_hair_color_label")
        self.dataset_hair_rect_label = self.findChild(QLabel, "dataset_hair_rect_label") 

        self.dataset_stats_label = self.findChild(QLabel, "dataset_stats_label")

        self.image_stats_label = self.findChild(QLabel, "image_stats_label")
        
        self.pixmap = QPixmap('assets/placeholder.png')

        # Add placeholder Pic to label
        self.image_label.setPixmap(self.pixmap)

        self.upload_button.clicked.connect(self.uploadImages)
        self.next_image_button.clicked.connect(self.nextImage) 

        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

        self.MplWidget.hide()
        self.next_image_button.hide()

        self.image_eye_color_label.hide()
        self.image_hair_color_label.hide()
        self.dataset_eye_color_label.hide()
        self.dataset_hair_color_label.hide()

        # Show The App
        self.show()


    def displayDatasetStats(self):

        text =  ''
        for name, value in average_results_dic.items():

            if name.split()[-1] == 'Color':
                color = QColor(value[0], value[1], value[2]).name()

                if name == 'Average Hair Color':
                    self.dataset_eye_rect_label.setStyleSheet('QLabel { background-color: %s }' %color)

                elif name == 'Average Eye Color':
                    self.dataset_hair_rect_label.setStyleSheet('QLabel { background-color: %s }' %color)
            else:
                text += f'{name}: {value}\n\n'

        self.dataset_stats_label.setText(text)

        self.MplWidget.canvas.axes.clear()

        x_axis = range(len(images))  

        self.MplWidget.canvas.axes.set_facecolor('black')
        self.MplWidget.canvas.axes.plot(x_axis, age_list, label='Age')
        
        self.MplWidget.canvas.axes.scatter(male, [age_list[i] for i in male], color='blue', label='Male')
        self.MplWidget.canvas.axes.scatter(female, [age_list[i] for i in female], color='pink', label='Female')

        self.MplWidget.canvas.axes.set_xlabel("Images Indexes", fontsize=14, color='white')
        self.MplWidget.canvas.axes.set_ylabel("Age of the person in image", fontsize=14, color='white')

        self.MplWidget.canvas.axes.spines['bottom'].set_color('white')
        self.MplWidget.canvas.axes.spines['top'].set_color('white')
        self.MplWidget.canvas.axes.spines['left'].set_color('white')
        self.MplWidget.canvas.axes.spines['right'].set_color('white')

        self.MplWidget.canvas.axes.tick_params(axis='x', colors='white')
        self.MplWidget.canvas.axes.tick_params(axis='y', colors='white')

        # self.MplWidget.canvas.axes.xaxis.label.set_color('white')
        # self.MplWidget.canvas.axes.yaxis.label.set_color('white')
        
        self.MplWidget.canvas.axes.legend() 
        self.MplWidget.canvas.draw()
        self.MplWidget.show()

        self.dataset_eye_color_label.show()
        self.dataset_hair_color_label.show()

    def displayImageStats(self):

        # Display a Image
        self.pixmap = QPixmap(images[self.index])
        # Add the image to label
        self.image_label.setPixmap(self.pixmap)

        hair_color = QColor(hair_color_list[self.index][2],
        hair_color_list[self.index][1],
        hair_color_list[self.index][0]).name()

        self.image_hair_rect_label.setStyleSheet('QLabel { background-color: %s }' %hair_color)

        eye_color = QColor(eye_color_list[self.index][2],
        eye_color_list[self.index][1],
        eye_color_list[self.index][0]).name()

        self.image_eye_rect_label.setStyleSheet('QLabel { background-color: %s }' %eye_color)

        self.image_eye_color_label.show()
        self.image_hair_color_label.show()

        self.image_stats_label.setText(f'Gender: {gender_list[self.index]}\n\nAge: {age_list[self.index]}\n\nRace: {race_list[self.index]}')


    def uploadImages(self):

        selected_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')

        if selected_dir:

            global images, hair_color_list, eye_color_list, gender_list, age_list, race_list, male, female, average_results_dic
            images = []
            self.index = 0

            for images_names in os.listdir(selected_dir):
                extension = images_names.split('.')[-1]
                if extension == 'jpg' or extension == 'png':
                    images.append(os.path.join(selected_dir, images_names))

            eye_color_list = []
            hair_color_list = []
            gender_list = []
            age_list = []
            race_list = []

            average_results_dic = {'Male Count': 0, 'Female Count': 0}

            male = []
            female = []

            for image_index, image_path in enumerate(images):
                
                image = cv2.imread(image_path)

                _, age_gender_results = predictAgeGender(image, age_gender_model, draw=False, display=False)
                
                gender, age_range =  age_gender_results[0]

                average_results_dic[gender + ' Count'] += 1

                if gender == 'Female':
                    female.append(image_index)

                elif gender == 'Male':
                    male.append(image_index)

                age = (age_range[0]+age_range[1])/2

                age_list.append(age)
                gender_list.append(gender)

                _, (race, _) = predictRace(image_path, race_model, draw=False, display=False)

                race_list.append(race.upper())

                if race.upper() + ' Race Count' in average_results_dic:
                    average_results_dic[race.upper() + ' Race Count'] += 1
                else:
                    average_results_dic[race.upper() + ' Race Count'] = 1

                
                _, hair_color = getHairColor(image, hair_model, threshold=0.8, debugging=False, draw=False, display=False)
                _, eye_color = getEyesColor(image, face_mesh, debugging=False, draw=False, display=False)

                eye_color_list.append(eye_color)
                hair_color_list.append(hair_color)

                if 'Average Hair Color' in average_results_dic:
                    average_results_dic['Average Hair Color'] = [
                        (average_results_dic['Average Hair Color'][0] + hair_color[0])/2,
                        (average_results_dic['Average Hair Color'][1] + hair_color[1])/2,
                        (average_results_dic['Average Hair Color'][2] + hair_color[2])/2,
                    ]

                else:
                    average_results_dic['Average Hair Color'] = hair_color

                if 'Average Eye Color' in average_results_dic:
                    average_results_dic['Average Eye Color'] = [
                        (average_results_dic['Average Eye Color'][0] + eye_color[0])/2,
                        (average_results_dic['Average Eye Color'][1] + eye_color[1])/2,
                        (average_results_dic['Average Eye Color'][2] + eye_color[2])/2,
                    ]

                else:
                    average_results_dic['Average Eye Color'] = eye_color

            average_results_dic['Average Age'] = round(sum(age_list)/len(age_list), 1)

            self.displayDatasetStats()

            self.displayImageStats()
            self.next_image_button.show()
    
    def nextImage(self):

        self.index += 1
        self.index %= len(images)

        self.displayImageStats()


# Initialize The App
app = QApplication(sys.argv)

UIWindow = App()
app.setStyleSheet(qdarkstyle.load_stylesheet_pyside2())

app.exec_()
