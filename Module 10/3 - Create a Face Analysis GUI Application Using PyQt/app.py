# Import the required libraries.
import os
import cv2
import sys
import qdarkstyle
from PyQt5 import uic
import mediapipe as mp
import tensorflow as tf
from deepface import DeepFace
from PyQt5.QtGui import QPixmap, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from previous_lesson import predictAgeGender, predictRace, getHairColor, getEyesColor
from PyQt5.QtWidgets import QLabel, QPushButton, QApplication, QMainWindow, QFileDialog

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

# Initialize a list to store the path of the images that are inside the selected folder.
images = []

# Initialize lists to store the indexes of the male and female images.
male = []
female = []

# Initialize lists to store the face analysis results for the images.
eye_color_list = []
hair_color_list = []
gender_list = []
age_list = []
race_list = []

# Initialize a dictionary to store the average of the face analysis results of the images. 
average_results_dic = {}

# Create a class.
class App(QMainWindow):

    # Create a constructor that will be called when an object is created from the class
    def __init__(self):
        
        # Call the constructor of the parent class.
        super(App, self).__init__()

        # Load the ui file.
        uic.loadUi("app.ui", self)

        # Set the window title.
        self.setWindowTitle("AI Executable GUI App")
        
        # Initialize a variable to store the index of the image to display.
        self.index = 0

        # Get the widgets created in the UI.
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
        
        # Load the paceholder image.
        self.pixmap = QPixmap('assets/placeholder.png')

        # Set the label to the placeholder image.
        # Required to display the image over the label.
        self.image_label.setPixmap(self.pixmap)
        
        # Connect the buttons to the functions that will be called whenever the buttons are presssed.
        self.upload_button.clicked.connect(self.uploadImages)
        self.next_image_button.clicked.connect(self.nextImage) 

        # Add the matplotlib toolbar.
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

        # Hide the toolbar.
        self.MplWidget.hide()
        
        # Hide the next image button.
        self.next_image_button.hide()

        # Hide the face analysis results labels.
        self.image_eye_color_label.hide()
        self.image_hair_color_label.hide()
        self.dataset_eye_color_label.hide()
        self.dataset_hair_color_label.hide()

        # Show the Application.
        self.show()

    # Create a function to display the whole dataset face analysis results.
    def displayDatasetStats(self):

        # Initialize a variable to store the average results of the images.
        average_results =  ''
        
        # Iterate over the items in the average results dictionary.
        for name, value in average_results_dic.items():
            
            # Check if the item we are iterating upon, is the average hair color or avaerage eye color.
            if name.split()[-1] == 'Color':
                
                # Get the color name.
                color = QColor(value[2], value[1], value[0]).name()

                # Check if the item we are iterating upon, is the average hair color.
                if name == 'Average Hair Color':
                    
                    # Set the dataset_hair_rect_label's background color equal to the hair color.
                    self.dataset_hair_rect_label.setStyleSheet('QLabel { background-color: %s }' %color)

                # Check if the item we are iterating upon, is the avaerage eye color.
                elif name == 'Average Eye Color':
                    
                    # Set the dataset_eye_rect_label's background color equal to the eye color.
                    self.dataset_eye_rect_label.setStyleSheet('QLabel { background-color: %s }' %color)
           
            # Otherwise.
            else:
                
                # Store the item key and value in the variable we had intialized.
                average_results += f'{name}: {value}\n\n'

        # Set the dataset_stats_label equal to the average_results.
        self.dataset_stats_label.setText(average_results)

        # Clear the canvas of the matplotlib figure.
        self.MplWidget.canvas.axes.clear()

        # Get the x-axis values.
        x_axis = range(len(images))  

        # Set the color of the graph.
        self.MplWidget.canvas.axes.set_facecolor('black')
        
        # Plot the age values as line.
        self.MplWidget.canvas.axes.plot(x_axis, age_list, label='Age')
        
        # Plot the age values as points with blue color for male and pink color for female.
        self.MplWidget.canvas.axes.scatter(male, [age_list[i] for i in male], color='blue', label='Male')
        self.MplWidget.canvas.axes.scatter(female, [age_list[i] for i in female], color='pink', label='Female')

        # Set the x and y axis label.
        self.MplWidget.canvas.axes.set_xlabel("Images Indexes", fontsize=14, color='white')
        self.MplWidget.canvas.axes.set_ylabel("Age of the person in image", fontsize=14, color='white')

        # Set the top, bottom, left, and right spine color.
        self.MplWidget.canvas.axes.spines['bottom'].set_color('white')
        self.MplWidget.canvas.axes.spines['top'].set_color('white')
        self.MplWidget.canvas.axes.spines['left'].set_color('white')
        self.MplWidget.canvas.axes.spines['right'].set_color('white')

        # Set the x and y axis color.
        self.MplWidget.canvas.axes.tick_params(axis='x', colors='white')
        self.MplWidget.canvas.axes.tick_params(axis='y', colors='white')
        
        # Place a legend on the Axes.
        self.MplWidget.canvas.axes.legend() 
        
        # Draw on the figure.
        self.MplWidget.canvas.draw()
        
        # Show the figure.
        self.MplWidget.show()

        # Show the eye and hair color label.
        self.dataset_eye_color_label.show()
        self.dataset_hair_color_label.show()

    # Create a function to display an image face analysis results.
    def displayImageStats(self):

        # Get the Image.
        self.pixmap = QPixmap(images[self.index])
        
        # Add the image to label.
        # Required to display the image over the label.
        self.image_label.setPixmap(self.pixmap)

        # Check if the hair color for the image was found.
        if hair_color_list[self.index] != None:
            
            # Get the hair color name.
            hair_color = QColor(hair_color_list[self.index][2],
            hair_color_list[self.index][1],
            hair_color_list[self.index][0]).name()

            # Set the image_hair_rect_label's background color equal to the hair color.
            self.image_hair_rect_label.setStyleSheet('QLabel { background-color: %s }' %hair_color)
            
            # Show the image_hair_rect_label (hair color:) label and the image_hair_color_label.
            self.image_hair_rect_label.show()
            self.image_hair_color_label.show()
        
        # Otherwise.    
        else:
            
            # Hide the labels.
            self.image_hair_rect_label.hide()
            self.dataset_hair_color_label.hide()

        # Check if the eyes color for the image was found.
        if eye_color_list[self.index] != None:
            
            # Get the eyes color name.
            eye_color = QColor(eye_color_list[self.index][2],
            eye_color_list[self.index][1],
            eye_color_list[self.index][0]).name()
            
            # Set the image_eye_rect_label's background color equal to the eyes color.
            self.image_eye_rect_label.setStyleSheet('QLabel { background-color: %s }' %eye_color)
            
            # Show the image_eye_rect_label (Eye color:) label and the image_hair_color_label.
            self.image_eye_rect_label.show()
            self.image_eye_color_label.show()
            
       # Otherwise.    
        else:
            
            # Hide the labels.
            self.image_eye_color_label.hide()
            self.image_eye_rect_label.hide()
        
        # Set the image_stats_label text.
        self.image_stats_label.setText(f'Gender: {gender_list[self.index]}\n\nAge: {age_list[self.index]}\n\nRace: {race_list[self.index]}')

    # Create a function to upload images.
    def uploadImages(self):

        # Get the path of the selected folder.
        selected_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')

        # Check if a folder was selected.
        if selected_dir:
            
            # Access the variables from outside of this function.
            global images, hair_color_list, eye_color_list, gender_list, age_list, race_list, male, female, average_results_dic
            
            # Re-intialize the variables.
            images = []
            eye_color_list = []
            hair_color_list = []
            gender_list = []
            age_list = []
            race_list = []
            average_results_dic = {'Male Count': 0, 'Female Count': 0}
            male = []
            female = []
            
            self.index = 0

            # Iterate over the images names.
            for images_names in os.listdir(selected_dir):
                
                # Get the extension of the file.
                extension = images_names.split('.')[-1]
                
                # Check if it is an image.
                if extension == 'jpg' or extension == 'png':
                    
                    # Append the image path into the list.
                    images.append(os.path.join(selected_dir, images_names))

            # Iterate over the images.
            for image_index, image_path in enumerate(images):
                
                # Read the image.
                image = cv2.imread(image_path)
                
                # Predict the age and gender.
                _, age_gender_results = predictAgeGender(image, age_gender_model, draw=False, display=False)
                
                # Check if the age and gender are predicted successfully.
                if len(age_gender_results) == 0:
                    
                    # Continue to the next iteration.
                    continue
                
                # Get the age and gender of the first face.
                # We are assuming that each image have only face.
                gender, age_range =  age_gender_results[0]

                # Increment the gender count.
                average_results_dic[gender + ' Count'] += 1

                # Check if the Female gender is predicted. 
                if gender == 'Female':
                    
                    # Append the image index into the list.
                    female.append(image_index)

                # Check if the Male gender is predicted. 
                elif gender == 'Male':
                    
                    # Append the image index into the list.
                    male.append(image_index)

                # Get the predicted age from the range.
                age = (age_range[0]+age_range[1])/2

                # Append the predicted age into the list.
                age_list.append(age)
                
                # Append the predicted gender into the list.
                gender_list.append(gender)

                # Predict the race of the person inside the image.
                _, (race, _) = predictRace(image_path, race_model, draw=False, display=False)

                # Append the predicted race into the list.
                race_list.append(race.upper())

                # Check if an item with the '{Predicted Race} Race Count' key is in the dictionary.
                if race.upper() + ' Race Count' in average_results_dic:
                    
                    # Increment the race counter.
                    average_results_dic[race.upper() + ' Race Count'] += 1
                
                # Otherwise.
                else:
                    
                    # Initialize a counter for the race.
                    average_results_dic[race.upper() + ' Race Count'] = 1

                # Get the hair and eyes color.
                _, hair_color = getHairColor(image, hair_model, threshold=0.8, debugging=False, draw=False, display=False)
                _, eye_color = getEyesColor(image, face_mesh, debugging=False, draw=False, display=False)

                # Append the hair and eyes color into the lists.
                eye_color_list.append(eye_color)
                hair_color_list.append(hair_color)
                
                # Check if the hair color is found.
                if hair_color != None:
                    
                    # Check if an item with the 'Average Hair Color' key is in the dictionary.     
                    if 'Average Hair Color' in average_results_dic:
                        
                        # Add the new values into the previous average values and divide by 2, to take average.
                        average_results_dic['Average Hair Color'] = [
                            (average_results_dic['Average Hair Color'][0] + hair_color[0])/2,
                            (average_results_dic['Average Hair Color'][1] + hair_color[1])/2,
                            (average_results_dic['Average Hair Color'][2] + hair_color[2])/2,
                        ]

                    # Otherwise.
                    else:
                        
                        # Initialize an item to store the avarage hair color.
                        average_results_dic['Average Hair Color'] = hair_color
                
                # Check if the eye color is found.       
                if eye_color != None:
                    
                    # Add the new values into the previous average values and divide by 2, to take average.
                    if 'Average Eye Color' in average_results_dic:
                        average_results_dic['Average Eye Color'] = [
                            (average_results_dic['Average Eye Color'][0] + eye_color[0])/2,
                            (average_results_dic['Average Eye Color'][1] + eye_color[1])/2,
                            (average_results_dic['Average Eye Color'][2] + eye_color[2])/2,
                        ]

                    # Otherwise.
                    else:
                        
                        # Initialize an item to store the avarage eye color.
                        average_results_dic['Average Eye Color'] = eye_color

            # Calculate the average age of the images.
            average_results_dic['Average Age'] = round(sum(age_list)/len(age_list), 1)

            # Display the whole dataset stats.
            self.displayDatasetStats()

            # Display the selected image stats.
            self.displayImageStats()
            
            # Show the next image button.
            self.next_image_button.show()
    
    # Create a function to switch images that are being displayed.
    def nextImage(self):
        
        # Increment the image index.
        self.index += 1
        self.index %= len(images)

        # Display the image stats.
        self.displayImageStats()


# Initialize the Application.
app = QApplication(sys.argv)
UIWindow = App()
app.setStyleSheet(qdarkstyle.load_stylesheet_pyside2())

# Run the Application.
app.exec_()
