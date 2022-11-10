import cv2
import math
import pymunk
import itertools
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe face mesh class.
mp_face_mesh = mp.solutions.face_mesh

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

# Initialize the mediapipe drawing styles class.
mp_drawing_styles = mp.solutions.drawing_styles


# Initialize a list to store the indexes of the upper lips outer outline landmarks.
lips_upper_outer_ids = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

# Initialize a list to store the indexes of the lower lips outer outline landmarks.
lips_lower_outer_ids = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Initialize a list to store the indexes of the upper lips inner outline landmarks.
lips_upper_inner_ids = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

# Initialize a list to store the indexes of the lower lips inner outline landmarks.
lips_lower_inner_ids = [324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

# Initialize a list to store the indexes of the upper part of the left eye outline landmarks.
left_eye_upper_ids = [246, 161, 160, 159, 158, 157, 173]

# Initialize a list to store the indexes of the lower part of the left eye outline landmarks.
left_eye_lower_ids = [133, 155, 154, 153, 145, 144, 163, 7, 33]

# Initialize a list to store the indexes of the upper part of the right eye outline landmarks.
right_eye_upper_ids=[466, 388, 387, 386, 385, 384, 398]

# Initialize a list to store the indexes of the lower part of the right eye outline landmarks.
right_eye_lower_ids=[362, 382, 381, 380, 374, 373, 390, 249, 263]

# Initialize a list to store the indexes of the right eye outline landmarks (for highlighting eye during selection).
# These landmarks cover wider area than the indexes we are using for eyeliner,
# And we are using these for making it easier for user to select the eyes.
selector_eye_ids = [342, 445, 444, 443, 442, 441, 413, 463, 341, 256, 252, 253, 254, 339, 255, 359]

# Initialize a list to store the indexes of the right cheek outline landmarks.
right_cheek_landmarks_ids = [379, 365, 397, 288, 361, 323, 454, 356, 372, 346, 280, 425, 432, 430]


def detectFacialLandmarks(image, face_mesh, draw=True, display = True):
    '''
    This function performs facial landmarks detection on an image.
    Args:
        image:     The input image of person(s) whose facial landmarks needs to be detected.
        face_mesh: The Mediapipe's face landmarks detection function required to perform the landmarks detection.
        draw:      A boolean value that is if set to true the function draws Face(s) landmarks on the output image. 
        display:   A boolean value that is if set to true the function displays the original input image, 
                   and the output image with the face landmarks drawn and returns nothing.
    Returns:
        output_image:   A copy of input image with face landmarks drawn.
        face_landmarks: An array containing the face landmarks (x and y coordinates) of a face in the image.
    '''
    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize an array to store the face landmarks.
    face_landmarks = np.array([])
    
    # Create a copy of the input image to draw facial landmarks.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the facial landmarks detection on the image.
    results = face_mesh.process(imgRGB)
    
    # Check if facial landmarks are found. 
    if results.multi_face_landmarks:

        # Iterate over the found faces.
        for face in results.multi_face_landmarks:
            
            # Convert the Face landmarks x and y coordinates into their original scale,
            # And store them into a numpy array.
            # For simplicity, we are only storing face landmarks of a single face, 
            # you can extend it to work with multiple faces if you want.
            face_landmarks = np.array([(landmark.x*width, landmark.y*height)
                                        for landmark in face.landmark], dtype=np.int32)
            
            # Check if facial landmarks are specified to be drawn.
            if draw:

                # Draw the facial landmarks on the output image with the face mesh tesselation
                # connections using default face mesh tesselation style.
                mp_drawing.draw_landmarks(image=output_image, landmark_list=face,
                                          connections=mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None, 
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Draw the facial landmarks on the output image with the face mesh contours
                # connections using default face mesh contours style.
                mp_drawing.draw_landmarks(image=output_image, landmark_list=face,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None, 
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    # Check if the original input image and the output image are specified to be displayed.
    if display:
        
        # Display the original input image and the output image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Sample Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    # Otherwise
    else:
        
        # Return the output image and landmarks returned by the detector.
        return output_image, face_landmarks               


def detectHandsLandmarks(image, hands, draw=True, display=True):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Mediapipe's Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, and the output 
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the hand landmarks on the output image.
            mp_drawing.draw_landmarks(image=output_image,
                                      landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.
                                      DrawingSpec(color=(255, 255, 255),
                                                  thickness=6, circle_radius=6),
                                      connection_drawing_spec=mp_drawing.
                                      DrawingSpec(color=(0, 255, 0),
                                                  thickness=4, circle_radius=4))

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Sample Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

        # Iterate over the found hands.
        for hand_world_landmarks in results.multi_hand_world_landmarks:

            # Plot the hand landmarks in 3D.
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results


def countFingers(image, results, consider_thumbs=True, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:           The image of the hands on which the fingers counting is required to be performed.
        results:         The output of the hands landmarks detection performed on the image.
        consider_thumbs: A boolean value that is if set to false the function doesnot consider 
                         hands thumbs in the count.
        draw:            A boolean value that is if set to true the function writes the total count of 
                         fingers up, of the hands on the image.
        display:         A boolean value that is if set to true the function displays the resultant image
                         and returns nothing.
    Returns:
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        fingers_statuses: A dictionary containing the status (i.e., up or down) of each finger of both hands.
        tips_landmarks:   A dictionary containing the landmarks of the tips of the fingers of both hands.
    '''

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP]

    # Initialize a dictionary to store the status
    # (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False,
                        'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False,
                        'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}

    # Initialize a dictionary to store the tips landmarks of each finger of the hands.
    tips_landmarks = {'RIGHT': {'THUMB': (None, None), 'INDEX': (None, None),
                                'MIDDLE': (None, None), 'RING': (None, None),
                                'PINKY': (None, None)},
                      'LEFT': {'THUMB': (None, None), 'INDEX': (None, None),
                               'MIDDLE': (None, None), 'RING': (None, None),
                               'PINKY': (None, None)}}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand i.e. left or right.
        hand_label = hand_info.classification[0].label

        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:

            # Retrieve the label (i.e., index, middle, etc.) of the
            # finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]

            # Store the tip landmark of the finger in the dictionary.
            tips_landmarks[hand_label.upper()][finger_name] = \
                (int(hand_landmarks.landmark[tip_index].x*width),
                 int(hand_landmarks.landmark[tip_index].y*height))

            # Check if the finger is up by comparing the y-coordinates
            # of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y <
                    hand_landmarks.landmark[tip_index - 2].y):

                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

        # Store the tip landmark of the thumb in the dictionary.
        tips_landmarks[hand_label.upper()]['THUMB'] = \
            (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*width),
             int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*height))
        
        # Check if thumbs are specified to be consider in the count.
        if consider_thumbs:
        
            # Retrieve the x-coordinates of the tip and mcp landmarks of the thumb of the hand.
            thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

            # Check if the thumb is up by comparing the hand label and
            # the x-coordinates of the retrieved landmarks.
            if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or \
            (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):

                # Update the status of the thumb in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_THUMB"] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

    # Check if the total count of the fingers of both hands are specified to be written on the image.
    if draw:

        # Write the total count of the fingers of both hands on the image.
        cv2.putText(image, " Total Fingers: ", (10, 55),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (20, 255, 155), 3)
        cv2.putText(image, str(sum(count.values())), (width//2-150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20, 255, 155), 10, 10)

    # Check if the image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise.
    else:

        # Return the count of fingers up, each finger status, and tips landmarks.
        return count, fingers_statuses, tips_landmarks


def recognizeGestures(image, results, hand_label='LEFT', draw=True, display=True):
    '''
    This function will determine the gesture a hand in the image.
    Args:
        image:      The image of the hands on which the hand gesture recognition is required to be performed.
        results:    The output of the hands landmarks detection performed on the image.
        hand_label: The label of the hand i.e. left or right, of which the gesture is to be recognized.      
        draw:       A boolean value that is if set to true the function writes the gesture of the hand on the
                    image, after recognition.
        display:    A boolean value that is if set to true the function displays the resultant image and 
                    returns nothing.
    Returns:
        hands_gestures:        The recognized gesture of the specified hand in the image.
        fingers_tips_position: The fingers tips landmarks coordinates of the other hand in the image.
    '''

    # Initialize a variable to store the gesture of the hand in the image.
    hand_gesture = 'UNKNOWN'

    # Initialize a variable to store the color we will use to write the hand gesture on the image.
    # Initially it is red which represents that the gesture is not recognized.
    color = (0, 0, 255)

    # Get the count of fingers up, fingers statuses, and tips landmarks of the detected hand(s).
    count, fingers_statuses, fingers_tips_position = countFingers(image, results, draw=False,
                                                                  display=False)

    # Check if the number of the fingers up of the hand is 1 and the finger that is up,
    # is the index finger.
    if count[hand_label] == 1 and fingers_statuses[hand_label+'_INDEX']:

        # Set the gesture recognized of the hand to 'INDEX POINTING UP' SIGN.
        hand_gesture = 'INDEX POINTING UP'

        # Update the color value to green.
        color = (0, 255, 0)
        
    # Check if the number of the fingers up of the hand is 1 and the finger that is up,
    # is the pinky finger.
    elif count[hand_label] == 1 and fingers_statuses[hand_label+'_PINKY']:

        # Set the gesture recognized of the hand to 'PINKY POINTING UP' SIGN.
        hand_gesture = 'PINKY POINTING UP'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 2 and the fingers that are up,
    # are the index finger and the middle finger.
    elif count[hand_label] == 2 and fingers_statuses[hand_label+'_INDEX'] and \
            fingers_statuses[hand_label+'_MIDDLE']:

        # Set the gesture recognized of the hand to 'VICTORY' SIGN.
        hand_gesture = 'VICTORY'

        # Update the color value to green.
        color = (0, 255, 0)
        
    # Check if the number of fingers up of the hand is 2 and the fingers that are up,
    # are the index finger and the thumb.
    elif count[hand_label] == 2 and fingers_statuses[hand_label+'_INDEX'] and \
            fingers_statuses[hand_label+'_THUMB']:

        # Set the gesture recognized of the hand to 'LOSER' SIGN.
        hand_gesture = 'LOSER'

        # Update the color value to green.
        color = (0, 255, 0)
        
    # Check if the number of fingers up of the hand is 3 and the fingers that are up,
    # are the middle finger, ring finger, and pinky finger. 
    elif count[hand_label] == 3 and fingers_statuses[hand_label+'_MIDDLE'] and \
            fingers_statuses[hand_label+'_RING'] and fingers_statuses[hand_label+'_PINKY']:

        # Set the gesture recognized of the hand to 'MIDDLE RING PINKY POINTING UP' SIGN.
        hand_gesture = 'MIDDLE RING PINKY POINTING UP'

        # Update the color value to green.
        color = (0, 255, 0)
        
    # Check if the number of fingers up of the hand is 3 and the fingers that are up,
    # are the index finger, middle finger, and the thumb.
    elif count[hand_label] == 3 and fingers_statuses[hand_label+'_INDEX'] and \
            fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_THUMB']:

        # Set the gesture recognized of the hand to 'INDEX MIDDLE THUMB POINTING UP' SIGN.
        hand_gesture = 'INDEX MIDDLE THUMB POINTING UP'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 3 and the fingers that are up,
    # are the index finger, pinky finger, and the thumb.
    elif count[hand_label] == 3 and fingers_statuses[hand_label+'_INDEX'] and \
            fingers_statuses[hand_label+'_PINKY'] and fingers_statuses[hand_label+'_THUMB']:

        # Set the gesture recognized of the hand to 'SPIDERMAN' SIGN.
        hand_gesture = 'SPIDERMAN'

        # Update the color value to green.
        color = (0, 255, 0)
        
    # Check if the number of fingers up of the hand is 4 and the thumb is closed.
    elif count[hand_label] == 4 and not(fingers_statuses[hand_label+'_THUMB']):

        # Set the gesture recognized of the hand to 'ALL FINGERS POINTING UP' SIGN.
        hand_gesture = 'ALL FINGERS POINTING UP'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 5.
    elif count[hand_label] == 5:

        # Set the gesture recognized of the hand to 'HIGH-FIVE' SIGN.
        hand_gesture = 'HIGH-FIVE'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the recognized hand gesture is specified to be written.
    if draw:

        # Write the recognized hand gesture on the image.
        cv2.putText(image, hand_label + ' HAND: ' + hand_gesture, (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 4, color, 5)

    # Check if the image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the hand gesture name and the fingers tips position of the both hands.
        return hand_gesture, fingers_tips_position


def calculateDistance(image, point1, point2, draw=True, display=True):
    '''
    This function will calculate distance between two points on an image.
    Args:
        image:   The image on which the two points are.
        point1:  A point with x and y coordinates values on the image.
        point2:  Another point with x and y coordinates values on the image.
        draw:    A boolean value that is if set to true the function draws a line between the 
                 points and write the calculated distance on the image
        display: A boolean value that is if set to true the function displays the output image 
                 and returns nothing.
    Returns:
        distance: The calculated distance between the two points.

    '''

    # Initialize the value of the distance variable.
    distance = None

    # Get the x and y coordinates of the points.
    x1, y1 = point1
    x2, y2 = point2

    # Check if all the coordinates values are processable.
    if isinstance(x1, int) and isinstance(y1, int) \
            and isinstance(x2, int) and isinstance(y2, int):

        # Calculate the distance between the two points.
        distance = math.hypot(x2 - x1, y2 - y1)

        # Check if the distance is greater than the upper threshold.
        if distance > 230:

            # Set the distance to the upper threshold.
            distance = 230

        # Check if the distance is lesser than the lower threshold.
        elif distance < 30:

            # Set the distance to the lower threshold.
            distance = 30

        if draw:

            # Draw a line between the two points on the image.
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                     (255, 0, 255), 4)

            # Draw a circle on the first point on the image.
            cv2.circle(image, (int(x1), int(y1)), 20, (0, 255, 0), -1)

            # Draw a circle on the second point on the image.
            cv2.circle(image, (int(x2), int(y2)), 20, (0, 255, 0), -1)

            # Write the calculated distance between the two points on the image.
#             cv2.putText(image, f'Distance: {round(distance, 2)}', (10, 30),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Check if the image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Return the calculated distance.
    return distance

def selectFacePart(image, face_landmarks, hands_results, hand_label='RIGHT'):
    '''
    This function will allow the user to select face parts utilizing hand gestures.
    Args:
        image:          The image/frame of the user with his index finger pointing towards a face part to select.
        face_landmarks: An array containing the face landmarks (x and y coordinates) of the face in the image.
        hands_results:  The output of the hands landmarks detection performed on the image. 
        hand_label:     The label of the hand i.e. left or right, of which the gesture is required to be recognized. 
    Returns:
        output_image:       A copy of the input image with transparent contours drawn, highlighting the selectable face parts. 
        selected_face_part: The name of the face part selected by the user in the image/frame.
    '''
    
    # Initialize a variable to store the selected face part.
    selected_face_part = None
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Initialize a list to store the lips landmarks.
    lips_landmarks = []
    
    # Initialize a list to store the right eye landmarks.
    right_eye_landmarks = []
    
    # Initialize a list to store the right cheek landmarks.
    right_cheek_landmarks = []
    
    # Get the height and width of the image.
    height, width, _ = image.shape

    # Get the count of fingers up, fingers statuses, and tips landmarks of the detected hand(s).
    # I have modified this countFingers() function from previous module to ignore the thumbs count logic, if consider_thumbs
    # is False, to remove the limatation of always having to face the palm of hand towards the camera to get correct results.
    count, fingers_statuses, fingers_tips_position = countFingers(image, hands_results, consider_thumbs=False,
                                                                  draw=False, display=False)
    
    # Check if the number of the fingers up of the selector hand is 1 and the finger that is up, is the index finger.
    # And the number of the fingers up, of the opposite hand is 0.
    if count[hand_label] == 1 and fingers_statuses[hand_label+'_INDEX'] \
    and count['LEFT' if hand_label=='RIGHT' else 'RIGHT'] == 0:
        
        # Get the x and y coordinates of the tip landmark of the index finger of the selector hand. 
        index_x, index_y = fingers_tips_position[hand_label]['INDEX']
        
        # Lips Selection part.
        ####################################################################################################################
        
        # Iterate over the indexes of the upper and lower lips outline.
        for index in lips_upper_outer_ids+lips_lower_outer_ids:
            
            # Get the landmark at the index we are iterating upon,
            # And append it into the list.
            lips_landmarks.append(face_landmarks[index])
        
        # Convert the lips landmarks list into a numpy array.
        lips_landmarks = np.array(lips_landmarks, np.int32)
        
        # Draw filled lips contours on the copy of the image.
        cv2.drawContours(output_image, contours=[lips_landmarks], contourIdx=-1, 
                         color=(255, 255, 255), thickness=-1)
        
        # Check if the index finger tip is inside the lips contours (outline). 
        if cv2.pointPolygonTest(lips_landmarks,(index_x, index_y), measureDist=False)  == 1:
            
            # Update the selected face part variable to LIPS.
            selected_face_part = 'LIPS'
            
        # Eyes Selection part.
        ####################################################################################################################
        
        # Iterate over the indexes of the right eye ouline.
        for index in selector_eye_ids:
            
            # Get the landmark at the index we are iterating upon,
            # And append it into the list.
            right_eye_landmarks.append(face_landmarks[index])
        
        # Convert the right eye landmarks list into a numpy array.
        right_eye_landmarks = np.array(right_eye_landmarks, np.int32)
        
        # Draw filled right eye contours on the copy of the image.
        cv2.drawContours(output_image, contours=[right_eye_landmarks], contourIdx=-1, 
                         color=(255, 255, 255), thickness=-1)  
        
        # Check if the index finger tip is inside the right eye contours (outline). 
        if cv2.pointPolygonTest(right_eye_landmarks,(index_x, index_y), measureDist=False)  == 1:
            
            # Update the selected face part variable to EYES.
            selected_face_part = 'EYES'
        
        # Face Selection part.
        ####################################################################################################################
        
        # Iterate over the indexes of the right cheek ouline.
        for index in right_cheek_landmarks_ids:
            
            # Get the landmark at the index we are iterating upon,
            # And append it into the list.
            right_cheek_landmarks.append(face_landmarks[index])
        
        # Convert the right cheek landmarks list into a numpy array.
        right_cheek_landmarks = np.array(right_cheek_landmarks, np.int32)
        
        # Draw filled right cheek contours on the copy of the image.
        cv2.drawContours(output_image, contours=[right_cheek_landmarks], contourIdx=-1, 
                         color=(255, 255, 255), thickness=-1)    
        
        # Check if the index finger tip is inside the right cheek contours (outline). 
        if cv2.pointPolygonTest(right_cheek_landmarks,(index_x, index_y), measureDist=False)  == 1:
            
            # Update the selected face part variable to FACE.
            selected_face_part = 'FACE'
        
        ####################################################################################################################
        
        # Perform weighted addition between the original image and 
        # its copy with the contours drawn to get a transparency effect. 
        output_image = cv2.addWeighted(output_image, 0.15, image, 0.85, 0)
    
    # Return the image with transparent contours drawn, and the selected face part
    return output_image, selected_face_part

def getFacePartMask(image, face_landmarks, face_part='FACE',  display=True):
    '''
    This function will generate a face part mask image utilizing face landmarks.
    Args:
        image:          The image of the face whose face part mask image is required.
        face_landmarks: An array containing the face landmarks (x and y coordinates) of the face in the image.
        face_part:      The face part name whose mask is to be generated.
        display:        A boolean value that is if set to true the function displays the face image, 
                        and the generated face part mask image and returns nothing.
    Returns:
        mask: The face part mask image with values 255 at the specified face part region and 0 at the remaining regions.
    '''
    
    # Get the height and width of the face image.
    height, width, _ = image.shape
    
    # Initialize a list to store the face outline landmarks.
    face_outline_landmarks = []
    
    # Initialize a list to store the lips landmarks.
    lips_landmarks = []
    
    # Initialize a list to store the mouth landmarks.
    mouth_landmarks= []
    
    # Initialize a list to store the left eye landmarks.
    left_eye_landmarks = []
    
    # Initialize a list to store the right eye landmarks.
    right_eye_landmarks = []
    
    # Find Convex hull of the face landmarks.
    # Convex hull is the smallest convex set of points (boundary) that contains all the points in it.
    # This will return the indexes of the face outline landmarks.
    face_outline_indexes = cv2.convexHull(face_landmarks, returnPoints=False)

    # Iterate over the indexes of the face outline.   
    for index in face_outline_indexes:

        # Get the landmark at the index we are iterating upon,
        # And append it into the list.
        face_outline_landmarks.append(face_landmarks[index[0]])
    
    # Iterate over the indexes of the upper and lower lips outer outline.
    for index in lips_upper_outer_ids+lips_lower_outer_ids:
        
        # Get the landmark at the index we are iterating upon,
        # And append it into the list.
        lips_landmarks.append(face_landmarks[index])
    
    # Iterate over the indexes of the upper and lower lips innner outline.   
    for index in lips_upper_inner_ids+lips_lower_inner_ids:
        
        # Get the landmark at the index we are iterating upon,
        # And append it into the list.
        mouth_landmarks.append(face_landmarks[index])
    
    # Iterate over the indexes of the left eye outline.
    for index in left_eye_upper_ids+left_eye_lower_ids:
        
        # Get the landmark at the index we are iterating upon,
        # And append it into the list.
        left_eye_landmarks.append(face_landmarks[index])
        
    # Iterate over the indexes of the right eye outline.
    for index in right_eye_upper_ids+right_eye_lower_ids:
        
        # Get the landmark at the index we are iterating upon,
        # And append it into the list.
        right_eye_landmarks.append(face_landmarks[index])

    # Initialize a black empty canvas to draw the face part(s) on.
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Check if the specified face part whose mask is needed is the FACE.
    if face_part == 'FACE':
        
        # Draw (white) filled face contours on the mask (black canvas).
        cv2.drawContours(mask, contours=[np.int32(face_outline_landmarks)], contourIdx=-1, 
                             color=(255, 255, 255), thickness=-1)  

        # Remove right eye area from the mask by drawing filled right eye contours with black color.
        cv2.drawContours(mask, contours=[np.array(right_eye_landmarks, np.int32)], contourIdx=-1, 
                         color=(0, 0, 0), thickness=-1)
        
        # Remove left eye area from the mask by drawing filled left eye contours with black color.
        cv2.drawContours(mask, contours=[np.array(left_eye_landmarks, np.int32)], contourIdx=-1, 
                         color=(0, 0, 0), thickness=-1) 
        
        # Remove lips area from the mask by drawing filled lips contours with black color.
        cv2.drawContours(mask, contours=[np.array(lips_landmarks, np.int32)], contourIdx=-1, 
                         color=(0, 0, 0), thickness=-1)
    
    # Check if the specified face part whose mask is needed is the LIPS.
    elif face_part == 'LIPS':
        
        # Draw (white) filled lips contours on the mask (black canvas).
        cv2.drawContours(mask, contours=[np.int32(lips_landmarks)], contourIdx=-1, 
                         color=(255, 255, 255), thickness=-1)  
    
        # Remove mouth area from the mask by drawing filled mouth contours with black color.
        cv2.drawContours(mask, contours=[np.array(mouth_landmarks, np.int32)], contourIdx=-1, 
                         color=(0, 0, 0), thickness=-1)    
        
    # Check if the specified face part whose mask is needed is the EYES.
    elif face_part == 'EYES':
        
        # Draw (white) hollow right eye contours on the mask (black canvas) with thickness 2.
        cv2.drawContours(mask, contours=[np.int32(right_eye_landmarks)], contourIdx=-1, 
                         color=(255, 255, 255), thickness=2) 
        
        # Draw (white) hollow left eye contours on the mask (black canvas) with thickness 2.
        cv2.drawContours(mask, contours=[np.int32(left_eye_landmarks)], contourIdx=-1, 
                     color=(255, 255, 255), thickness=2)
        
    # Check if the original input image and the generated mask image are specified to be displayed.
    if display:
        
        # Display the original input image and the generated mask image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Sample Image");plt.axis('off');
        plt.subplot(122);plt.imshow(mask);plt.title("Output Image");plt.axis('off');
        
    # Otherwise
    else: 
    
        # Return the face part mask image with values 255 at the specified face part region.
        return mask
    

def applyMakeup(image, face_part_mask, hsv_value, display=True):
    '''
    This function will apply virtual makeup (like foundation, lipstick, and eyeliner) on a face.
    Args:
        image:          The image to apply virtual face makeup on.
        face_part_mask: The mask image of the face part on which to apply virtual makeup on. 
        hsv_value:      A list containing hue, saturation, and value channel scale factors.
        display:        A boolean value that is if set to true the function displays the original input image, 
                        and the output image with the virtual face makeup applied and returns nothing.
    Returns: 
        output_image: A copy of the input image with the virtual face makeup applied. 
    '''
    
    # Define a helper nested function to modify hue, saturation, and value channels.
    # We are defining it inside this function scope because we need this changeHSV() function only here.
    def changeHSV(input_image, hsv_scale_factor):
        '''
        This function will increase/decrease the Hue, Saturation, and Value (Brighness) channels of an image.
        Args:
            image:            The image whose Hue, Saturation, and Value channels are to be modified.
            hsv_scale_factor: A list containing hue, saturation, and value channels scale factors.
        Returns:
            output_image: A copy of the input image with the Hue, Saturation, and Value channels values modified.
        '''
        
        # Get the hue, saturation, and value channels scale factors.
        hue_scale, saturation_scale, value_scale = hsv_scale_factor

        # Conver the image from BGR into HSV format.
        image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        # Convert the pixel values of the image into float.
        image_hsv = np.array(image_hsv, dtype=np.float64)

        # Split the hue, saturation, and value channel of the image.
        hue_channel, saturation_channel, value_channel = cv2.split(image_hsv)
        
        # Scale up or down the pixel values of the hue channel accordingly to the scale factor.
        # This np.mod() function will keep the resultant values between [0-179].
        # As in opencv, hue channel only have values between [0-179].
        hue_channel = np.mod(hue_channel+hue_scale, 180)

        # Scale up or down the pixel values of the saturation channel accordingly the scale factor.
        saturation_channel += saturation_scale

        # Scale up or down the pixel values of the value channel accordingly the scale factor.
        value_channel += value_scale

        # Merge the Hue, Saturation, and Value channel.
        image_hsv = cv2.merge((hue_channel, saturation_channel, value_channel))

        # Set values > 255 to 255 and values < 0 to 0.
        image_hsv[image_hsv > 255] = 255
        image_hsv[image_hsv < 0] = 0

        # Convert the image into uint8 type and BGR format.
        output_image = cv2.cvtColor(np.array(image_hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

        # Return the output image with the Hue, Saturation, and Value channels values modified.
        return output_image
    
    # Create copies of the input image.
    output_image = image.copy()
    modified_image = image.copy()
    
    # Get the face part(s) contours from the face part mask image.
    # cv2.findContours() requires a single channel image, so we will only pass the first channel of the mask image.
    contours, _ = cv2.findContours(image=face_part_mask[:,:,0], mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    # Iterate over the found contours.
    for cnt in contours:
        
        # Get the bounding box (enclosing a face part) coordinates.
        x1, y1, width, height = cv2.boundingRect(cnt)
        
        # Calculate the bounding box x2 and y2 values utilizing the x1, y1, width, height values.
        x2, y2 = x1+width, y1+height
        
        # Crop the face part region from a copy of the image.
        image_roi = modified_image[y1:y2, x1:x2]
        
        # Modify the Hue, Saturation, and Value (Brighness) channels of the cropped image.
        modified_roi = changeHSV(image_roi, hsv_value)
        
        # Apply bilateral filter to smoothen the cropped image.
        modified_roi = cv2.bilateralFilter(src=modified_roi, d=5, sigmaColor=4, sigmaSpace=4)
        
        # Put back the cropped modified part into the copy of the image.
        modified_image[y1:y2, x1:x2] = modified_roi
    
    # Update the pixel values of the output image with the modified image values at the indexes,
    # where face_part_mask!=0 i.e. where mask is not black and face parts are drawn there. 
    output_image[np.mean(face_part_mask, axis=2)!=0] = modified_image[np.mean(face_part_mask, axis=2)!=0]
    
    # Check if the original input image and the output image are specified to be displayed.
    if display:
        
        # Display the original input image and the output image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Sample Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    # Otherwise
    else:
        
        # Return the output image with the virtual face makeup applied.
        return output_image
    
def draw(frame, canvas, current_gesture, hands_tips_positions, prev_coordinates, paint_color, brush_size=20, eraser_size=80):
    '''
    This function will draw, erase and clear a canvas based on different hand gestures.
    Args:
        frame:                A frame/image from the webcam feed.
        canvas:               A black image equal to the webcam feed size, to draw on.
        current_gesture:      The current gesture of the hand recognized using our gesture recognizer from a previous lesson.
        hands_tips_positions: A dictionary containing the landmarks of the tips of the fingers of a hand.
        prev_coordinates:     The hand brush x and y coordinates from the previous frame.
        paint_color:          The color to draw with, on the canvas.
        brush_size:           The size of the paint brush to draw with, on the canvas.
        eraser_size:          The size of the eraser to erase with, on the canvas.
    Returns:
        canvas: The black image with the intented drawings on it, in the paint color.
    '''
    
    # Get the hand brush previous x and y coordinates values (i.e. from the previous frame).
    prev_x, prev_y = prev_coordinates
    
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
     # Check if the current hand gesture is INDEX POINTING UP.
    if current_gesture == 'INDEX POINTING UP':

        # Write the current mode on the frame with the paint color.
        cv2.putText(img=frame, text='Paint Mode Enabled', org=(10, frame_height-30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=paint_color, thickness=2)

        # Get the x and y coordinates of tip of the index finger of the hand.
        x, y = hands_tips_positions['INDEX']

        # Check if x and y have valid values.
        # These will be none if the right hand was not detected in the frame.
        # This check will be necessary if you are checking gesture of a different hand and
        # want tips landmarks of the different one. But we are not doing that right now,
        # so if you want you can remove this check.
        if x and y:

            # Check if the previous x and y donot have valid values.
            if not(prev_x) and not(prev_y):

                # Set the previous x and y to the current x and y values.
                prev_x, prev_y = x, y

            # Draw a line on the canvas from previous x and y to the current x and y with the paint color 
            # and thickness equal to the brush_size.
            cv2.line(img=canvas, pt1=(prev_x, prev_y), pt2=(x, y), color=paint_color, thickness=brush_size)

            # Update the previous x and y to the current x and y values.
            prev_x, prev_y = x, y

        
    # Check if the current hand gesture is HIGH-FIVE.
    elif current_gesture == 'HIGH-FIVE':

        # Write the current mode on the frame.
        cv2.putText(img=frame, text='Erase Mode Enabled', org=(10, frame_height-30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                    color=paint_color, thickness=2)

        # Get the x and y coordinates of tip of the middle finger of the hand.
        x1, y = hands_tips_positions['MIDDLE'] 

        # Get the x coordinate of tip of the ring finger of the hand.
        x2, _ = hands_tips_positions['RING'] 

        # Check if the right hand was detected in the frame.
        if x1 and x2 and y:

            # Calculate the midpoint between tip x coordinate of the middle and ring finger
            x = (x1 + x2) // 2

            # Check if the previous x and y donot have valid values.
            if not(prev_x) and not(prev_y):

                # Set the previous x and y to the current x and y values.
                prev_x, prev_y = x, y

            # Draw a circle on the frame at the current x and y coordinates, equal to the eraser size.
            # This is drawn just to represent an eraser on the current x and y values.
            cv2.circle(img=frame, center=(x, y), radius=int(eraser_size/2), color=(255,255,255), thickness=-1)

            # Draw a black line on the canvas from previous x and y to the current x and y.
            # This will erase the paint between previous x and y and the current x and y.
            cv2.line(img=canvas, pt1=(prev_x, prev_y), pt2=(x, y), color=(0,0,0), thickness=eraser_size)

            # Update the previous x and y to the current x and y values.
            prev_x, prev_y = x, y
    
    # Check if the current hand gesture is SPIDERMAN.
    elif current_gesture == 'SPIDERMAN':

        # Write 'Clear Everything' on the frame.
        cv2.putText(img=frame, text='Clear Everything', org=(10, frame_height-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=paint_color, thickness=2)

        # Clear the canvas, by re-initializing it to a complete black image.
        canvas = np.zeros((frame_height, frame_width, 3), np.uint8)
    
    # Return the canvas along with the previous x and y coordinates.
    return canvas, (prev_x, prev_y)


def selectShape(frame, hand_tips_positions, DEFAULT_SHAPE_SIZE): 
    '''
    This function will select a shape utilizing a hand tips landmarks.
    Args:
        frame:               The current frame/image of a real-time webcam feed.
        hand_tips_positions: A dictionary containing the landmarks of the tips of the fingers of a hand.
        DEFAULT_SHAPE_SIZE:  The default size of each drawable shape.
    Returns:
        shape_selected: The shape to draw, selected by the user using middle finger tip.
    '''
    
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
    # Initialize a variable to store the selected shape.
    shape_selected = None
    
    # Read the shapes selection tab image and resize its width equal to the frame width.
    shapes_selector_tab = cv2.imread('media/overlays/shapes_selector.png')
    shapes_selector_height, shapes_selector_width, _ = shapes_selector_tab.shape
    shapes_selector_tab = cv2.resize(shapes_selector_tab, (frame_width, int((frame_width/shapes_selector_width)*shapes_selector_height)))
    
    # Overlay the shape selection tab image on the top of the frame.
    frame[0:shapes_selector_tab.shape[0], 0:shapes_selector_tab.shape[1]] = shapes_selector_tab
    
    # Get the x and y coordinates of tip of the MIDDLE finger of the hand. 
    x, y = hand_tips_positions['MIDDLE']
    
    # Check if the MIDDLE finger tip is over the shape selection tab image.
    if y <= shapes_selector_tab.shape[0]:
        
        # Check if the MIDDLE finger tip is over the Circle ROI.
        if x>(int(frame_width//11.5)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//11.5)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Circle'.
            shape_selected='Circle'
        
        # Check if the MIDDLE finger tip is over the Polygon ROI.
        elif x>(int(frame_width//4.35)-DEFAULT_SHAPE_SIZE/2) and \
        x<(int(frame_width//4.35)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Polygon'.
            shape_selected='Polygon'
        
        # Check if the MIDDLE finger tip is over the Rectangle ROI.
        elif x>(int(frame_width//2.37)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//2.37)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Rectangle'.
            shape_selected='Rectangle'
        
        # Check if the MIDDLE finger tip is over the Square ROI.
        elif x>(int(frame_width//1.64)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//1.64)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Square'.
            shape_selected='Square'
        
        # Check if the MIDDLE finger tip is over the Triangle ROI.
        elif x>(int(frame_width//1.095)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//1.095)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Triangle'.
            shape_selected='Triangle'
        
        # Check if the MIDDLE finger tip is over the Right Triangle ROI.
        elif x>(int(frame_width//1.292)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//1.292)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Right Triangle'.
            shape_selected='Right Triangle'
    
    # Return the selected shape.
    return shape_selected

def drawShapes(frame, canvas, shape_selected, hand_gesture, hand_tips_positions, paint_color, shape_size):
    '''
    This function will draw a selected shape on a frame or canvas based on hand gestures.
    Args:
        frame:               The current frame/image of a real-time webcam feed.
        canvas:              A black image equal to the webcam feed size, to draw on.
        shape_selected:      The shape to draw, selected by the user using middle finger tip.
        hand_gesture:        The current hand gesture recognized in the frame.
        hand_tips_positions: A dictionary containing the landmarks of the tips of the fingers of a hand.
        paint_color:         The color to draw shapes with, on the canvas.
        shape_size:          The size of which, the selected shape is to be draw.
    Returns:
        frame:          The frame with the selected shape drawn and current active mode written.
        canvas:         The black image with the selected shapes drawn on it, in the paint color.
        shape_selected: The name of the selected shape.
    '''
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
    # Get the x and y coordinates of tip of the MIDDLE finger of the hand. 
    x, y = hand_tips_positions['MIDDLE']
    
    # Check if the current hand gesture is 'VICTORY'. 
    if hand_gesture == 'VICTORY':
        
        # Write 'Shape Selection Mode Enabled' on the frame.
        cv2.putText(frame, 'Shape Selection Mode Enabled', (10, frame_height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, paint_color, 2)
        
        # Store the frame reference in the image variable.
        # Now if we draw on the image (variable), the drawing will be made on the frame.
        image = frame
            
    # Check if the current hand gesture is 'INDEX MIDDLE THUMB POINTING UP'. 
    elif hand_gesture == 'INDEX MIDDLE THUMB POINTING UP':
        
        # Write 'Shape Placement Mode Enabled' on the frame.
        cv2.putText(frame, 'Shape Placement Mode Enabled', (10, frame_height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, paint_color, 2)
        
        # Store the canvas reference in the image variable.
        # Now if we draw on the image (variable), the drawing will be made on the canvas.
        image = canvas
           
    # Check if the current hand gesture is 'SPIDERMAN'.
    elif hand_gesture == 'SPIDERMAN':

        # Write 'Clear Everything' on the frame.
        cv2.putText(img=frame, text='Clear Everything', org=(10, frame_height-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=paint_color, thickness=2)

        # Clear the canvas, by re-initializing it to a complete black image.
        canvas = np.zeros((frame_height, frame_width, 3), np.uint8)
    
    # Check if the current hand gesture is 'VICTORY' or 'INDEX MIDDLE THUMB POINTING UP'.
    if hand_gesture == 'VICTORY' or hand_gesture == 'INDEX MIDDLE THUMB POINTING UP':
        
        # Check if the selected shape is 'Circle'.
        if shape_selected == 'Circle':
            
            # Get the radius of the circle to draw.
            circle_radius = shape_size//2
            
            # Draw the circle on the image that will be either frame or canvas based on current gesture.
            image = cv2.circle(img=image, center=(x, y), radius=circle_radius,
                               color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Polygon'.
        elif shape_selected == 'Polygon':
            
            # Get the radius of the polygon to draw.
            polygon_radius = shape_size//2
            
            # Initialize a list to store the polygon contour (boundary) points.
            polygon_pts=[]
            
            # Iterate over 6 times. 
            # As we are drawing a polygon (hexagon) with 6 sides.
            for i in range(6):
                
                # Get the x and y coordinates of a polygon corner.
                poly_x = x + polygon_radius* math.cos(i * 2 * math.pi / 6)
                poly_y = y + polygon_radius * math.sin(i * 2 * math.pi / 6)
                
                # Append the x and y coordinates into the list.
                polygon_pts.append((poly_x, poly_y))
            
            # Draw the polygon on the image that will be either frame or canvas based on current gesture.
            image = cv2.fillPoly(image, pts = [np.array(polygon_pts, np.int32)],
                                 color=paint_color)
        
        # Check if the selected shape is 'Rectangle'.
        elif shape_selected == 'Rectangle':
            
            # Get the rectangle height and width.
            rec_width = shape_size*2
            rec_height = shape_size
            
            # Draw the rectangle on the image that will be either frame or canvas based on current gesture.
            image = cv2.rectangle(image, pt1=(x-rec_width//2,y-rec_height//2),
                                  pt2=(x+rec_width//2,y+rec_height//2),
                                  color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Square'.
        elif shape_selected == 'Square':
            
            # Draw the square on the image that will be either frame or canvas based on current gesture.
            image = cv2.rectangle(image, pt1=(x-shape_size//2, y-shape_size//2),
                                  pt2=(x+shape_size//2, y+shape_size//2),
                                  color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Triangle'.
        elif shape_selected == 'Triangle':
            
            # Get the x and y coordinates of the triangle corners.
            triangle_pts= [(x, y-shape_size//2),
                           (x-shape_size//2, y+shape_size//2),
                           (x+shape_size//2, y+shape_size//2)]
            
            # Draw the triangle on the image that will be either frame or canvas based on current gesture.
            image = cv2.drawContours(image=image, contours=[np.array(triangle_pts, np.int32)], contourIdx=0, 
                                     color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Right Triangle'.
        elif shape_selected == 'Right Triangle':
            
            # Get the x and y coordinates of the right triangle corners.
            triangle_pts= [(x-shape_size//2, y-shape_size//2),
                           (x-shape_size//2, y+shape_size//2),
                           (x+shape_size//2, y+shape_size//2)]
            
            # Draw the right triangle on the image that will be either frame or canvas based on current gesture.
            image = cv2.drawContours(image=image, contours=[np.array(triangle_pts, np.int32)], contourIdx=0, 
                                     color=paint_color, thickness=-1)
    
    # Check if the current hand gesture is 'INDEX MIDDLE THUMB POINTING UP'. 
    if hand_gesture == 'INDEX MIDDLE THUMB POINTING UP':
        
        # Update the selected shape variable to 'None'.
        shape_selected=None
    
    # Return the frame, canvas, and the selected shape.
    return frame, canvas, shape_selected

def detectPoseLandmarks(image, pose, draw=True, display=True):
    '''
    This function performs pose landmarks detection on an image.
    Args:
        image:   The input image with a prominent person whose pose landmarks needs to be detected.
        pose:    The Mediapipe's pose landmarks detection function required to perform the pose detection.
        draw:    A boolean value that is if set to true the function draws the detected landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, the segmentation mask, 
                 the resultant image, and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image:   The input image with the detected pose landmarks drawn.
        pose_landmarks: An array containing the detected landmarks (x and y coordinates) converted into their original scale.
    '''
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Landmarks Detection on the image.
    results = pose.process(imageRGB)
    
    # Initialize a list to store the pose landmarks.
    pose_landmarks = []
    
    # Initialize a variable to store the segmentation mask.
    segmentation_mask =  np.zeros(shape=(height, width), dtype=np.uint8)
    
    # Check if any landmarks are found.
    if results.pose_landmarks:
        
        # Get the segmentation mask of the person detected in the image.
        segmentation_mask = results.segmentation_mask
        
        # Convert the pose landmarks x and y coordinates into their original scale,
        # And store them into a numpy array.
        pose_landmarks = np.array([(landmark.x*width, landmark.y*height)
                                    for landmark in results.pose_landmarks.landmark], dtype=np.int32)
        
        # Check if pose landmarks are specified to be drawn.
        if draw:
            
            # Draw Pose landmarks on the output image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.
                                      get_default_pose_landmarks_style())
    
    # Check if the original input image, the segmentation mask, and the resultant image are specified to be displayed.
    if display:

        # Display the original input image, the segmentation mask, and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(131);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(132);plt.imshow(segmentation_mask, cmap='gray');plt.title("Segmentation Mask");plt.axis('off');
        plt.subplot(133);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');


        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and landmarks detected.
        return output_image, pose_landmarks
        
def calculateAngle(image, landmarks, draw=True, display=True):
    '''
    This function calculates angle between three different landmarks.
    Args:
        image:    The image on which the three points are.
        landmark: The x, y coordinates of the first, second and the third landmark.
        draw:     A boolean value that is if set to true the function draws the points and the lines  
                  joining the points along with an arc between the lines on the image.
        display:  A boolean value that is if set to true the function writes the calculated angle 
                  on the output image, displays the output image and returns nothing.
    Returns:
        angle: The calculated angle between the three landmarks points on the imaghe.
    '''

    # Get the three landmarks x and y coordinates.
    (x1, y1), (x2, y2), (x3, y3) = landmarks

    # Calculate the angle between the three points
    angle = round(math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)), 1)
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Check if the points and the the lines joining the points are specified to be drawn.
    if draw:
            
        # Draw lines between the three points on the image. 
        cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=4)
        cv2.line(image, pt1=(x2, y2), pt2=(x3, y3), color=(255, 0, 255), thickness=4)
        
        # Draw circles on the points on the image.
        cv2.circle(image, center=(int(x1), int(y1)), radius=15, color=(0, 255, 0), thickness=-1)
        cv2.circle(image, center=(int(x2), int(y2)), radius=15, color=(0, 255, 0), thickness=-1)
        cv2.circle(image, center=(int(x3), int(y3)), radius=15, color=(0, 255, 0), thickness=-1)
        
        # Draw an arc between the two lines.
        ####################################################################################################
        
        # Get the starting and ending points of the arc.
        start_point = int((3/4*x2)+(1/4*x1)), int((3/4*y2)+(1/4*y1))
        end_point = int((3/4*x2)+(1/4*x3)), int((3/4*y2)+(1/4*y3))
        
        # Calculate the radius of the arc.
        radius = int(np.sqrt((start_point[0]-end_point[0])*(start_point[0]-end_point[0]) 
                             + (start_point[1]-end_point[1])*(start_point[1]-end_point[1]))) 
        
        # Calculate the starting and ending angles of the elliptic arc in degrees.
        # Starting angle is where you want to start drawing your arc. (This will be 0 if you want to draw a complete circle).
        # Ending angle is where you want to stop drawing your arc. (This will be 360 if you want to draw a complete circle).
        start_angle = int((180/math.pi*math.atan2(start_point[1]-y2, start_point[0]-x2)))
        end_angle = int((180/math.pi*math.atan2(end_point[1]-y2, end_point[0]-x2)))
        
        # Check if starting angle is greater than ending angle.
        # then we have to draw elliptic arc in counterclockwise direction.
        if start_angle > end_angle:
            
            # Subtract 360 from the absolute value of the ending angle.
            # This is required to draw the elliptic arc in counterclockwise direction.
            end_angle = 360 - abs(end_angle)
        
        # Draw the the elliptic arc between the two lines on the image.
        cv2.ellipse(image, center=(x2, y2), axes=(radius, radius), 
                    angle=0., startAngle=start_angle, endAngle=end_angle, color=(0, 255, 0), thickness=10)
        
        ####################################################################################################

    # Check if the image is specified to be displayed.
    if display:
        
        # Write the calculated angle on the frame. 
        cv2.putText(image, f'ANGLE: {angle}', (10, int(frame_height/20)),
                    cv2.FONT_HERSHEY_PLAIN, int(frame_width/350), (255, 255, 0), int(frame_width/200))
        
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise.
    else:
    
        # Return the output image and the calculated angle.
        return image, angle
        
        
def applyNeuralStyleTransfer(image, net, style_image=None, display=True):
    '''
    This function will apply neural style transfer on an image.
    Args:
        image:       The input (content) image whose style needs to be transferred.
        net:         A neural style transfer model (trained on a style image) loaded from the disk.
        style_image: The style image on which the neural style transfer model is trained on.
        display:     A boolean value that is if set to true the function displays the input 
                     (content) image, the style image, and the generated image and returns nothing.
    Returns:
        output_image: A copy of the input image with the neural style transfer applied.
    '''
    
    # Get the height and width of the image.
    image_height, image_width, _ = image.shape
    
    # Create a copy of the input image
    output_image = image.copy()
    
    # Resize the image to make its width 600, while keeping its aspect ratio constant. 
    output_image = cv2.resize(output_image, dsize=(600, int((600/output_image.shape[1])*output_image.shape[0])))
    
    # Get the new height and width of the image.
    new_height, new_width, _ = output_image.shape
    
    # Perform the required pre-processings on the image.
    # This will apply mean subtraction to all 3 channels of the image,
    # and will create a 4-dimensional blob from the image.
    blob = cv2.dnn.blobFromImage(output_image, scalefactor=1.0, size=(new_width, new_height), 
                                 mean=(103.939, 116.779, 123.680), swapRB=False, crop=False)
    
    # Set the input for the model.
    net.setInput(blob)
    
    # Get the current time before performing neural style transfer.
    start = time()
    
    # Perform the neural style transfer on the image.
    output_image = net.forward()
    
    # Get the current time after performing neural style transfer.
    end = time()
    
    # Reshape the generated output of the model.
    # This will change the shape from (1, number of channels, height, width) to (number of channels, height, width).
    output_image = output_image.reshape((3, output_image.shape[2], output_image.shape[3]))
       
    # Add the values back in, that we had subtracted during the pre-processing step.
    output_image[0] += 103.939
    output_image[1] += 116.779
    output_image[2] += 123.680
    
    # Clip the values < 0. This will set the values < 0 in the image to 0.
    output_image[output_image<0] = 0
    
    # Divide the image pixel values with the maximum value in the image, to get the range of 0-1.
    output_image = output_image/output_image.max()
    
    # Change the array ordering to the (height, width, number of channels).
    output_image = output_image.transpose(1, 2, 0)
    
    # Resize the ouput image back to its original size.
    output_image = cv2.resize(output_image, (image_width, image_height))
    
    # Check if the input image and the output image are specified to be displayed.
    if display:
        
        # Display the time take by the process.
        print('Time taken: '+str(round(end - start, 2))+' Seconds.')
        
        # Display the input (content) image, the style image and the output (generated) image.
        plt.figure(figsize=[15,15])
        plt.subplot(131);plt.imshow(image[:,:,::-1]);plt.title("Content Image");plt.axis('off');
        plt.subplot(132);plt.imshow(style_image[:,:,::-1]);plt.title("Style Image");plt.axis('off');
        plt.subplot(133);plt.imshow(output_image[:,:,::-1]);plt.title("Generated Image");plt.axis('off');
        
    # Otherwise.
    else:
        
        # Return the output image.
        return output_image
    
def getFaceKeypoints(image, face_landmarks, draw=True, display=True):
    '''
    This function will extract nose, left eye center, right eye center landmarks.
    Arg:
        image:          A image of the person whose facial landmarks needs to be extracted.
        face_landmarks: An array containing the face landmarks (x and y coordinates) of the face in the image. 
        draw:           A boolean value that is if set to true the function draws the extracted landmarks 
                        on the output image. 
        display:        A boolean value that is if set to true the function displays the original input image, 
                        and the output image with the extracted landmarks drawn and returns nothing.
    Returns:
        output_image:        A copy of input image with extracted landmarks drawn, if they were specified.
        extracted_keypoints: A tuple containing the nose, left eye center, right eye center landmarks.
    '''
    
    # Get the height and width of the image.
    image_height, image_width, _ = image.shape
    
    # Create a copy of the input image to draw facial landmarks.
    output_image = image.copy()
    
    # Get the indexes of the left and right eyes landmarks in a list. 
    LEFT_EYE_INDEXES = set(list(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
    RIGHT_EYE_INDEXES = set(list(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
    
    # Specify the index of the nose tip landmark. 
    NOSE_TIP_INDEX = 1
    
    # Get the nose tip landmark coordinates.
    nose_tip = face_landmarks[NOSE_TIP_INDEX]
    
    # Initialize a list to store the left eye landmarks.
    left_eye_landmarks = [] 
    
    # Iterate over the indexes of the left eye landmarks. 
    for INDEX in LEFT_EYE_INDEXES:

        # Append the landmarks into the list.
        left_eye_landmarks.append(face_landmarks[INDEX])
    
    # Initialize a list to store the right eye landmarks.
    right_eye_landmarks = []
    
    # Iterate over the indexes of the right eye landmarks. 
    for INDEX in RIGHT_EYE_INDEXES:

        # Append the landmarks into the list.
        right_eye_landmarks.append(face_landmarks[INDEX])
    
    # Calculate the center of the left and right eye.
    left_eye_center = np.array(left_eye_landmarks).mean(axis=0)
    right_eye_center = np.array(right_eye_landmarks).mean(axis=0)
    
    # Store the extracted keypoints in a tuple.
    extracted_keypoints = (nose_tip, left_eye_center, right_eye_center)
    
    # Check if the keypoints are specified to be drawn.
    if draw:
        
        # Draw the nose tip landmark on the output image.
        cv2.circle(img=output_image, center=nose_tip, radius=int(image_height/50), color=(255,255,255),
                   thickness=-1)
        
        # Draw the left eye center landmark on the output image.
        cv2.circle(img=output_image, center=(int(left_eye_center[0]), int(left_eye_center[1])),
                   radius=int(image_height/50), color=(255,255,255),thickness=-1)
        
        # Draw the right eye center landmark on the output image.
        cv2.circle(img=output_image, center=(int(right_eye_center[0]), int(right_eye_center[1])), 
                   radius=int(image_height/50), color=(255,255,255),thickness=-1)

    # Check if the original input image and the output image are specified to be displayed.
    if display:
        
        # Display the original input image and the output image.
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
        
    # Otherwise
    else:
        
        # Return the output image and the nose and eyes landmarks.
        return output_image, extracted_keypoints 
    
    
def convertPoints(frame, coordinates):
    '''
    This function will convert points from the coordinate system (used in OpenCV) to the
    coordinate system (used in Pymunk) and viceversa.
    Args:
        coordinates: The x and y coordinates that are about to be converted.
    Returns:
        converted_coordinates: The x and y coordinates after the conversion.
    '''
    
    # Get the height and width of the frame.
    height, width, _ = frame.shape
    
    # Get the x and y coordinates.
    x, y = coordinates
    
    # Convert the coordinates.
    converted_coordinates = int(x), int(height-y)
    
    # Return the converted coordinates.
    return converted_coordinates

def createBall(frame, ball_position, radius):
    '''
    This function creates a ball with actual physical properties.
    Args:
        frame:         A frame/image from a real-time webcam feed. 
        ball_position: The postion of the ball to be created.
        radius:        The radius of the ball to be created.
    Returns:
        body:       The body of the ball created.
        shape:      The shape of the ball created.
        ball_color: The color of the ball created.
    '''
    
    # Calculate the moment of inertia for the ball.
    moment = pymunk.moment_for_circle(mass=1, inner_radius=0, outer_radius=radius)
    
    # Create a dynamic body for the ball.
    body = pymunk.Body(mass=1, moment=moment, body_type=pymunk.Body.DYNAMIC)
    
    # Convert the points from OpenCV coordinates system to the Pymunk coordinates system.
    body.position = convertPoints(frame, ball_position)
    
    # Create a circle shape and attach it to the body. 
    shape = pymunk.Circle(body, radius)
    
    # Set the density of the shape.
    shape.density = 0.0001
    
    # Set the elasticity of the shape.
    shape.elasticity = 0.98
    
    # Set the friction of the shape.
    shape.friction = 1.0
    
    # Get a random color for the ball.
    ball_color = tuple(np.random.choice(range(256), size=3))
    
    # Return the ball's body, shape, and color.
    return (body, shape), ball_color

def drawBalls(frame, balls_to_draw):
    '''
    This function draws balls on a frame/image.
    Args:
        frame:         The frame/image to draw balls on, from a real-time webcam feed. 
        balls_to_draw: A list containing the bodies, radiuses, and the colors of the balls to draw.
    Returns:
        frame: The frame/image with the balls drawn, from a real-time webcam feed.
    '''
    
    # Iterate over the balls to draw.
    for ball_to_draw in balls_to_draw:
        
        # Get the ball's body, shape, and color.
        ball_body, _, radius, ball_color = ball_to_draw
        
        # Get the RGB values of the ball color.
        r, g, b = ball_color
        
        # Convert the ball's position from the Pymunk's coordinates system to the OpenCV coordinates system.
        x, y = convertPoints(frame, ball_body.position)
        
        # Draw the ball on the frame.
        cv2.circle(img=frame, center=(x,y), radius=radius,  color=(int(b), int(g), int(r)), thickness=-1)
    
    # Return the frame with the balls drawn.
    return frame


def createObstacle(space, starting_point, ending_point, thickness):
    '''
    This function creates a static line segment that will act as an obstacle for the objects.
    Args:
        space:          The space which contains the simulation.
        starting_point: The starting point of the line segment, that is to be created.
        ending_point:   The ending point of the line segment, that is to be created.
        thickness:      The thickness of the line segment, that is to be created.
    Returns:
        starting_point: The starting point of the line segment, that is created.
        ending_point:   The ending point of the line segment, that is created.
        thickness:      The thickness of the line segment, that is created.
    '''
    
    # Create a static body for the line segment.
    segment_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    
    # Create a line segment shape and attach it to the body. 
    segment_shape = pymunk.Segment(body=segment_body, a=starting_point, b=ending_point, radius=thickness/2)
    
    # Set the density of the shape. 
    segment_shape.density = 1
    
    # Set the elasticity of the shape.
    segment_shape.elasticity = 0.98
    
    # Set the friction of the shape.
    segment_shape.friction = 1.0
    
    # Add both the body and the shape of the line segment to the simulation.
    space.add(segment_body, segment_shape)
    
    # Return the starting point, ending point, and the thickness of the line segment.
    return (starting_point, ending_point), thickness

def drawObstacles(frame, segments_to_draw):
    '''
    This function draws the line segments on a frame/image.
    Args:
        frame:            The frame/image to draw segments on, from a real-time webcam feed. 
        segments_to_draw: A list containing the starting points, ending points, and the thicknesses of the segments to draw.
    Returns:
        frame: The frame/image with the segments drawn, from a real-time webcam feed.
    '''
    
    # Iterate over the segments to draw.
    for segment_to_draw in segments_to_draw:
        
        # Get the starting point, ending point, and the thickness of the segment, we are iterating upon.
        (starting_point, ending_point), thickness = segment_to_draw
        
        # Conver the starting point and the ending point from the 
        # Pymunk's coordinates system to the OpenCV coordinates system.
        starting_point = convertPoints(frame, starting_point)
        ending_point = convertPoints(frame, ending_point)
        
        # Draw the line segment on the frame.
        cv2.line(img=frame, pt1=starting_point, pt2=ending_point, color=(0,0,255),
                 thickness=thickness)
    
    # Return the frame with the line segments drawn.
    return frame
    
def rotate(point, origin, angle):
    '''
    This function rotates a point (x, y) by an angle (counterclockwise) around the origin.
    Args:
        point:  The x and y coordinates values of the point to be rotated.
        origin: The x and y coordinates values of the origin around which the rotation will be performed.
        angle:  The angle (in radians) with which the point is to be rotated.
    Returns:
        rotated_x: The x-coordinate value of the rotated point.
        rotated_y: The y-coordinate value of the rotated point.
    '''

    # Get the x and y coordinates of the origin.
    origin_x, origin_y = origin

    # Get the x and y coordinates of the point.
    x, y = point

    # Rotate the x-coordinate of the point.
    rotated_x = origin_x + math.cos(angle) * (x - origin_x) - math.sin(angle) * (y - origin_y)

    # Rotate the y-coordinate of the point.
    rotated_y = origin_y + math.sin(angle) * (x - origin_x) + math.cos(angle) * (y - origin_y)

    # Return the x and y coordinates of the rotated point.
    return rotated_x, rotated_y
