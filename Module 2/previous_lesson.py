import cv2
import math
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


# For homework, eyebrow landmarks
# Initialize a list to store the indexes of the upper part of the right eye outline landmarks.
# Updated the list of indexes for the upper landmarks of the eyebrows
right_eyebrow_upper_ids = [293, 334, 296, 336]
right_eyebrow_lower_ids = [285, 295, 282, 283, 276]

left_eyebrow_upper_ids = [63, 105, 66, 107]
left_eyebrow_lower_ids = [55, 65, 52, 53, 46]

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

        # Set the gesture recognized of the hand to INDEX POINTING UP SIGN.
        hand_gesture = 'INDEX POINTING UP'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 2 and the fingers that are up,
    # are the index and the middle finger.
    elif count[hand_label] == 2 and fingers_statuses[hand_label+'_INDEX'] and \
            fingers_statuses[hand_label+'_MIDDLE']:

        # Set the gesture recognized of the hand to VICTORY SIGN.
        hand_gesture = 'VICTORY'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 3 and the fingers that are up,
    # are the index, pinky, and the thumb.
    elif count[hand_label] == 3 and fingers_statuses[hand_label+'_INDEX'] and \
            fingers_statuses[hand_label+'_PINKY'] and fingers_statuses[hand_label+'_THUMB']:

        # Set the gesture recognized of the hand to SPIDERMAN SIGN.
        hand_gesture = 'SPIDERMAN'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 5.
    elif count[hand_label] == 5:

        # Set the gesture recognized of the hand to HIGH-FIVE SIGN.
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

        # Return the hand gesture name and the fingers tips position of the other hand.
        return hand_gesture, fingers_tips_position['LEFT' if hand_label == 'RIGHT' else 'RIGHT']


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
            cv2.putText(image, f'Distance: {round(distance, 2)}', (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

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

    # For homework, eyebrow landmarks
    right_eyebrow_landmarks = []
    left_eyebrow_landmarks = []
    
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

        # Eyebrow Selection Part (Challenge)
        ####################################################################################################################
        # Iterate over the indexes of the eyebrow ouline.

        # Right Eyebrow
        for index in right_eyebrow_upper_ids+right_eyebrow_lower_ids:
            
            # Get the landmark at the index we are iterating upon,
            # And append it into the list.
            right_eyebrow_landmarks.append(face_landmarks[index])
        
        # Convert the right cheek landmarks list into a numpy array.
        right_eyebrow_landmarks = np.array(right_eyebrow_landmarks, np.int32)
        
        # Draw filled right cheek contours on the copy of the image.
        cv2.drawContours(output_image, contours=[right_eyebrow_landmarks], contourIdx=-1, 
                         color=(255, 255, 255), thickness=-1)    
        
        # Check if the index finger tip is inside the right cheek contours (outline). 
        if cv2.pointPolygonTest(right_eyebrow_landmarks,(index_x, index_y), measureDist=False)  == 1:
            
            # Update the selected face part variable to Right Eyebrow.
            selected_face_part = 'RIGHT EYEBROW'



        # Left Eyebrow
        for index in left_eyebrow_upper_ids+left_eyebrow_lower_ids:
            
            # Get the landmark at the index we are iterating upon,
            # And append it into the list.
            left_eyebrow_landmarks.append(face_landmarks[index])
        
        # Convert the left eyebrow landmarks list into a numpy array.
        left_eyebrow_landmarks = np.array(left_eyebrow_landmarks, np.int32)
        
        # Draw filled right cheek contours on the copy of the image.
        cv2.drawContours(output_image, contours=[left_eyebrow_landmarks], contourIdx=-1, 
                         color=(255, 255, 255), thickness=-1)    
        
        # Check if the index finger tip is inside the right cheek contours (outline). 
        if cv2.pointPolygonTest(left_eyebrow_landmarks,(index_x, index_y), measureDist=False)  == 1:
            
            # Update the selected face part variable to Left Eyebrow.
            selected_face_part = 'LEFT EYEBROW'

            
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

    # For homework, eyebrow landmarks
    right_eyebrow_landmarks = []
    left_eyebrow_landmarks = []
    
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


    # CHALLENGE PART
    # Right Eyebrow
    for index in right_eyebrow_upper_ids+right_eyebrow_lower_ids:
        # Get the landmark at the index we are iterating upon,
        # And append it into the list.
        right_eyebrow_landmarks.append(face_landmarks[index])
    
    # Left Eyebrow
    for index in left_eyebrow_upper_ids+left_eyebrow_lower_ids:
        
        # Get the landmark at the index we are iterating upon,
        # And append it into the list.
        left_eyebrow_landmarks.append(face_landmarks[index])
        

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

    # Check if the specified face part whose mask is needed is the EYES.
    elif face_part == 'EYEBROWS':
        
        # Draw (white) hollow right eye contours on the mask (black canvas) with thickness 2.
        cv2.drawContours(mask, contours=[np.int32(right_eyebrow_landmarks)], contourIdx=-1, 
                         color=(255, 255, 255), thickness=12) 
        
        # Draw (white) hollow left eye contours on the mask (black canvas) with thickness 2.
        cv2.drawContours(mask, contours=[np.int32(left_eyebrow_landmarks)], contourIdx=-1, 
                     color=(255, 255, 255), thickness=12)
        
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
            output_image: A copy of the input image with the Hue, Saturation, and Value channels modified.
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
        saturation_channel *= saturation_scale

        # Scale up or down the pixel values of the value channel accordingly the scale factor.
        value_channel *= value_scale

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
