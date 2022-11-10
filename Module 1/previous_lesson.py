import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2, min_detection_confidence=0.3)
hands_videos = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


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


def countFingers(image, results, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image.
        draw:    A boolean value that is if set to true the function writes the total count of 
                 fingers up, of the hands on the image.
        display: A boolean value that is if set to true the function displays the resultant image
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

    elif count[hand_label] == 1 and fingers_statuses[hand_label+'_PINKY']:

        # Set the gesture recognized of the hand to INDEX POINTING UP SIGN.
        hand_gesture = 'PINKY'

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


def changeSatValue(image, scale_factor, channel, display=True):
    '''
    This function will increase/decrease the Saturation or Brighness of an image.
    Args:
        image:        The image whose Saturation or Brighness is to be changed.
        scale_factor: A number that will multiply/scale the required channel of the image.
        channel:      The channel either Saturation or Value whose needed to be modified.
        display:      A boolean value that is if set to true the function displays the original image,
                      and the output image with the modified Saturation or Brighness and returns nothing.
    Returns:
        output_image: A copy of the input image with the Saturation or Brighness modified.

    '''

    # Convert the image from BGR into HSV format.
    image_hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

    # Convert the pixel values of the image into float.
    image_hsv = np.array(image_hsv, dtype=np.float64)

    # Split the hue, saturation, and value channel of the image.
    hue_channel, saturation_channel, value_channel = cv2.split(image_hsv)

    # Check if the channel that is needed to be changed is Saturation.
    if channel == 'Saturation':

        # Scale up or down the pixel values of the channel utilizing the scale factor.
        saturation_channel *= scale_factor

    # Check if the channel that is needed to be changed is Value.
    elif channel == 'Value':

        # Scale up or down the pixel values of the channel utilizing the scale factor.
        value_channel *= scale_factor

    # Merge the Hue, Saturation, and Value channel.
    image_hsv = cv2.merge((hue_channel, saturation_channel, value_channel))

    # Set values > 255 to 255.
    image_hsv[image_hsv > 255] = 255

    # Set values < 0 to 0.
    image_hsv[image_hsv < 0] = 0

    # Convert the image into uint8 type and BGR format.
    output_image = cv2.cvtColor(
        np.array(image_hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

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

    # Otherwise
    else:

        # Return the output image.
        return output_image


def changeContrast(image, scale_factor, display=True):
    '''
    This function will modify the Contrast of an image.
    Args:
        image:        The image whose Contrast is to be changed.
        scale_factor: A number that will scale the Contrast of the image.
        display:      A boolean value that is if set to true the function displays the original image,
                      and the output image with the modified Contrast and returns nothing.
    Returns:
        output_image: A copy of the input image with the Contrast modified.

    '''

    # Change the contrast of a copy of the image.
    output_image = cv2.convertScaleAbs(
        image.copy(), alpha=float(scale_factor), beta=0)

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

    # Otherwise
    else:

        # Return the output image.
        return output_image


def gammaCorrection(frame, scale_factor, display=True):
    '''
    This function will perform the Gamma Correction of an image.
    Args:
        image:        The image on which Gamma Correction is to be performed.
        scale_factor: A number that will be used to calculate the required gamma value.
        display:      A boolean value that is if set to true the function displays the original image,
                      and the output image with the modified Contrast and returns nothing.
    Returns:
        output_image: A copy of the input image with the gamma corrected. 
    '''

    # Calculate the gamma value from the passed scale factor.
    gamma = 1.0/scale_factor

    # Initialize the look-up table of 256 elements with values zero.
    table = np.zeros(shape=(1, 256), dtype=np.uint8)

    # Iterate the number of times equal to the number of columns (256) of the look-up table.
    for i in range(table.shape[1]):

        # Calculate the value of the ith column of the look-up table.
        # And clip (limit) the values between 0 and 255.
        table[0, i] = np.clip(a=pow(i/255.0, gamma)*255.0, a_min=0, a_max=255)

    # Perform look-up table transform of the image.
    output_image = cv2.LUT(frame, table)

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

    # Otherwise.
    else:

        # Return the output image.
        return output_image
