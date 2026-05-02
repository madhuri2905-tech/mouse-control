#Imports OpenCV to handle video capture and image processing.
import cv2
#importing mediapipe to capture the hand and it's landmarks
import mediapipe as mp
#pyautogui - it is a library which controlls the mouse movement and keyboard usage;
#using pyautogui to control the mouse moments using hand lamdmarks
import pyautogui
#importing numerical python(numpy)
import numpy as np

#Stores the screen width and height in variables.
screen_w,screen_h = pyautogui.size()

#Accesses the hand tracking module from MediaPipe.
mp_hands = mp.solutions.hands
#Initializes the hand detection model to track only one hand.
hands = mp_hands.Hands(max_num_hands = 1)
#cLoads drawing tools to visualize hand landmarks.
mp_draw = mp.solutions.drawing_utils

#Starts capturing video from the default webcam.
cap = cv2.VideoCapture(0)

#sets the x and y coordinates of the screen ; stored in variables prev_ x and prev_y
#Initializes previous mouse coordinates to zero.
prev_x,prev_y = 0,0
#smoothening: smoothens the movement to make it easy for the moments
#Defines smoothing factor for cursor movement.
smoothening = 7

#starting a infinity loop
while True:
    #capturing the image - variable : img
    #sucess is a variable which tells if the img is captures using boolean(T or F)
    success,img = cap.read()
    #fliping the images because we want to see the original image not mirror image 
    #Flips the image horizontally.
    img = cv2.flip(img,1)
    #getting the height and width of the image 
    h,w,_ = img.shape

    #converting BGR to RGB because cv2 works well with RGB and cant work with BGR
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #the hand is captured
    result = hands.process(rgb)

    if result.multi_hand_landmarks :
        #takes and stores all the 21 handlandmarks in the list 
        for handLms in result.multi_hand_landmarks:
            #empty list which has the landmarks stored in it
            lm_list = []

            #sets an id for every landmark
            for id,lm in enumerate(handLms.landmark):
                #gets the x and y coordinates of the landmarks
                cx,cy = int(lm.x*w),int(lm.y*h)
                #appends - id and x and y coordinates of the landmarks into the list
                lm_list.append((id,cx,cy))

            #draws the landmarks on the hand
            mp_draw.draw_landmarks(img,handLms,mp_hands.HAND_CONNECTIONS)

            if lm_list:
                #sets the x and y coordinate of the tip of index finger(landmark = 8)
                x1,y1 = lm_list[8][1],lm_list[8][2]
                #sets the x and y coordinate of the tip of thumb finger(landmark = 4)
                x2,y2 = lm_list[4][1],lm_list[4][2]
                #sets the x and y coordinate of the tip of middle finger(landmarks = 12)
                x3,y3 = lm_list[12][1],lm_list[12][2]

                #a variable which has the video coordinates converted into screen coordinates
                #it is done by the interp function of the library numpy
                screen_x = np.interp(x1, [0,w], [0,screen_w])
                screen_y = np.interp(y1, [0,h], [0,screen_h])

                #smoothening the coordinates both x and y
                curr_x = prev_x +(screen_x-prev_x)/smoothening
                curr_y = prev_y +(screen_y-prev_y)/smoothening

                #moveTo: is a function which is responsible in moving the couser to certain x and y coordinate
                #Moves the mouse cursor to calculated position            
                pyautogui.moveTo(curr_x,curr_y)
                #Updates previous coordinates for next frame
                prev_x,prev_y = curr_x,curr_y

                #Defines a function to calculate distance between two points.
                def distance(p1,p2):
                    #Returns distance between points.
                    return np.hypot(p1[0]-p2[0],p1[1]-p2[1])
                
                #Checks if index finger and thumb are close.
                if distance((x1,y1),(x2,y2))<30:
                    # a left mouse click.
                    pyautogui.click()
                    #Displays "left click" text on screen.
                    cv2.putText(img,'left click',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                #Checks if middle finger and thumb are close.
                if distance((x3,y3),(x2,y2))<30:
                    #Performs a right mouse click.
                    pyautogui.rightClick()
                    #Displays "right click" text on screen.
                    cv2.putText(img,'right click',(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    #Shows the video window with hand tracking
    cv2.imshow('hand gesture control',img)
    #Checks if ESC key is pressed.
    if cv2.waitKey(1) & 0xFF == 27:
        #Exits the loop.
        break
#Releases the webcam.    
cap.release()
#Closes all OpenCV windows.
cv2.destroyAllWindows()




