import cv2
import mediapipe as mp

mpdraw=mp.solutions.drawing_utils
mphand=mp.solutions.hands
video=cv2.VideoCapture(0)

with mphand.Hands(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as hands:
    
    while True:
        ret,frame=video.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame.flags.writeable=False
        result=hands.process(frame)
        frame.flags.writeable=True
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks:
            for handmark in result.multi_hand_landmarks:
                mpdraw.draw_landmarks(frame,handmark, mphand.HAND_CONNECTIONS)
            
        cv2.imshow("Frame",frame)
        k=cv2.waitKey(1)
        if k == ord('q'):
            break
video.release()
cv2.destroyAllWindows()
    
