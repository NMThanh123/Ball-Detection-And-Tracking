# In this code you can adjust range color hsv in line 21 to line 26 to detect ball
import cv2 as cv
import numpy as np
import time


# open video clip
cap = cv.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# function to detect ball
def ball_detect(frame):
    # filter frame
    frame_copy = cv.GaussianBlur(frame, (3,3), 0)

    # convert to hsv
    hsv = cv.cvtColor(frame_copy, cv.COLOR_BGR2HSV)
    # filter balls with inRange function
    low_H = 25
    high_H = 70
    low_S = 120
    high_S = 140
    low_V = 90
    high_V = 255
    mask = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # noise remove with morphology
    kernel_ci = np.array([[0,0,1,0,0],
                          [0,1,1,1,0],
                          [1,1,1,1,1],
                          [0,1,1,1,0],
                          [0,0,1,0,0]], dtype=np.uint8)

    mask = cv.morphologyEx(mask, cv.MORPH_DILATE ,kernel_ci, iterations=1)

    # find contours
    contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    # detect circle
    min_radius = 10
    max_radius = 25
    balls = []
    
    for c in contours:
        (x,y),radius = cv.minEnclosingCircle(c)
        radius = int(radius)
        if (radius > min_radius) and (radius < max_radius):
            x,y,w,h = cv.boundingRect(c)
            balls.append((x,y,w,h))
            
    return balls

# check location ball pass end line
def check_location(box_x, width, end_point):
    if int(box_x + width/2) >= end_point:
        return True
    else:
        return False

def check_start(box_x, width):
    if int(box_x + width/2) > 20:
        return True
    else:
        return False

def main():

    # define parameter 
    count = 0
    tracking = 0
    skip = 1
    number_frame = 0
    pre_frame_time = 0
    next_frame_time = 0
    passing = True
    end_point = 944
    list_obj = []
    
    while cap.isOpened():
        # creat list object to tracking
        temp_list_obj = list_obj
        list_obj = []
        number_frame += 1

        # read next frame
        ok, frame = cap.read()
        if frame is None:
            break
        # resize frame
        frame = cv.resize(frame, (1000, 600))
        # draw line end
        cv.line(frame, (950, 0), (950, 600), (0,0,255), 3)


        if number_frame % skip == 0 or tracking == 0:
            # detect ball 
            balls = ball_detect(frame)
            # if ball are detected -> tracking = 1
            if len(balls) > 0 :
                tracking = 1
                for box in balls:
                    if not check_location(box[0], box[2], end_point):
                        check_object = False
                    
                        # check new object and initialize tracker
                        if not check_object and check_start(box[0], box[2]):
                            new_tracker = cv.TrackerKCF_create()
                            new_tracker.init(frame, box)
                            new_object = {
                                'tracker': new_tracker,
                                'box': box
                            }
                            list_obj.append(new_object)
    

            temp_list_obj = list_obj
        
        if tracking == 1:
            # update tracker
            for i, obj in enumerate(temp_list_obj):
                tracker = obj['tracker']
                check, box = tracker.update(frame)
        
                # draw bounding box ball
                if check:
                    p1 = (int(box[0]-4),int(box[1])-5)
                    p2 = (int(box[0] + box[2]+5),int(box[1] + box[3]+5))
                    cv.rectangle(frame, p1, p2, (0,255,0), 2, 1)
                    cv.putText(frame, 'Ball '+ str(i+1), (int(box[0]-3),int(box[1])-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # check if the center ball pass end line
                    if int((box[0] + box[2]/2)) >= 941:
                        count += 1
                        temp_list_obj.remove(obj) 
                        
                             
        # check ball doesn't in balls
        if balls == [] or temp_list_obj == []:
            tracking = 0
        print(count)

        # display fps and count balls
        next_frame_time = time.time()
        fps = 1/(next_frame_time - pre_frame_time)
        pre_frame_time = next_frame_time
        fps = int(fps)
        cv.putText(frame, 'FPS: ' + str(fps), (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv.putText(frame, 'Ball: ' + str(count), (90, 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # show frame
        cv.imshow("detect ball", frame)
        if cv.waitKey(15) & 0xFF == ord('q') or cv.getWindowProperty('detect ball', cv.WND_PROP_VISIBLE) < 1: 
            break
     
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

