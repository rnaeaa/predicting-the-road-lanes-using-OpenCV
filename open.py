import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_of_intrest_masked(image):
    height=image.shape[0]
    poly=np.array([
        [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poly,255)
    masked_image=cv2.bitwise_and(image,mask)
    #cv2.polylines(image, np.int32([poly]), 1, (255,255,255))
    return masked_image



def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            print(line)
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)#(R,G,B) we are giving blue=255,10 is thickness
            
    return line_image  





def make_cordinates(image,line_parameters):
    slope,intercept=line_parameters[0],line_parameters[1]
    y1=image.shape[0]
    y2=int(y1*3/5)
    x1=int((y1-intercept)/(slope)) #y=mx+c  x=y-c/m
    x2=int((y2-intercept)/(slope))
    return np.array([x1,y1,x2,y2])

    
    



def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1) #see x-cordinate and then y-cordinates not x1,y1
        print(parameters, "\n")
        slope,intercept=parameters[0],parameters[1]
        if slope<0: #from above diagram slope negative for left line
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
            
    print(left_fit)
    print(right_fit)
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    print("left averge",left_fit_average)
    print("right average",right_fit_average)
    left_line=make_cordinates(image,left_fit_average)
    right_line=make_cordinates(image,right_fit_average)
    return np.array([left_line,right_line])
    
    
    


def canny(image):
	gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	blur=cv2.GaussianBlur(gray,(5,5),0)
	canny_image=cv2.Canny(blur,50,150)
	return canny_image

cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_image=canny(frame)
    cropped_image=region_of_intrest_masked(canny_image)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines=average_slope_intercept(frame,lines)
    line_image=display_lines(frame,averaged_lines)

    combo_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow("result",combo_image)
    if cv2.waitKey(1)==ord('q'):
        break

        
cap.release()
cv2.destroyAllWindows()



