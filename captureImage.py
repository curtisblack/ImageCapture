# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import requests
import cv2
import numpy as np
import RPi.GPIO as gpio

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    s1 = pts[0].item(0) + pts[0].item(1)
    s2 = pts[1].item(0) + pts[1].item(1)
    s3 = pts[2].item(0) + pts[2].item(1)
    s4 = pts[3].item(0) + pts[3].item(1)
    d1 = pts[0].item(0) - pts[0].item(1)
    d2 = pts[1].item(0) - pts[1].item(1)
    d3 = pts[2].item(0) - pts[2].item(1)
    d4 = pts[3].item(0) - pts[3].item(1)

        

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    #s = pts.sum(axis=1)
    s = [s1, s2, s3, s4]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    #d = np.diff(pts, axis=1)
    d = [d1, d2, d3, d4]
    
    tl = np.argmin(s)
    br = np.argmax(s)
    tr = np.argmax(d)
    bl = np.argmin(d)
 
    rect[0] = pts[tl]
    rect[2] = pts[br]
 
    rect[1] = pts[tr]
    rect[3] = pts[bl]
 
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts, width, height):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
 
    # return the warped image
    return warped

def name():
    t = localtime()
    return str(t.tm_year) + str(t.tm_mon).rjust(2, "0") + str(t.tm_mday).rjust(2, "0") + "T" + str(t.tm_hour).rjust(2, "0") + str(t.tm_min).rjust(2, "0") + str(t.tm_sec).rjust(2, "0") + ".png"

def send(f):
    uploadUrl = "http://www.destined.com/drawtransport/uploadDrawingType1.php"
    imageFile = f
    imageName = time.strftime("%Y%m%dT%H%M%S.png", time.localtime())
    imageType = "image/png"
    response = requests.post(url=uploadUrl, files={"file": (imageName, open(imageFile, "rb"), imageType)})
    print "Image Captured"
    #print response.text

def pointInPoly(point, poly):
    x = float(point.item(0))
    y = float(point.item(1))
    oddNodes = False
    N = len(poly)
    j = N - 1
    for i in range(N):
        pxi = float(poly[i].item(0))
        pyi = float(poly[i].item(1))
        pxj = float(poly[j].item(0))
        pyj = float(poly[j].item(1))
        if (pyi < y and pyj >= y) or (pyj < y and pyi >= y):
            if (pxi+(y-pyi)/(pyj-pyi)*(pxj-pxi) < x):
                oddNodes = not oddNodes
        j = i
    return oddNodes

def isOutside(contour, test):
    for i in range(len(test)):
        if pointInPoly(test[i], contour):
            return False
    return True
    
#def send():
#    n = name()
#    print n
#    camera.capture(data, format="rgb")
#    
#    #os.system('curl -T ' + n + ' ftp.destined.com --user "maas:l3@rn1ng"')

pin = 4
gpio.setmode(gpio.BCM)
gpio.setup(pin, gpio.IN)
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.awb_mode = "off"
camera.awb_gains = (1.37, 2.2)
camera.iso = 100
camera.exposure_mode = "off"
camera.rotation = 180
camera.brightness = 50
camera.contrast = 0
camera.saturation = 20
camera.resolution = (1280, 1024)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=camera.resolution)
mask = cv2.imread("train-mask.png", cv2.IMREAD_UNCHANGED)
#cv2.imshow("Mask", mask)
 
# allow the camera to warmup
time.sleep(0.1)

lastTime = time.time()
frames = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt != None:
        marker = None
        for c in cnts:
            if screenCnt != c:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) != 4 and len(approx) > 2 and isOutside(screenCnt, approx):
                    marker = approx
                    cv2.drawContours(image, [marker], -1, (0, 0, 255), 2)
                    break
        warped = four_point_transform(image, screenCnt, mask.shape[0], mask.shape[1])
        #cv2.imshow("Warped", warped)
        drawing = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
        final = cv2.multiply(drawing, mask, scale=1.0 / 255.0)
        cv2.imshow("Final", final)
        if gpio.input(pin):
            cv2.imwrite("drawing.png", final)
            send("drawing.png")
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    else:
        cv2.destroyWindow("Final")
                
    # show the frame
    #cv2.imshow("Frame", image)
    cv2.imshow("Frame", cv2.resize(image, (0, 0), fx=0.25, fy=0.25))
    #cv2.imshow("Edged", cv2.resize(edged, (0, 0), fx=0.25, fy=0.25))
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    frames += 1
    t = time.time()
    dt = t - lastTime
    if dt > 1:
        lastTime = t
        print "FPS:", frames / dt
        frames = 0
