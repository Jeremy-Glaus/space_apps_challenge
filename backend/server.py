# Python 3 server example
from asyncio.windows_events import NULL
from cgitb import text
from distutils.command.build import build
from http.server import BaseHTTPRequestHandler, HTTPServer
from pydoc import resolve
from turtle import color, textinput
import requests
import base64

import cv2
import numpy as np

# Server configuration
hostName = "localhost"
serverPort = 8080

# Global variables
resultImages = []

API_URL = "https://api-inference.huggingface.co/models/vblagoje/bert-english-uncased-finetuned-pos"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def getNasaImage(imageUrl):
    print("get")

def findObjects(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binarize the image
    ret, bw = cv2.threshold(gray, 128, 255, 
    cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # find connected components
    connectivity = 4
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 250 #threshhold value for objects in scene
    img2 = np.zeros((img.shape), np.uint8)
    for i in range(0, nb_components+1):
        if sizes[i - 1] >= min_size:
            color = np.random.randint(255,size=3)
            # draw the bounding rectangele around each object
            cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
            img2[output == i + 1] = color
    cv2.imwrite('found_objects_image.png', img2)

def getNasaImages(searchFor, inputText):
    print("Requesting https://images-api.nasa.gov/search?media_type=image&q="+searchFor)
    response = requests.get("https://images-api.nasa.gov/search?media_type=image&q="+searchFor)
    response = response.json();

    items = response.get("collection").get("items")
    counter = 0;
    for entity in items:
        # Get Image es base64
        imageurl = entity.get("links")[0].get("href")
        image = base64.b64encode(requests.get(imageurl).content)

        #Get Image from base64
        nparr = np.fromstring(base64.b64decode(image), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        findObjects(img)

        #Read image properties
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

        #Save original image for debugging
        cv2.imwrite('original_image.png', img)

        print('PATH: ' + inputText)
        # Determine color
        colorimg = NULL
        if ('yellow' in inputText):
            print("yellow")
            colorimg = np.full(img.shape, (255,255,0), np.uint8)
        elif ('red' in inputText):
            print("red")
            colorimg = np.full(img.shape, (0,255,255), np.uint8)
        elif ('blue' in inputText):
            print("blue")
            colorimg = np.full(img.shape, (255,0,0), np.uint8)
        elif ('green' in inputText):
            print("green")
            colorimg = np.full(img.shape, (0,255,0), np.uint8)
        elif ('purple' in inputText):
            print("purple")
            colorimg = np.full(img.shape, (150,0,255), np.uint8)
        else:
            print("white")
            colorimg = np.full(img.shape, (0,0,0), np.uint8)

        #Add color
        cv2.imwrite('color_image.png', colorimg)
        fused_img  = cv2.addWeighted(img, 0.8, colorimg, 0.5, 0)
        
        #Save processed image for debugging
        img = cv2.imwrite('processed_image.png', fused_img)

        #Limit numbers of images
        if(counter == 0):
            break;
        counter = counter + 1

class MyServer(BaseHTTPRequestHandler):
    #handle GET-Requests
    def do_GET(self):
        #Do nothing if the favicon is request
        print(self.path)
        if('favicon' not in self.path):
            text = self.path.split("?")[1]
            inputText = text
            #Send query to classification api
            print("Query: " + text)
            output = query({
                "inputs": text,
            })

            #Requests to NASA-API with every noun
            for entity in output:
                group = entity.get("entity_group")
                if ((group == "PROPN") or group == "NOUN"):
                    print(entity.get("entity_group"))
                    getNasaImages(entity.get("word"), text)

        #Ending headers
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

#random stuff
if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")