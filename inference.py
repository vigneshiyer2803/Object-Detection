from ultralytics import YOLO

model = YOLO("runs/detect/yolov10-bananas/weights/best.pt") #once weight are develped download it
results = model("test_images/banana.jpg")  #Change the image and path according to your path and image
results[0].show()
