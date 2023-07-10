import cv2

# Load the pre-trained car and motorcycle classifiers
car_classifier = cv2.CascadeClassifier('car_classifier.xml')
motorcycle_classifier = cv2.CascadeClassifier('motorcycle_classifier.xml')

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the codec for the video output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars and motorcycles in the frame
    cars = car_classifier.detectMultiScale(gray, 1.1, 5)
    motorcycles = motorcycle_classifier.detectMultiScale(gray, 1.1, 5)

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Draw rectangles around the detected motorcycles
    for (x, y, w, h) in motorcycles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Motorcycle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video file
    output.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture, close the output video file, and close the window
video_capture.release()
output.release()
cv2.destroyAllWindows()
