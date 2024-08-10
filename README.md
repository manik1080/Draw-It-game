# Draw-It on an air canvas

**Draw-it** is an interactive game that challenges users to draw various objects. This game takes it a step further, by allowing users to draw their sketches in the air using an air canvas. The game uses computer vision and machine learning methods, specifically OpenCV, MediaPipe, and a TensorFlow Lite model, to track hand movements, draw on an air canvas and recognize the objects drawn.


## Detailed Description

- **Hand Tracking**: Using MediaPipe, we detect hand landmarks in real-time, users can draw or erase on the air canvas by raising one or two fingers, respectively.
- **Air Canvas**: Users can draw or erase on the air canvas by raising one or two fingers, respectively, with sketches appearing on the screen in real-time.
- **Object Recognition**: A TensorFlow model is trained to recognise the sketches as one of the classes: `airplane`, `ambulance`, `apple`, `axe`, `banana`, `basket`, `bed`, `carrot`, `cat` or `fish`.
- **Class Selection**: The game randomly selects a class (e.g., `cat`, `carrot`) for the user to draw.
- **Lightweight Execution**: The TensorFlow model has been quantized and converted to TensorFlow Lite, allowing it to run efficiently on a CPU.
- **Timed Gameplay**: The user has 10 or 20 seconds to draw the given object. The model will try to recognize the drawing, and if correct, the player earns a point.


## Installation

1. Clone repository
2. Install dependencies using:
   `pip install -r requirements.txt`
3. Run main.py
   `python main.py`
