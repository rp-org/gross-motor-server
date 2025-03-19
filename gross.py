import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp

# Trained model
model = tf.keras.models.load_model("v07_Action_Recognition_15_epochs_66_videos_per_action_train_106_test_26.keras")

# Initialize MediaPipe components for pose detection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define actions
actions = np.array(['hands_up', 't_pose'])    

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# Function to process a video and make a prediction
def predict_action(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []  # store keypoints over frames
    sequence_length = 30

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame_num in range(sequence_length):  # process first 30 frames
            ret, frame = cap.read()
            if not ret:
                break

            # convert to RGB and process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

    cap.release()

    # if sequence is shorter than 30, pad with last frame's keypoints
    while len(sequence) < sequence_length:
        sequence.append(sequence[-1])  # repeat last frame's keypoints

    # convert sequence to NumPy array and reshape
    sequence = np.expand_dims(np.array(sequence), axis=0)

    # prediction
    prediction = model.predict(sequence)
    predicted_action = actions[np.argmax(prediction)]           

    print(f"\nPredicted Action: {predicted_action}")

    return predicted_action

# Get feedback for incorrect action
def incorrect_action_feedback(expected_action, predicted_action):
  feedback_map = {
    ("hands_up", "t_pose"): "Oops! Try again! Let's raise hands!.",
    ("t_pose", "hands_up"): "Oops! Try again! Let's make a T Pose!"
  }
  return feedback_map.get((expected_action, predicted_action), "")

# function to calculate angle
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
      angle = 360-angle

    return angle

# Extract angles
def extract_angles(frame, landmarks):

  def get_landmark_coordinates(landmark):
    return [landmarks[landmark].x, landmarks[landmark].y]
  
  # exract normalized landmarks
  # left
  left_hip, left_shoulder, left_wrist = map(get_landmark_coordinates, [mp_holistic.PoseLandmark.LEFT_HIP.value, mp_holistic.PoseLandmark.LEFT_SHOULDER.value, mp_holistic.PoseLandmark.LEFT_WRIST.value])
  # right
  right_hip, right_shoulder, right_wrist = map(get_landmark_coordinates, [mp_holistic.PoseLandmark.RIGHT_HIP.value, mp_holistic.PoseLandmark.RIGHT_SHOULDER.value, mp_holistic.PoseLandmark.RIGHT_WRIST.value])

  # calculate angle
  left_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
  right_angle = calculate_angle(right_hip, right_shoulder, right_wrist)

  return left_angle, right_angle

# Ensure angles are in [0, 360] range
def normalize_angle(angle):
    angle = int(angle)  # Ensure integer conversion
    return angle + 360 if angle < 0 else angle

# Calculate stars
def calculate_score(action, left_angle, right_angle):
  action_ideal_angles = {
    "hands_up": 170,  # midpoint of 150-190
    "t_pose": 95      # midpoint of 80-110
  }

  if action not in action_ideal_angles:
    return 0  # unknown actions
  
  ideal_angle = action_ideal_angles[action]
    
  # left right deviation from the ideal angle
  left_deviation = abs(left_angle - ideal_angle)
  right_deviation = abs(right_angle - ideal_angle)
    
  # avg deviation
  avg_deviation = (left_deviation + right_deviation) / 2

  max_deviation = 40  # max acceptable deviation (beyond this - 0 stars)
  score = max(100  - (avg_deviation / max_deviation) * 100 , 0)  # normalize score to [0,100]
  print("Score: ", score, " Avg: ", avg_deviation)

  # round 
  return round(score)

# Generate final feedback message
def generate_final_feedback(predicted_action, left_angle, right_angle):
  feedback_templates = {
      "hands_up": [
            (170, 190, "Perfect! Your hands are raised correctly. Keep it up!"),
            (160, 190, "Great job! Your hands are fully extended. You're doing it right!"),
            (150, 190, "Almost there! Try lifting your hands a bit higher for the perfect posture."),
            (0, 149, "Oops! Try again!"),
      ],

      "t_pose": [
            (100, 110, "Awesome! Your arms are perfectly aligned for the T-pose. Well done!"),
            (90, 110, "Nice work! You’ve got the right posture!"),
            (80, 110, "Almost correct! Adjust your arms slightly to maintain the perfect T-pose."),
            (111, 180, "Try to lower your hands a bit!"),
            (0, 79, "Try to raise your hands a bit!"),
      ]
  }

  for min_angle, max_angle, feedback in feedback_templates.get(predicted_action, []):
    if min_angle <= left_angle <= max_angle and min_angle <= right_angle <= max_angle:
      return feedback

  return "Oops! Try again!"

# Process a video and make prediction
def predict_video(video_path, expected_action):
    cap = cv2.VideoCapture(video_path)
    sequence = []  # store keypoints over frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequence_length = 30
    star_count = 0
    feedback=""

    landmarks = None

    # perform action prediction
    predicted_action = predict_action(video_path)

    print("EXPECTED: ", expected_action,  " PREDICTED: ", predicted_action)

    # early feedback for incorrect action
    feedback = incorrect_action_feedback(expected_action, predicted_action)
    if feedback:
        return {"predicted_action": predicted_action, "score": 0, "feedback": feedback}

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame_num in range(min(frame_count, sequence_length)):  # process first 30 frames
            ret, frame = cap.read()
            if not ret:
                break

            # convert to RGB and process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # extract landmarks 
            if results.pose_landmarks:
              landmarks = results.pose_landmarks.landmark

            # extract angles
            if landmarks:
              left_angle, right_angle = extract_angles(frame, landmarks)
            else: 
              left_angle, right_angle = 0, 0

        print("RIGHT ANGLE: ", right_angle, " LEFT ANGLE: ", left_angle)
        
        feedback = generate_final_feedback(predicted_action, left_angle, right_angle)
      
        score = calculate_score(predicted_action, left_angle, right_angle)   

    cap.release()
    return {"predicted_action": predicted_action, "score": score, "feedback": feedback}