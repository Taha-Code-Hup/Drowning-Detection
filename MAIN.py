import cv2
import time
import requests
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')

# Path to the input video and output video
input_video_path = 'video.mp4'
output_video_path = 'output_video.avi'

# Load the video
cap = cv2.VideoCapture(input_video_path)

# Get video details
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define VideoWriter to save the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize tracking data for persons
person_tracking = {}
no_movement_duration = 4  # Threshold in seconds for detecting drowning
emergency_detected = False  # Flag to indicate drowning detected
last_telegram_sent_time = time.time()  # Track when the last message was sent
message_interval = 30  # Send messages every 4 seconds if new drowning is detected

# Telegram bot details (replace with your bot token and chat ID)
TELEGRAM_BOT_TOKEN = '6935545635:AAHTNLop7_m4POMWuxSeMgfGCal_helyFyA'
CHAT_ID = '328102547'

def send_telegram_message(message, image_path=None):
    # Send the message
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    params = {
        'chat_id': CHAT_ID,
        'text': message
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("Message sent to Telegram successfully.")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")

    # Send the image if provided
    if image_path:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
        with open(image_path, 'rb') as image_file:
            files = {'photo': image_file}
            params = {'chat_id': CHAT_ID}
            response = requests.post(url, files=files, params=params)
            if response.status_code == 200:
                print("Image sent to Telegram successfully.")
            else:
                print(f"Failed to send image. Status code: {response.status_code}")

def interpolate_box(last_box, next_box, alpha):
    x1 = int(last_box[0] * (1 - alpha) + next_box[0] * alpha)
    y1 = int(last_box[1] * (1 - alpha) + next_box[1] * alpha)
    x2 = int(last_box[2] * (1 - alpha) + next_box[2] * alpha)
    y2 = int(last_box[3] * (1 - alpha) + next_box[3] * alpha)
    return (x1, y1, x2, y2)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect persons in the current frame
    results = model(frame)[0]
    current_time = time.time()

    detected_person_ids = set()
    new_drowning_person_detected = False  # Track if a new drowning person is detected in this frame

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0].item()
        cls = int(result.cls[0].item())
        label = model.names[cls]

        if label == 'person':
            # Unique ID for each person based on their bounding box coordinates
            person_id = f"{x1}_{y1}_{x2}_{y2}"
            detected_person_ids.add(person_id)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if person_id not in person_tracking:
                # New person detected
                print(f"New person detected with ID {person_id} at frame {frame_count}")
                person_tracking[person_id] = {
                    'last_box': (x1, y1, x2, y2),
                    'last_movement_time': current_time,
                    'last_frame': frame_count,
                    'next_box': (x1, y1, x2, y2),
                    'next_frame': frame_count,
                    'drowning_flagged': False
                }
            else:
                # Update the person's movement tracking
                last_box = person_tracking[person_id]['last_box']
                distance_moved = ((center_x - (last_box[0] + last_box[2]) // 2) ** 2 +
                                  (center_y - (last_box[1] + last_box[3]) // 2) ** 2) ** 0.5

                if distance_moved < 10:
                    time_no_movement = current_time - person_tracking[person_id]['last_movement_time']
                    if time_no_movement > no_movement_duration:
                        # Person is flagged as drowning
                        if not person_tracking[person_id]['drowning_flagged']:
                            person_tracking[person_id]['drowning_flagged'] = True
                            new_drowning_person_detected = True
                            emergency_detected = True
                            print(f"Drowning detected for person {person_id} at frame {frame_count}")

                # Update movement tracking
                person_tracking[person_id]['last_movement_time'] = current_time
                person_tracking[person_id]['last_box'] = (x1, y1, x2, y2)
                person_tracking[person_id]['last_frame'] = frame_count

                # Set color based on drowning status
                color = (0, 0, 255) if person_tracking[person_id]['drowning_flagged'] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Update the next_box and next_frame for interpolation
            person_tracking[person_id]['next_box'] = (x1, y1, x2, y2)
            person_tracking[person_id]['next_frame'] = frame_count

    # Send emergency message and image if a new person is detected drowning and 4 seconds have passed
    if new_drowning_person_detected and (current_time - last_telegram_sent_time >= message_interval):
        # Save the current frame as an image
        image_path = f"drowning_{frame_count}.jpg"
        cv2.imwrite(image_path, frame)

        # Send a message to Telegram
        telegram_message = f"EMERGENCY: Drowning detected at frame {frame_count}."
        send_telegram_message(telegram_message, image_path)

        # Update the last telegram sent time
        last_telegram_sent_time = current_time

    # Interpolation for missing frames
    for person_id, data in person_tracking.items():
        if person_id not in detected_person_ids:
            # Interpolate if detection is missing for up to 10 frames
            if data['last_frame'] is not None and frame_count - data['last_frame'] <= 10:
                # Calculate the interpolation factor (alpha) based on frame count
                alpha = (frame_count - data['last_frame']) / (data['next_frame'] - data['last_frame'] + 1e-5)
                interpolated_box = interpolate_box(data['last_box'], data['next_box'], alpha)

                x1, y1, x2, y2 = interpolated_box

                # Set color to red if drowning has been detected, else green
                color = (0, 0, 255) if emergency_detected else (0, 255, 0)

                # Draw the interpolated bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, 'Interpolated', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display emergency message on the screen if drowning detected
    if emergency_detected:
        emergency_text = "EMERGENCY: DROWNING"
        font_scale = 3
        font_thickness = 4
        text_size, _ = cv2.getTextSize(emergency_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_width, text_height = text_size
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2

        # Display the emergency message in red at the center of the frame
        cv2.putText(frame, emergency_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), font_thickness)

    # Write the processed frame to the output video
    out.write(frame)

    # Show the current frame in a window
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
