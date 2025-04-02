import cv2
import time
import numpy as np
from face_detector import recognize_faces
from database import check_player, add_player
from display_manager import show_message, show_message_above_face

def main():
    # Initialize the facial recognition system
    video_capture = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video device.")
        return

    # Create window with normal resizable behavior
    cv2.namedWindow('Event Game', cv2.WINDOW_NORMAL)
    
    # Dictionary to track registration progress for each face
    # Format: {player_id: {'start_time': timestamp, 'complete': boolean, 'last_seen': timestamp, 
    #                       'smooth_coords': (x, y, w, h)}}
    registration_tracking = {}
    
    # Other tracking variables
    registration_duration = 10.0  # seconds required for registration
    last_recognition_time = 0
    recognition_cooldown = 0.5  # seconds between recognition attempts
    face_tracking_timeout = 2.0  # seconds to keep tracking a face that disappeared
    display_general_message = "Welcome! Step in front of the camera to play."
    smoothing_factor = 0.3  # Lower = smoother but more lag (0.0-1.0)

    while True:
        # Capture a single frame from the video feed
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        current_time = time.time()
        frame_with_messages = frame.copy()  # Create a copy to draw messages on
        
        # Only perform recognition periodically to avoid excessive processing
        if current_time - last_recognition_time > recognition_cooldown:
            # Process the captured frame for facial recognition
            detected_faces = recognize_faces(frame)
            last_recognition_time = current_time
            
            # Update tracking for detected faces
            current_ids = {face['player_id'] for face in detected_faces}
            
            # Process each detected face
            for face in detected_faces:
                player_id = face['player_id']
                current_coords = face['coordinates']
                has_played = face['has_played']
                
                # Update tracking info
                if player_id in registration_tracking:
                    # Update last seen time
                    tracking_info = registration_tracking[player_id]
                    tracking_info['last_seen'] = current_time
                    
                    # Apply exponential smoothing to coordinates
                    if 'smooth_coords' in tracking_info:
                        x_old, y_old, w_old, h_old = tracking_info['smooth_coords']
                        x_new, y_new, w_new, h_new = current_coords
                        
                        # Smooth the coordinates
                        x_smooth = int(smoothing_factor * x_new + (1 - smoothing_factor) * x_old)
                        y_smooth = int(smoothing_factor * y_new + (1 - smoothing_factor) * y_old)
                        w_smooth = int(smoothing_factor * w_new + (1 - smoothing_factor) * w_old)
                        h_smooth = int(smoothing_factor * h_new + (1 - smoothing_factor) * h_old)
                        
                        tracking_info['smooth_coords'] = (x_smooth, y_smooth, w_smooth, h_smooth)
                    else:
                        # First time seeing this face, initialize smooth coordinates
                        tracking_info['smooth_coords'] = current_coords
                else:
                    # New face, initialize tracking
                    registration_tracking[player_id] = {
                        'start_time': current_time,
                        'complete': False,
                        'last_seen': current_time,
                        'smooth_coords': current_coords
                    }
        
        # Clean up tracking for faces not seen for a while
        for tracked_id in list(registration_tracking.keys()):
            if current_time - registration_tracking[tracked_id]['last_seen'] > face_tracking_timeout:
                registration_tracking.pop(tracked_id)
        
        # Display messages for all tracked faces (not just recently detected ones)
        for player_id, tracking_info in registration_tracking.items():
            # Skip faces that haven't been seen recently
            if current_time - tracking_info['last_seen'] > face_tracking_timeout:
                continue
                
            # Use the smoothed coordinates for display
            coordinates = tracking_info['smooth_coords']
            
            # Get player status
            has_played = check_player(player_id)
            
            # Create appropriate message
            if has_played:
                message = "Already played"
                show_message_above_face(frame_with_messages, message, coordinates)
            elif tracking_info['complete']:
                message = "Ready to play!"
                show_message_above_face(frame_with_messages, message, coordinates)
            else:
                # Calculate time elapsed for this face
                elapsed_time = current_time - tracking_info['start_time']
                # Calculate progress percentage (0.0 to 1.0)
                registration_progress = min(elapsed_time / registration_duration, 1.0)
                
                if registration_progress >= 1.0:
                    # Registration complete, mark as played
                    add_player(player_id)
                    tracking_info['complete'] = True
                    message = "Ready to play!"
                    show_message_above_face(frame_with_messages, message, coordinates)
                else:
                    # Still registering
                    message = "Registering..."
                    show_message_above_face(frame_with_messages, message, coordinates, registration_progress)
        
        # Show general message at bottom if no faces being tracked
        if not registration_tracking:
            frame_with_messages = show_message(display_general_message, frame_with_messages)
            
        # Display the resulting frame
        cv2.imshow('Event Game', frame_with_messages)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()