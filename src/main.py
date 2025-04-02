import cv2
import time
from face_detector import recognize_face
from database import check_player, add_player
from display_manager import show_message

def main():
    # Initialize the facial recognition system
    video_capture = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video device.")
        return

    # Create window with normal resizable behavior
    cv2.namedWindow('Event Game', cv2.WINDOW_NORMAL)
    
    # Initialize message and tracking variables
    message = "Welcome! Please stand in front of the camera."
    registration_start_time = None
    registration_duration = 10.0  # seconds required for registration
    current_face_id = None
    registration_complete = False
    last_recognition_time = 0
    recognition_cooldown = 0.5  # Check more frequently than before

    while True:
        # Capture a single frame from the video feed
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        current_time = time.time()
        registration_progress = None
        
        # Only perform recognition every few seconds to avoid excessive processing
        if current_time - last_recognition_time > recognition_cooldown:
            # Process the captured frame for facial recognition
            player_id = recognize_face(frame)
            last_recognition_time = current_time

            # When a face is detected
            if player_id:
                # If this is a returning player who has already played
                if check_player(player_id):
                    message = "Sorry! You have already played."
                    registration_progress = None
                    registration_start_time = None
                    current_face_id = None
                else:
                    # If this is the same face we've been tracking
                    if current_face_id == player_id:
                        if registration_start_time is None:
                            # Start tracking this face
                            registration_start_time = current_time
                            message = "Please hold still for registration..."
                        else:
                            # Calculate time elapsed
                            elapsed_time = current_time - registration_start_time
                            # Calculate progress percentage (0.0 to 1.0)
                            registration_progress = min(elapsed_time / registration_duration, 1.0)
                            
                            # Update message based on progress
                            message = f"Registering... Please remain still"
                            
                            # Check if registration is complete
                            if elapsed_time >= registration_duration and not registration_complete:
                                add_player(player_id)
                                message = "Registration complete! You're now ready to play!"
                                registration_complete = True
                    else:
                        # A different face detected, reset tracking
                        current_face_id = player_id
                        registration_start_time = current_time
                        registration_complete = False
                        message = "New face detected. Please hold still for registration..."
            else:
                # No face detected, reset tracking
                message = "No face detected. Please position yourself in the frame."
                registration_start_time = None
                current_face_id = None
                registration_progress = None
        
        # Display message on frame
        frame = show_message(message, frame, registration_progress)
            
        # Display the resulting frame
        cv2.imshow('Event Game', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()