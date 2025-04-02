import cv2

def show_message(message, frame=None, registration_progress=None):
    """
    Display a message in the console and optionally on a frame
    
    Args:
        message: Text to display
        frame: Optional video frame to draw text on
        registration_progress: Optional float between 0-1 showing registration progress
    """
    # If a frame is provided, draw the message on it
    if frame is not None:
        # Create a semi-transparent overlay for the message
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, frame.shape[0] - 60), 
                     (frame.shape[1] - 10, frame.shape[0] - 10), 
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(overlay, message, (20, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add progress bar if registration is in progress
        if registration_progress is not None:
            # Calculate progress bar dimensions
            bar_width = frame.shape[1] - 40
            bar_height = 10
            filled_width = int(bar_width * registration_progress)
            
            # Draw background bar
            cv2.rectangle(overlay, 
                         (20, frame.shape[0] - 80),
                         (20 + bar_width, frame.shape[0] - 70),
                         (100, 100, 100), -1)
            
            # Draw filled portion
            cv2.rectangle(overlay, 
                         (20, frame.shape[0] - 80),
                         (20 + filled_width, frame.shape[0] - 70),
                         (0, 255, 0), -1)
            
            # Show percentage
            percent_text = f"{int(registration_progress * 100)}%"
            cv2.putText(overlay, percent_text, 
                       (20 + bar_width + 10, frame.shape[0] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame