import cv2
import numpy as np

def show_message_above_face(frame, message, coordinates, registration_progress=None):
    """
    Display a message above a face in the frame
    
    Args:
        frame: Video frame to draw on
        message: Text to display
        coordinates: Tuple of (x, y, w, h) defining face location
        registration_progress: Optional float between 0-1 showing registration progress
    """
    x, y, w, h = coordinates
    
    # Position the message above the face
    text_x = max(x, 10)
    text_y = max(y - 10, 20)  # Position above the face
    
    # Make the background box slightly larger to accommodate small changes
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    padding = 8  # Extra padding to reduce flickering
    
    # Create a semi-transparent background for the message
    cv2.rectangle(frame, 
                 (text_x - padding, text_y - text_size[1] - padding),
                 (text_x + text_size[0] + padding, text_y + padding),
                 (0, 0, 0), -1)
    
    # Add text
    cv2.putText(frame, message, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add progress bar if registration is in progress
    if registration_progress is not None:
        # Calculate progress bar dimensions
        bar_width = w
        bar_height = 8  # Increase height for better visibility
        filled_width = int(bar_width * registration_progress)
        
        # Position bar below the face
        bar_x = x
        bar_y = y + h + 5
        
        # Draw background bar
        cv2.rectangle(frame, 
                     (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Draw filled portion
        cv2.rectangle(frame, 
                     (bar_x, bar_y),
                     (bar_x + filled_width, bar_y + bar_height),
                     (0, 255, 0), -1)

def show_message(message, frame=None, registration_progress=None):
    """
    Display a general message at the bottom of the frame
    
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