import cv2
import numpy as np
import sqlite3
import os
import shutil
import pickle

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def initialize_db():
    """Create the database if it doesn't exist"""
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Check if database file exists but is corrupted
    db_path = 'data/players_database.db'
    if os.path.exists(db_path):
        try:
            # Test connection and check if face_features column exists
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if the table has the correct schema
            cursor.execute("PRAGMA table_info(players)")
            columns = [info[1] for info in cursor.fetchall()]
            
            # If the table doesn't have face_features column, drop and recreate it
            if 'face_features' not in columns:
                print("Updating database schema...")
                cursor.execute("DROP TABLE IF EXISTS players")
                conn.commit()
            
            conn.close()
        except sqlite3.DatabaseError:
            # If corrupted, backup and remove the file
            print("Database file corrupted. Creating new database.")
            backup_path = db_path + '.backup'
            if os.path.exists(backup_path):
                os.remove(backup_path)
            shutil.move(db_path, backup_path)
    
    # Create new database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_features BLOB,
        played BOOLEAN DEFAULT FALSE
    )
    ''')
    conn.commit()
    conn.close()

def extract_face_features(face_img):
    """Extract face features using OpenCV"""
    try:
        # Resize to a standard size
        resized = cv2.resize(face_img, (100, 100))
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better feature extraction
        equalized = cv2.equalizeHist(gray)
        # Flatten the image into a feature vector
        features = equalized.flatten()
        # Return normalized features
        return features / 255.0
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def recognize_faces(frame):
    """Process a video frame and recognize multiple faces using OpenCV"""
    # Initialize database if needed
    initialize_db()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return []  # No faces detected
    
    # Connect to the database
    conn = sqlite3.connect('data/players_database.db')
    cursor = conn.cursor()
    
    # List to store face info: (player_id, face_coordinates, has_played)
    detected_faces = []
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        # Extract features
        features = extract_face_features(face_img)
        if features is None:
            continue
            
        # Check if this face exists in the database
        cursor.execute("SELECT id, face_features, played FROM players")
        best_match_id = None
        best_match_score = float('inf')  # Lower is better
        has_played = False
        
        for row in cursor.fetchall():
            player_id, stored_features_blob, played = row
            if stored_features_blob:
                stored_features = pickle.loads(stored_features_blob)
                
                # Compare using Euclidean distance
                distance = np.linalg.norm(features - stored_features)
                
                # If distance is small enough, consider it a match
                if distance < 40 and distance < best_match_score:  # Threshold value
                    best_match_id = player_id
                    best_match_score = distance
                    has_played = bool(played)
        
        if best_match_id:
            # Add to detected faces list
            face_info = {
                'player_id': best_match_id,
                'coordinates': (x, y, w, h),
                'has_played': has_played
            }
            detected_faces.append(face_info)
        else:
            # If no match, add new face to database but don't mark as played yet
            features_blob = pickle.dumps(features)
            cursor.execute("INSERT INTO players (face_features, played) VALUES (?, FALSE)", (features_blob,))
            player_id = cursor.lastrowid
            conn.commit()
            
            face_info = {
                'player_id': player_id,
                'coordinates': (x, y, w, h),
                'has_played': False
            }
            detected_faces.append(face_info)
    
    conn.close()
    return detected_faces