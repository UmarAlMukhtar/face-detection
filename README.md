# Event Face Recognition Game

This project implements a facial recognition system for a game stall at events. The system allows participants to play the game only once by recognizing their faces and checking their participation status against a database.

## Project Structure

```
event-face-recognition
├── src
│   ├── main.py                # Entry point of the application
│   ├── face_detector.py        # Functions for capturing and processing facial images
│   ├── database.py             # Handles database operations
│   └── display_manager.py      # Manages display output and UI elements
├── data
│   └── players_database.db     # SQLite database for storing player information
├── requirements.txt            # Lists project dependencies
├── .gitignore                  # Specifies files to ignore in version control
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**

   ```
   git clone <repository-url>
   cd event-face-recognition
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:

   ```
   pip install -r requirements.txt
   ```

3. **Database Initialization:**
   The `players_database.db` file will be created automatically when the application runs for the first time. The system will create the necessary directory structure.

## Usage Guidelines

1. **Run the application:**
   Execute the following command to start the game:

   ```
   python src/main.py
   ```

2. **Gameplay:**

   - The system will capture the player's face and check if they have played before.
   - New players must remain in front of the camera for 10 seconds to complete registration.
   - The system shows a progress bar indicating registration status.
   - If the player is recognized and has already participated, a message will inform them they've already played.
   - If the player is new, after successful registration they will be allowed to play.

3. **Controls:**
   - Press 'q' to quit the application

## Features

- **Face detection** using OpenCV's Haar Cascade classifier
- **Face recognition** using histogram-based feature extraction and matching
- **10-second registration process** to ensure intentional participation
- **Visual feedback** with progress bar during registration
- **Local database** for storing player information
- **Privacy-conscious** by not storing actual face images

## Dependencies

- `opencv-python`: For image processing, face detection, and capturing video
- `numpy`: For numerical operations and feature comparison
- `sqlite3`: For database operations (included in Python standard library)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
