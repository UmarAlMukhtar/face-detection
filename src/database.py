import sqlite3
import os

def connect_db():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/players_database.db')
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
    return conn

def add_player(player_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE players SET played = TRUE WHERE id = ?", (player_id,))
    conn.commit()
    conn.close()

def check_player(player_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT played FROM players WHERE id = ?", (player_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None and result[0]