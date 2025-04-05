import sqlite3

class Database:
    def __init__(self, db_name='voice_notes.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_table()
        
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            notes TEXT,
            category TEXT
        )
        ''')
        self.conn.commit()
        
    def store_notes(self, name, notes, category):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO face_notes (name, notes, category)
        VALUES (?, ?, ?)
        ''', (name, notes, category))
        self.conn.commit()
    
    def get_id(self, name):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT id FROM face_notes WHERE name = ?
        ''', (name,))
        result = cursor.fetchone()
        return result if result else None 
        
    def get_notes(self, id):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT name, notes, category FROM face_notes WHERE id = ?
        ''', (id,))
        result = cursor.fetchone()
        return result if result else None
        
    def edit(self, id, name, notes, category):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE face_notes
            SET name = ?, notes = ?, category = ?
            WHERE id = ?
            """,
            (name, notes, category, id)
        )
        self.conn.commit()
        return True