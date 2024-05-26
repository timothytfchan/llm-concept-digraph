import sqlite3
import os
import logging

def create_tables(results_db_path):
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            model_name TEXT,
            persona TEXT,
            temperature REAL,
            variable_a TEXT,
            variable_b TEXT,
            response TEXT,
            choice TEXT
        )
    ''')
    conn.commit()
    conn.close()

def record_exists(results_db_path, model_name, persona, temperature, variable_a, variable_b):
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 1 FROM experiments
        WHERE model_name = ? AND persona = ? AND temperature = ? AND variable_a = ? AND variable_b = ?
    ''', (model_name, persona, temperature, variable_a, variable_b))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def insert_record(results_db_path, model_name, persona, temperature, variable_a, variable_b, response, choice):
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO experiments (model_name, persona, temperature, variable_a, variable_b, response, choice)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (model_name, persona, temperature, variable_a, variable_b, response, choice))
    conn.commit()
    conn.close()

def get_all_records(results_db_path):
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM experiments')
    records = cursor.fetchall()
    conn.close()
    return records

def generate_pairs(variables):
    comparisons = [(a, b) for i, a in enumerate(variables) for j, b in enumerate(variables) if i != j]
    return comparisons

def log_message(message, level='INFO'):
    logging.basicConfig(filename='experiment.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if level == 'INFO':
        logging.info(message)
    elif level == 'WARNING':
        logging.warning(message)
    elif level == 'ERROR':
        logging.error(message)
