#!/usr/bin/env python3

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cursor = conn.cursor()

print('Checking all tables and their text columns:')
print('=' * 60)

# Get all tables
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]
    
    # Get row count
    try:
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
        count = cursor.fetchone()[0]
    except Exception as e:
        count = f"Error: {e}"
    
    print(f'\n📋 Table: {table_name}')
    print(f'   Rows: {count}')
    
    # Get all columns and their data types
    cursor.execute("""
        SELECT column_name, data_type, character_maximum_length, is_nullable
        FROM information_schema.columns 
        WHERE table_name = %s AND table_schema = 'public'
        ORDER BY ordinal_position;
    """, (table_name,))
    
    columns = cursor.fetchall()
    
    if columns:
        print('   📝 Columns:')
        text_columns = []
        
        for col_name, data_type, max_length, nullable in columns:
            # Identify text-like columns
            is_text_column = data_type.lower() in ['text', 'varchar', 'character varying', 'char', 'character']
            
            if is_text_column:
                text_columns.append(col_name)
                marker = ' 📄 TEXT'
            else:
                marker = ''
            
            length_info = f'({max_length})' if max_length else ''
            nullable_info = 'NULL' if nullable == 'YES' else 'NOT NULL'
            
            print(f'      - {col_name}: {data_type}{length_info} {nullable_info}{marker}')
        
        # Summary of text columns
        if text_columns:
            print(f'   🎯 Text Columns: {", ".join(text_columns)}')
        else:
            print('   ⚠️  No text columns found')
            
        # For specific tables, show sample data from text columns
        if any(keyword in table_name.lower() for keyword in ['llamaindex', 'document', 'embedding']) and text_columns:
            print('   📊 Sample text data:')
            for text_col in text_columns[:2]:  # Show max 2 text columns
                try:
                    cursor.execute(f'SELECT "{text_col}" FROM "{table_name}" WHERE "{text_col}" IS NOT NULL LIMIT 1;')
                    sample = cursor.fetchone()
                    if sample and sample[0]:
                        sample_text = str(sample[0])[:100] + '...' if len(str(sample[0])) > 100 else str(sample[0])
                        print(f'      {text_col}: "{sample_text}"')
                except Exception as e:
                    print(f'      {text_col}: Error reading sample - {e}')
    else:
        print('   ⚠️  No columns found')

print('\n' + '=' * 60)
print('🔍 Summary - Tables with Text Content:')

# Re-run to show summary
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]
    
    # Get text columns
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %s AND table_schema = 'public' 
        AND data_type IN ('text', 'varchar', 'character varying', 'char', 'character')
        ORDER BY ordinal_position;
    """, (table_name,))
    
    text_columns = [row[0] for row in cursor.fetchall()]
    
    if text_columns:
        print(f'  📋 {table_name}: {", ".join(text_columns)}')

cursor.close()
conn.close()
