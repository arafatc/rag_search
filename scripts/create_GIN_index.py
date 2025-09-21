#!/usr/bin/env python3

import psycopg2
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def create_indexes():
    """Create GIN indexes for enhanced LlamaIndex tables"""
    
    # SQL commands to create GIN indexes
    index_commands = [
        # Table: data_data_llamaindex_enhanced_hierarchical
        """
        CREATE INDEX IF NOT EXISTS idx_hierarchical_text_gin 
        ON data_data_llamaindex_enhanced_hierarchical
        USING GIN (to_tsvector('english', text));
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_hierarchical_node_id_gin 
        ON data_data_llamaindex_enhanced_hierarchical
        USING GIN (to_tsvector('english', node_id));
        """,
        
        # Table: data_llamaindex_enhanced_semantic
        """
        CREATE INDEX IF NOT EXISTS idx_semantic_text_gin 
        ON data_llamaindex_enhanced_semantic
        USING GIN (to_tsvector('english', text));
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_semantic_node_id_gin 
        ON data_llamaindex_enhanced_semantic
        USING GIN (to_tsvector('english', node_id));
        """,
        
        # Table: data_llamaindex_enhanced_structure_aware
        """
        CREATE INDEX IF NOT EXISTS idx_structure_text_gin 
        ON data_llamaindex_enhanced_structure_aware
        USING GIN (to_tsvector('english', text));
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_structure_node_id_gin 
        ON data_llamaindex_enhanced_structure_aware
        USING GIN (to_tsvector('english', node_id));
        """
    ]
    
    try:
        # Connect to database
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print(" Error: DATABASE_URL not found in environment variables")
            sys.exit(1)
            
        print(" Connecting to database...")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        print(" Creating GIN indexes for enhanced LlamaIndex tables...")
        print("=" * 70)
        
        # Execute each index creation command
        for i, command in enumerate(index_commands, 1):
            try:
                # Extract index name from command for logging
                lines = [line.strip() for line in command.strip().split('\n') if line.strip()]
                index_line = next((line for line in lines if 'CREATE INDEX' in line), '')
                index_name = index_line.split('IF NOT EXISTS')[1].split('ON')[0].strip() if 'IF NOT EXISTS' in index_line else 'Unknown'
                
                print(f" Creating index {i}/6: {index_name}")
                cursor.execute(command)
                conn.commit()
                print(f" Success")
                
            except psycopg2.Error as e:
                print(f" Failed: {e}")
                conn.rollback()
                continue
        
        print("\n" + "=" * 70)
        print(" Verifying created indexes...")
        
        # Verify indexes were created
        cursor.execute("""
            SELECT schemaname, tablename, indexname, indexdef 
            FROM pg_indexes 
            WHERE indexname LIKE '%_gin' 
            AND tablename LIKE '%llamaindex_enhanced%'
            ORDER BY tablename, indexname;
        """)
        
        indexes = cursor.fetchall()
        
        if indexes:
            print(f" Found {len(indexes)} GIN indexes:")
            for schema, table, index_name, index_def in indexes:
                print(f"   {table}: {index_name}")
        else:
            print("  No GIN indexes found (they may not have been created)")
            
        print("\n" + "=" * 70)
        print(" Index size information:")

        # Get index sizes
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                pg_size_pretty(pg_relation_size(indexname::regclass)) as size
            FROM pg_indexes 
            WHERE indexname LIKE '%_gin' 
            AND tablename LIKE '%llamaindex_enhanced%'
            ORDER BY pg_relation_size(indexname::regclass) DESC;
        """)
        
        index_sizes = cursor.fetchall()
        
        for schema, table, index_name, size in index_sizes:
            print(f"   {index_name}: {size}")
            
        cursor.close()
        conn.close()
        
        print("\n Index creation completed successfully!")
        print("\n Benefits:")
        print("   - Faster full-text search queries using ts_rank_cd")
        print("   - Improved performance for websearch_to_tsquery operations")
        print("   - Enhanced lexical retrieval speed in hybrid search")
        
    except psycopg2.Error as e:
        print(f" Database error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f" Unexpected error: {e}")
        sys.exit(1)

def check_table_existence():
    """Check if the target tables exist before creating indexes"""
    try:
        database_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Check which tables exist
        tables_to_check = [
            'data_data_llamaindex_enhanced_hierarchical',
            'data_llamaindex_enhanced_semantic', 
            'data_llamaindex_enhanced_structure_aware'
        ]
        
        print(" Checking table existence...")
        existing_tables = []
        
        for table in tables_to_check:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            
            exists = cursor.fetchone()[0]
            if exists:
                existing_tables.append(table)
                print(f"  {table}")
            else:
                print(f"  {table} (not found)")
        
        cursor.close()
        conn.close()
        
        if not existing_tables:
            print("\n  No target tables found. Please run data ingestion first.")
            return False
            
        print(f"\n Found {len(existing_tables)}/{len(tables_to_check)} tables")
        return True
        
    except Exception as e:
        print(f" Error checking tables: {e}")
        return False

if __name__ == "__main__":
    print(" GIN Index Creator for Enhanced LlamaIndex Tables")
    print("=" * 70)
    
    # Check if tables exist first
    if check_table_existence():
        create_indexes()
    else:
        sys.exit(1)