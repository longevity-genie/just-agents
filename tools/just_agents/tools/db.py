from typing import Dict, List, Union, Optional
from eliot import start_action, Message
from pathlib import Path
import sqlite3

def sqlite_query(database: Union[Path, str], sql: str) -> str:
    """Execute a SQL query and return results as formatted text.

    Args:
        database (Union[Path, str]): Path to the SQLite database file
        sql (str): SQL query to execute

    Returns:
        str: Query results formatted as semicolon-separated text.
             First line contains column names, followed by rows of data.
             Returns empty string if no results found.

    Raises:
        sqlite3.Error: If there's any database-related error
    """
    with start_action(action_type="sqlite_query", database=str(database), sql=sql) as action:
        try:
            # Convert string path to Path object if needed
            db_path = Path(database).absolute() if isinstance(database, str) else database

            # Use context manager for database connection
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                # Fetch and format query results
                rows = cursor.fetchall()
                if not rows:
                    action.log(message_type="query_result", status="empty")
                    return ""

                # Extract column names from cursor description
                names = [description[0] for description in cursor.description]
                
                # Format output with column names as first line
                text = "; ".join(names) + "\n"
                
                # Add data rows, converting all values to strings
                text += "\n".join("; ".join(str(i) for i in row) for row in rows) + "\n"
                
                action.log(message_type="query_result", status="success", row_count=len(rows))
                return text

        except sqlite3.Error as e:
            action.log(message_type="query_error", error=str(e))
            raise

def sqlite_query_read_only(database: Union[Path, str], sql: str) -> str:
    """Execute a read-only SQL query and return results as formatted text.
    Prevents any database modifications.

    Args:
        database (Union[Path, str]): Path to the SQLite database file
        sql (str): SQL query to execute (SELECT queries only)

    Returns:
        str: Query results formatted as semicolon-separated text.
             First line contains column names, followed by rows of data.
             Returns empty string if no results found.

    Raises:
        ValueError: If the query contains potential modification statements
        sqlite3.Error: If there's any database-related error
    """
    with start_action(action_type="sqlite_query_read_only", database=str(database), sql=sql) as action:
        # Check for potentially dangerous operations
        sql_lower = sql.lower().strip()
        forbidden_keywords = [
            'insert', 'update', 'delete', 'drop', 'alter', 'create', 
            'replace', 'truncate', 'attach', 'detach', 'vacuum'
        ]
        
        if not sql_lower.startswith('select'):
            raise ValueError("Only SELECT queries are allowed in read-only mode")
            
        for keyword in forbidden_keywords:
            if keyword in sql_lower:
                raise ValueError(f"Forbidden keyword '{keyword}' detected in query")

        try:
            db_path = Path(database).absolute() if isinstance(database, str) else database

            # Use URI format to enable strict read-only mode
            uri = f"file:{db_path}?mode=ro&immutable=1&nolock=1"
            
            with sqlite3.connect(uri, uri=True) as conn:
                # Set additional safety restrictions
                conn.execute("PRAGMA query_only = ON;")
                conn.execute("PRAGMA temp_store = MEMORY;")
                
                cursor = conn.cursor()
                cursor.execute(sql)
                
                # Rest of the function is same as sqlite_query
                rows = cursor.fetchall()
                if not rows:
                    action.log(message_type="query_result", status="empty")
                    return ""

                names = [description[0] for description in cursor.description]
                text = "; ".join(names) + "\n"
                text += "\n".join("; ".join(str(i) for i in row) for row in rows) + "\n"
                
                action.log(message_type="query_result", status="success", row_count=len(rows))
                return text

        except sqlite3.Error as e:
            action.log(message_type="query_error", error=str(e))
            raise

def extract_db_structure(db_path: Union[Path, str]) -> Dict[str, List[Dict]]:
    """
    Extracts the structure (table names and column definitions) of all tables in the database.

    Args:
        db_path (Union[Path, str]): The path to the SQLite database file.

    Returns:
        Dict[str, List[Dict]]: A dictionary where keys are table names and values are lists of
                              dictionaries containing column information.

    Raises:
        FileNotFoundError: If the database file does not exist.
        ValueError: If the database file has an unsupported extension.
        sqlite3.Error: If there's any database-related error
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        raise FileNotFoundError(f"The database file '{db_path}' does not exist.")
    if db_path.suffix not in {'.sqlite', '.db'}:
        raise ValueError(f"Unsupported database file extension: '{db_path.suffix}'. Expected '.sqlite' or '.db'.")
    
    with start_action(action_type="extract_db_structure", database=str(db_path)) as action:
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                try:
                    # Query to get all table names in the database
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    # Loop through the tables and get their schema
                    db_structure = {}
                    for (table_name,) in tables:
                        # Get basic column info
                        cursor.execute(f"PRAGMA table_info('{table_name}');")
                        columns = cursor.fetchall()
                        
                        # Get foreign key information
                        cursor.execute(f"PRAGMA foreign_key_list('{table_name}');")
                        fk_data = cursor.fetchall()
                        
                        # Get index information
                        cursor.execute(f"PRAGMA index_list('{table_name}');")
                        indexes = cursor.fetchall()
                        
                        # Create a mapping of column name to its indexes
                        column_indexes = {}
                        for idx in indexes:
                            index_name = idx[1]
                            unique = bool(idx[2])
                            cursor.execute(f"PRAGMA index_info('{index_name}');")
                            idx_info = cursor.fetchall()
                            for col_info in idx_info:
                                col_name = col_info[2]
                                column_indexes.setdefault(col_name, []).append({
                                    'index_name': index_name,
                                    'unique': unique
                                })
                        
                        # Create foreign key mapping
                        fk_mapping = {}
                        for fk in fk_data:
                            fk_column = fk[3]
                            fk_mapping[fk_column] = {
                                'table': fk[2],
                                'to_column': fk[4],
                                'on_update': fk[5],
                                'on_delete': fk[6]
                            }
                        
                        column_info = []
                        for column in columns:
                            try:
                                col = {
                                    'cid': column[0],
                                    'name': column[1],
                                    'type': column[2],
                                    'notnull': bool(column[3]),
                                    'dflt_value': column[4],
                                    'pk': bool(column[5]),
                                    'foreign_key': fk_mapping.get(column[1]),
                                    'indexes': column_indexes.get(column[1], []),
                                    'hidden': False  # SQLite 3.35.0+ supports hidden columns
                                }
                                column_info.append(col)
                            except IndexError as e:
                                action.log(message_type="column_parse_error", column=column, error=str(e))
                                continue  # Skip malformed column entries
                        
                        db_structure[table_name] = column_info

                    action.log(message_type="structure_extracted", table_count=len(tables))
                    return db_structure

                finally:
                    cursor.close()

        except sqlite3.Error as e:
            action.log(message_type="structure_error", error=str(e))
            raise
        except Exception as e:
            action.log(message_type="unexpected_error", error=str(e))
            raise

