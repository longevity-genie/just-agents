import sqlite3
from eliot import start_action
from pathlib import Path
def sqlite_query(database: str, sql: str) -> str:
    """Execute a SQL query and return results as formatted text.

    Args:
        database (Path | str): Path to the SQLite database file
        sql (str): SQL query to execute

    Returns:
        str: Query results formatted as semicolon-separated text.
             First line contains column names, followed by rows of data.
             Returns empty string if no results found.
    """
    # Log the query execution using eliot
    start_action(action_type="sqlite_query", database=database, sql=sql)

    # Convert string path to Path object if needed
    if isinstance(database, str):
        database = Path(database).absolute()

    # Establish database connection
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute(sql)
    try:
        # Fetch and format query results
        rows = cursor.fetchall()
        if rows is None or len(rows) == 0:
            conn.close()
            return ""

        # Extract column names from cursor description
        names = [description[0] for description in cursor.description]
        
        # Format output with column names as first line
        text = "; ".join(names) + "\n"
        
        # Add data rows, converting all values to strings
        for row in rows:
            row = [str(i) for i in row]
            text += "; ".join(row) + "\n"
    finally:
        # Ensure connection is closed even if an error occurs
        conn.close()

    return text

