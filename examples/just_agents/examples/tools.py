import json
from typing import Any, Dict


# ========================================
# 1. STATELESS TOOLS (Pure Functions)
# ========================================


def letter_count(word: str, letter: str) -> int:
    """ returns number of letters in the word """
    print(f"Calling letter_count('{word}', '{letter}')")
    word = word.lower()
    letter = letter.lower()
    count = 0
    for char in word:
        if letter == char:
            count += 1

    return count

def get_current_weather(location: str) -> str:
    """Gets the current weather in a given location.
    Args:
        location (str): The name of the location to get the weather for.
    Returns:
        str: A JSON string containing the location, temperature, and unit of measurement.
    """
    print("Function was actually called! with location: ", location, "")
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    elif "new york" in location.lower():
        return json.dumps({"location": "New York", "temperature": "68", "unit": "fahrenheit"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n: The position in the Fibonacci sequence (0-based).
        
    Returns:
        The nth Fibonacci number.
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def text_analyzer(text: str, analysis_type: str = "summary") -> Dict[str, Any]:
    """
    Analyze text and return various metrics.
    
    Args:
        text: The text to analyze.
        analysis_type: Type of analysis ('summary', 'detailed', 'stats').
        
    Returns:
        Dictionary containing analysis results.
    """
    words = text.split()
    sentences = text.split('.')
    characters = len(text)
    
    base_stats = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "character_count": characters,
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    
    if analysis_type == "summary":
        return {
            "type": "summary",
            "word_count": base_stats["word_count"],
            "sentence_count": base_stats["sentence_count"]
        }
    elif analysis_type == "detailed":
        return {
            "type": "detailed",
            **base_stats,
            "longest_word": max(words, key=len) if words else "",
            "shortest_word": min(words, key=len) if words else ""
        }
    else:  # stats
        return {
            "type": "statistical",
            **base_stats,
            "readability_score": min(100, max(0, 100 - base_stats["average_word_length"] * 5))
        }


# ========================================
# 2. STATIC METHODS (Class-based tools)
# ========================================

class MathUtilities:
    """Mathematical utility functions as static methods."""
    
    @staticmethod
    def calculate_circle_area(radius: float) -> float:
        """
        Calculate the area of a circle.
        
        Args:
            radius: The radius of the circle.
            
        Returns:
            The area of the circle.
        """
        import math
        return math.pi * radius ** 2
    
    @staticmethod
    def convert_temperature(temperature: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature between different units.
        
        Args:
            temperature: The temperature value to convert.
            from_unit: Source unit ('celsius', 'fahrenheit', 'kelvin').
            to_unit: Target unit ('celsius', 'fahrenheit', 'kelvin').
            
        Returns:
            The converted temperature.
        """
        # Convert to Celsius first
        if from_unit.lower() == 'fahrenheit':
            celsius = (temperature - 32) * 5/9
        elif from_unit.lower() == 'kelvin':
            celsius = temperature - 273.15
        else:  # already celsius
            celsius = temperature
        
        # Convert from Celsius to target
        if to_unit.lower() == 'fahrenheit':
            return celsius * 9/5 + 32
        elif to_unit.lower() == 'kelvin':
            return celsius + 273.15
        else:  # celsius
            return celsius

    class GeometryTools:
        """Nested class with geometric calculations."""
        
        @staticmethod
        def calculate_triangle_area(base: float, height: float) -> float:
            """
            Calculate the area of a triangle.
            
            Args:
                base: The base of the triangle.
                height: The height of the triangle.
                
            Returns:
                The area of the triangle.
            """
            return 0.5 * base * height
        
        @staticmethod
        def calculate_rectangle_perimeter(length: float, width: float) -> float:
            """
            Calculate the perimeter of a rectangle.
            
            Args:
                length: The length of the rectangle.
                width: The width of the rectangle.
                
            Returns:
                The perimeter of the rectangle.
            """
            return 2 * (length + width)


# ========================================
# 3. STATEFUL TOOLS (Instance methods - Transient Tools)
# ========================================

class DocumentProcessor:
    """A stateful document processor that maintains processing history."""
    
    def __init__(self, processor_name: str = "DefaultProcessor"):
        """Initialize the document processor."""
        self.processor_name = processor_name
        self.processed_documents = []
        self.total_words_processed = 0
        self.session_stats = {"documents": 0, "errors": 0}
    
    def process_document(self, document_text: str, document_name: str = "unnamed") -> Dict[str, Any]:
        """
        Process a document and track statistics.
        
        Args:
            document_text: The text content of the document.
            document_name: Optional name for the document.
            
        Returns:
            Processing results including statistics.
        """
        try:
            words = document_text.split()
            word_count = len(words)
            
            # Update state
            self.processed_documents.append({
                "name": document_name,
                "word_count": word_count,
                "processed_at": f"session_doc_{len(self.processed_documents) + 1}"
            })
            self.total_words_processed += word_count
            self.session_stats["documents"] += 1
            
            return {
                "document_name": document_name,
                "word_count": word_count,
                "processor": self.processor_name,
                "session_total_words": self.total_words_processed,
                "session_document_count": self.session_stats["documents"],
                "status": "success"
            }
        except Exception as e:
            self.session_stats["errors"] += 1
            return {
                "document_name": document_name,
                "error": str(e),
                "processor": self.processor_name,
                "status": "error"
            }
    
    def get_processing_history(self) -> Dict[str, Any]:
        """
        Get the complete processing history for this session.
        
        Returns:
            Complete session statistics and document history.
        """
        return {
            "processor_name": self.processor_name,
            "session_stats": self.session_stats,
            "total_words_processed": self.total_words_processed,
            "processed_documents": self.processed_documents,
            "documents_in_session": len(self.processed_documents)
        }
    
    def reset_session(self) -> str:
        """
        Reset the processing session.
        
        Returns:
            Confirmation message.
        """
        docs_processed = len(self.processed_documents)
        words_processed = self.total_words_processed
        
        self.processed_documents.clear()
        self.total_words_processed = 0
        self.session_stats = {"documents": 0, "errors": 0}
        
        return f"Session reset. Previously processed {docs_processed} documents with {words_processed} total words."


class TaskManager:
    """A stateful task manager for tracking todo items."""
    
    def __init__(self, manager_name: str = "DefaultManager"):
        """Initialize the task manager."""
        self.manager_name = manager_name
        self.tasks = {}
        self.next_task_id = 1
        self.completed_count = 0
    
    def add_task(self, title: str, description: str = "", priority: str = "medium") -> Dict[str, Any]:
        """
        Add a new task to the manager.
        
        Args:
            title: The task title.
            description: Optional task description.
            priority: Task priority ('low', 'medium', 'high').
            
        Returns:
            Details of the created task.
        """
        task_id = self.next_task_id
        self.next_task_id += 1
        
        task = {
            "id": task_id,
            "title": title,
            "description": description,
            "priority": priority,
            "status": "pending",
            "created_by": self.manager_name
        }
        
        self.tasks[task_id] = task
        return task
    
    def complete_task(self, task_id: int) -> Dict[str, Any]:
        """
        Mark a task as completed.
        
        Args:
            task_id: The ID of the task to complete.
            
        Returns:
            Updated task details or error message.
        """
        if task_id not in self.tasks:
            return {"error": f"Task {task_id} not found"}
        
        if self.tasks[task_id]["status"] == "completed":
            return {"error": f"Task {task_id} is already completed"}
        
        self.tasks[task_id]["status"] = "completed"
        self.completed_count += 1
        
        return {
            "task_id": task_id,
            "status": "completed",
            "total_completed": self.completed_count,
            "remaining_tasks": len([t for t in self.tasks.values() if t["status"] == "pending"])
        }
    
    def list_tasks(self, status_filter: str = "all") -> Dict[str, Any]:
        """
        List tasks based on status filter.
        
        Args:
            status_filter: Filter by status ('all', 'pending', 'completed').
            
        Returns:
            List of tasks matching the filter.
        """
        if status_filter == "all":
            filtered_tasks = list(self.tasks.values())
        else:
            filtered_tasks = [t for t in self.tasks.values() if t["status"] == status_filter]
        
        return {
            "manager": self.manager_name,
            "filter": status_filter,
            "task_count": len(filtered_tasks),
            "tasks": filtered_tasks,
            "total_completed": self.completed_count,
            "total_tasks": len(self.tasks)
        }
