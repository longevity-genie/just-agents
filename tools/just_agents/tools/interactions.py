def get_user_input(prompt: str) -> str:
    """
    Get user input that works both in console and Jupyter notebooks.
    
    Args:
        prompt (str): The prompt to display to the user
        
    Returns:
        str: The user's input
    """
    try:
        # Try to import IPython for Jupyter environment
        from IPython.display import display
        from IPython.core.display import HTML
        # If we're in Jupyter, we can use input() directly as it's already patched
        return input(prompt)
    except ImportError:
        # If we're in console, use regular input
        return input(prompt)