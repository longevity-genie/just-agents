import json

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
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
