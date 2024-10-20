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