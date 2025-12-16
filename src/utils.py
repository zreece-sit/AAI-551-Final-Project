from functools import reduce

def filter_genres(data, min_count):
    """
    Filters genres based on a minimum count.
    """
    return [g for g, c in data.items() if c >= min_count]

def total_count(counts):
    """
    Computes total count using reduce.
    """
    return reduce(lambda a, b: a + b, counts)

def genre_generator(genres):
    """
    Generator that yields genres one at a time.
    """
    for genre in genres:
        yield genre
