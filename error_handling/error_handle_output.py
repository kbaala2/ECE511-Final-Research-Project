# user_processor.py

def process_user_record(user_record):
    """
    Expects a dictionary with:
      - "name": string
      - "scores": list of numbers

    Returns a summary dictionary.
    """
    if not isinstance(user_record, dict) or \
       "name" not in user_record or \
       "scores" not in user_record:
        raise ValueError("Invalid input: missing required fields")

    scores = user_record["scores"]
    if scores is None or (not isinstance(scores, list) and not isinstance(scores, tuple)):
        raise ValueError("Invalid input: 'scores' must be a list or tuple")
    elif len(scores) == 0:
        return {
            "name_upper": user_record["name"].upper(),
            "avg_score": 0,
            "is_passing": False
        }

    if not all(isinstance(score, (int, float)) for score in scores):
        raise ValueError("Invalid input: 'scores' must contain only numbers")

    avg_score = sum(scores) / len(scores)

    return {
        "name_upper": user_record["name"].upper(),
        "avg_score": avg_score,
        "is_passing": avg_score >= 70
    }


if __name__ == "__main__":
    # Intentionally buggy input for runtime-error testing
    sample_user = {
        "id": 101,
        "scores": None
    }

    try:
        result = process_user_record(sample_user)
        print(result)
    except ValueError as e:
        print(f"Error: {e}")