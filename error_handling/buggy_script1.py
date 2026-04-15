# buggy_user_processor.py

def process_user_record(user_record):
    """
    Expects a dictionary with:
      - "name": string
      - "scores": list of numbers

    Returns a summary dictionary.
    """
    avg_score = sum(user_record["scores"]) / len(user_record["scores"])

    return {
        "name_upper": user_record["name"].upper(),
        "avg_score": avg_score,
        "is_passing": avg_score >= 70,
    }


if __name__ == "__main__":
    # Intentionally buggy input for runtime-error testing
    sample_user = {
        "id": 101,
        "scores": None
    }

    result = process_user_record(sample_user)
    print(result)