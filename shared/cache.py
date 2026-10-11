from threading import Lock


_cache = {}
_cache_lock = Lock()


def normalize_question(question: str) -> str:
    return " ".join(
        question.lower().strip().split()
    )


def get_cached_answer(question: str):
    key = normalize_question(question)

    with _cache_lock:
        return _cache.get(key)


def set_cached_answer(question: str, value):
    key = normalize_question(question)

    with _cache_lock:
        _cache[key] = value


def clear_cache():
    with _cache_lock:
        _cache.clear()