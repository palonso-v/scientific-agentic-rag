
import sqlite3
from contextlib import closing

from app import database


def get_usage_summary():
    database.initialize_database()

    with closing(sqlite3.connect(database.DB_PATH)) as connection:
        cursor = connection.execute(
            """
            SELECT
                COUNT(*) AS total_requests,
                ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
                SUM(input_tokens) AS total_input_tokens,
                SUM(output_tokens) AS total_output_tokens,
                ROUND(SUM(estimated_cost_usd), 6) AS total_cost_usd
            FROM requests
            """
        )

        return cursor.fetchone()


def get_cache_statistics():
    database.initialize_database()

    with closing(sqlite3.connect(database.DB_PATH)) as connection:
        cursor = connection.execute(
            """
            SELECT
                cache_hit,
                COUNT(*) AS request_count,
                ROUND(AVG(latency_ms), 2) AS avg_latency_ms
            FROM requests
            GROUP BY cache_hit
            ORDER BY cache_hit
            """
        )

        return cursor.fetchall()


if __name__ == "__main__":
    summary = get_usage_summary()

    print("\n=== LLM Usage Summary ===")
    print("Total requests:", summary[0])
    print("Average latency (ms):", summary[1])
    print("Input tokens:", summary[2])
    print("Output tokens:", summary[3])
    print("Estimated cost (USD):", summary[4])

    print("\n=== Cache Statistics ===")

    for cache_hit, count, avg_latency in get_cache_statistics():
        label = "Cache hit" if cache_hit else "Cache miss"
        print(f"{label}: {count} requests, avg latency {avg_latency} ms")
