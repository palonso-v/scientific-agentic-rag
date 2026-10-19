import os

import psycopg
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is not configured."
        )

    return psycopg.connect(DATABASE_URL)


def initialize_database():
    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    latency_ms DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    cache_hit BOOLEAN DEFAULT FALSE,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    estimated_cost_usd DOUBLE PRECISION DEFAULT 0
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    request_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating IN (0, 1)),
                    comment TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (request_id)
                        REFERENCES requests(id)
                )
                """
            )


def save_request(
    question: str,
    answer: str,
    latency_ms: float,
    cache_hit: bool = False,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
) -> int:

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO requests (
                    question,
                    answer,
                    latency_ms,
                    cache_hit,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    question,
                    answer,
                    latency_ms,
                    cache_hit,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd,
                ),
            )

            row = cursor.fetchone()

            if row is None:
                raise RuntimeError(
                    "Database did not return a request ID."
                )

            return row[0]


def get_requests():
    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT id, question, answer, latency_ms, created_at
                FROM requests
                ORDER BY id DESC
                """
            )

            return cursor.fetchall()


def get_slow_requests(minimum_latency: float):
    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT question, latency_ms
                FROM requests
                WHERE latency_ms > %s
                ORDER BY latency_ms DESC
                """,
                (minimum_latency,),
            )

            return cursor.fetchall()


def update_request_answer(
    request_id: int,
    new_answer: str,
) -> bool:

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                UPDATE requests
                SET answer = %s
                WHERE id = %s
                """,
                (new_answer, request_id),
            )

            return cursor.rowcount > 0


def delete_request(request_id: int) -> bool:

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                DELETE FROM requests
                WHERE id = %s
                """,
                (request_id,),
            )

            return cursor.rowcount > 0


def save_feedback(
    request_id: int,
    rating: int,
    comment: str | None = None,
) -> int:

    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO feedback (
                    request_id,
                    rating,
                    comment
                )
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (
                    request_id,
                    rating,
                    comment,
                ),
            )

            row = cursor.fetchone()

            if row is None:
                raise RuntimeError(
                    "Database did not return a feedback ID."
                )

            return row[0]


def get_negative_feedback():
    with get_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    requests.question,
                    requests.answer,
                    feedback.rating,
                    feedback.comment
                FROM feedback
                JOIN requests
                    ON feedback.request_id = requests.id
                WHERE feedback.rating = 0
                ORDER BY feedback.id DESC
                """
            )

            return cursor.fetchall()


if __name__ == "__main__":
    initialize_database()
    print("PostgreSQL database initialized.")