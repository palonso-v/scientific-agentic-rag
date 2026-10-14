
import sqlite3
from pathlib import Path
from contextlib import closing

DB_PATH = Path("data") / "assistant.db"



def initialize_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        # Step 1: Create the table if it doesn't exist
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                latency_ms REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating IN (0, 1)),
                comment TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (request_id) REFERENCES requests(id)
            )
            """
        )

        # Step 2: Check which columns already exist
        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(requests)"
            ).fetchall()
        }

        # Step 3: Define the new columns
        new_columns = {
            "cache_hit": "INTEGER DEFAULT 0",
            "input_tokens": "INTEGER DEFAULT 0",
            "output_tokens": "INTEGER DEFAULT 0",
            "estimated_cost_usd": "REAL DEFAULT 0",
        }

        # Step 4: Add only the missing columns
        for column_name, column_type in new_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE requests "
                    f"ADD COLUMN {column_name} {column_type}"
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

    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            cursor = connection.execute(
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
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    answer,
                    latency_ms,
                    int(cache_hit),
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd,
                ),
            )

            return cursor.lastrowid


def get_requests():
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            SELECT id, question, answer, latency_ms, created_at
            FROM requests
            ORDER BY id DESC
            """
        )
        return cursor.fetchall()


def get_slow_requests(minimum_latency: float):
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            """
            SELECT question, latency_ms
            FROM requests
            WHERE latency_ms > ?
            ORDER BY latency_ms DESC
            """,
            (minimum_latency,),
        )
        return cursor.fetchall()


def update_request_answer(request_id: int, new_answer: str) -> bool:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            cursor = connection.execute(
                """
                UPDATE requests
                SET answer = ?
                WHERE id = ?
                """,
                (new_answer, request_id),
            )

            return cursor.rowcount > 0


def delete_request(request_id: int) -> bool:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            cursor = connection.execute(
                """
                DELETE FROM requests
                WHERE id = ?
                """,
                (request_id,),
            )

            return cursor.rowcount > 0



def save_feedback(
    request_id: int,
    rating: int,
    comment: str | None = None,
) -> int:

    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        with connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback (
                    request_id,
                    rating,
                    comment
                )
                VALUES (?, ?, ?)
                """,
                (request_id, rating, comment),
            )

            return cursor.lastrowid


def get_negative_feedback():
    with closing(sqlite3.connect(DB_PATH)) as connection:
        cursor = connection.execute(
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
    print(f"Database initialized at {DB_PATH}")
