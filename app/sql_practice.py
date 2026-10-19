
import sqlite3

from app.database import DB_PATH, initialize_database


def main():
    initialize_database()

    with sqlite3.connect(DB_PATH) as connection:

        # INSERT a request
        connection.execute(
            """
            INSERT INTO requests (question, answer, latency_ms)
            VALUES (?, ?, ?)
            """,
            (
                "What is PointNet?",
                "A neural network for point clouds.",
                1250.0,
            ),
        )

        # SELECT all requests
        cursor = connection.execute(
            """
            SELECT id, question, answer, latency_ms
            FROM requests
            """
        )

        for row in cursor.fetchall():
            print(row)


if __name__ == "__main__":
    main()
