
from app.database import (
    initialize_database,
    save_request,
    get_requests,
    get_slow_requests,
)

initialize_database()

request_id = save_request(
    question="What is DGCNN?",
    answer="A graph-based neural network for point clouds.",
    latency_ms=1800.0,
)

print("Saved request ID:", request_id)

print("\nAll requests:")
for row in get_requests():
    print(row)

print("\nSlow requests:")
for row in get_slow_requests(1500):
    print(row)
