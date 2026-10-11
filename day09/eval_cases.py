
EVAL_CASES = [
    {
        "id": "accuracy_pointnet_heavy",
        "question": (
            "What is PointNet classification accuracy under "
            "heavy noise on ModelNet40-E?"
        ),
        "expected_status": "answered",
        "expected_values": ["34.32", "1.46"],
        "expected_sources": [
            "ModelNet40E_p11_c1",
            "ModelNet40E_p24_c1",
        ],
    },
    {
        "id": "accuracy_dgcnn_heavy",
        "question": (
            "What is DGCNN classification accuracy under "
            "heavy noise on ModelNet40-E?"
        ),
        "expected_status": "answered",
        "expected_values": ["36.83", "4.06"],
        "expected_sources": [
            "ModelNet40E_p11_c1",
            "ModelNet40E_p24_c1",
        ],
    },
    {
        "id": "accuracy_pointnet_scanobjectnn",
        "question": (
            "What is PointNet classification accuracy under "
            "heavy noise on ScanObjectNN-E?"
        ),
        "expected_status": "answered",
        "expected_values": ["15.95", "2.10"],
        "expected_sources": ["ModelNet40E_p20_c1"],
    },
    {
        "id": "missing_latency",
        "question": (
            "What is PointNet's exact inference latency "
            "on an NVIDIA H100 GPU?"
        ),
        "expected_status": "insufficient_evidence",
        "expected_values": [],
        "expected_sources": [],
    },
]
