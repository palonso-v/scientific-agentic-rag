RETRIEVAL_CASES = [
    # --------------------------------------------------
    # Lexical / identifier-heavy cases
    # --------------------------------------------------

    {
        "id": "ax41_measurement",
        "question": (
            "Which instrument reported an aerosol optical "
            "depth of exactly 0.18 at 550 nanometers?"
        ),
        "expected_sources": [
            "retrieval-eval_p1_c1",
        ],
    },
    {
        "id": "ax41_not_calibration",
        "question": (
            "What was the aerosol optical depth measured "
            "by AX-41 during the atmospheric campaign?"
        ),
        "expected_sources": [
            "retrieval-eval_p1_c1",
        ],
    },
    {
        "id": "zk91_r17",
        "question": (
            "Which catalyst substantially reduced the "
            "activation-energy barrier of reaction R17?"
        ),
        "expected_sources": [
            "retrieval-eval_p25_c1",
        ],
    },
    {
        "id": "modelnet40e_exact",
        "question": (
            "Which benchmark called ModelNet40-E evaluates "
            "calibration and robustness under LiDAR-inspired "
            "corruptions?"
        ),
        "expected_sources": [
            "retrieval-eval_p13_c1",
        ],
    },

    # --------------------------------------------------
    # Confusable point-cloud cases
    # --------------------------------------------------

    {
        "id": "dgcnn_dynamic_graph",
        "question": (
            "Which point-cloud architecture repeatedly "
            "reconstructs its neighborhood graph in feature "
            "space while using EdgeConv?"
        ),
        "expected_sources": [
            "retrieval-eval_p6_c1",
        ],
    },
    {
        "id": "pointnet_global_pool",
        "question": (
            "Which point-cloud network processes points "
            "independently before using a symmetric pooling "
            "operation to form one global representation?"
        ),
        "expected_sources": [
            "retrieval-eval_p8_c1",
        ],
    },
    {
        "id": "pointnetpp_hierarchy",
        "question": (
            "Which point-cloud architecture builds features "
            "over progressively larger local regions through "
            "sampling and grouping?"
        ),
        "expected_sources": [
            "retrieval-eval_p7_c1",
        ],
    },
    {
        "id": "geocd_intrinsic",
        "question": (
            "Which point-set comparison method attempts to "
            "follow intrinsic surface structure instead of "
            "using only straight-line Euclidean proximity?"
        ),
        "expected_sources": [
            "retrieval-eval_p10_c1",
        ],
    },
    {
        "id": "chamfer_euclidean",
        "question": (
            "Which point-cloud distance finds nearest "
            "neighbors in both directions but operates in "
            "ambient Euclidean space?"
        ),
        "expected_sources": [
            "retrieval-eval_p11_c1",
        ],
    },

    # --------------------------------------------------
    # Confusable biology cases
    # --------------------------------------------------

    {
        "id": "crispr_double_break",
        "question": (
            "Which genome-editing system uses a guide RNA "
            "to position a nuclease that cuts the target DNA?"
        ),
        "expected_sources": [
            "retrieval-eval_p16_c1",
        ],
    },
    {
        "id": "base_editing",
        "question": (
            "Which CRISPR-derived technique can alter a "
            "selected nucleotide without making the usual "
            "double-strand DNA break?"
        ),
        "expected_sources": [
            "retrieval-eval_p17_c1",
        ],
    },
    {
        "id": "prime_editing",
        "question": (
            "Which genome-engineering approach combines "
            "modified Cas9 with reverse transcriptase?"
        ),
        "expected_sources": [
            "retrieval-eval_p18_c1",
        ],
    },
    {
        "id": "covid_receptor",
        "question": (
            "Which viral surface protein interacts with "
            "ACE2 to enable host-cell entry?"
        ),
        "expected_sources": [
            "retrieval-eval_p47_c1",
        ],
    },

    # --------------------------------------------------
    # JWST instrument confusion
    # --------------------------------------------------

    {
        "id": "nircam_imaging",
        "question": (
            "Which James Webb instrument is primarily a "
            "near-infrared camera rather than a spectrograph?"
        ),
        "expected_sources": [
            "retrieval-eval_p19_c1",
        ],
    },
    {
        "id": "nirspec_spectra",
        "question": (
            "Which JWST instrument disperses near-infrared "
            "light into spectra instead of mainly taking "
            "direct images?"
        ),
        "expected_sources": [
            "retrieval-eval_p20_c1",
        ],
    },
    {
        "id": "miri_longer_wavelengths",
        "question": (
            "Which Webb instrument operates at longer "
            "infrared wavelengths than NIRCam and NIRSpec?"
        ),
        "expected_sources": [
            "retrieval-eval_p21_c1",
        ],
    },

    # --------------------------------------------------
    # Semantic-heavy astronomy cases
    # --------------------------------------------------

    {
        "id": "transit_semantic",
        "question": (
            "Which planet-search technique looks for a "
            "repeating loss of stellar light when an "
            "orbiting body moves across the star?"
        ),
        "expected_sources": [
            "retrieval-eval_p22_c1",
        ],
    },
    {
        "id": "radial_velocity_semantic",
        "question": (
            "Which exoplanet technique infers an unseen "
            "companion from periodic motion of its host star "
            "toward and away from the observer?"
        ),
        "expected_sources": [
            "retrieval-eval_p23_c1",
        ],
    },
    {
        "id": "supernova_standard_candle",
        "question": (
            "Which stellar explosions can be calibrated "
            "to infer cosmological distances from their "
            "apparent brightness?"
        ),
        "expected_sources": [
            "retrieval-eval_p38_c1",
        ],
    },
    {
        "id": "bao_standard_ruler",
        "question": (
            "Which cosmological observable provides a "
            "preferred clustering scale that acts as a "
            "standard ruler?"
        ),
        "expected_sources": [
            "retrieval-eval_p40_c1",
        ],
    },
    {
        "id": "ligo_interferometer",
        "question": (
            "Which gravitational-wave experiment detects "
            "tiny relative changes in long perpendicular "
            "laser paths?"
        ),
        "expected_sources": [
            "retrieval-eval_p41_c1",
        ],
    },

    # --------------------------------------------------
    # Retrieval-method cases
    # --------------------------------------------------

    {
        "id": "bm25_lexical",
        "question": (
            "Which retrieval algorithm uses inverse document "
            "frequency together with term-frequency saturation "
            "and length normalization?"
        ),
        "expected_sources": [
            "retrieval-eval_p35_c1",
        ],
    },
    {
        "id": "rrf_fusion",
        "question": (
            "Which method merges multiple ranked retrieval "
            "lists using rank positions rather than requiring "
            "their raw scores to be comparable?"
        ),
        "expected_sources": [
            "retrieval-eval_p36_c1",
        ],
    },
    {
        "id": "dense_semantics",
        "question": (
            "Which retrieval approach can find passages with "
            "similar meaning even when the wording of the "
            "query and document differs substantially?"
        ),
        "expected_sources": [
            "retrieval-eval_p37_c1",
        ],
    },

    # --------------------------------------------------
    # ML concept confusion
    # --------------------------------------------------

    {
        "id": "bert_masked_bidirectional",
        "question": (
            "Which transformer model learns from hidden "
            "tokens using context on both sides of them?"
        ),
        "expected_sources": [
            "retrieval-eval_p31_c1",
        ],
    },
    {
        "id": "adam_moments",
        "question": (
            "Which optimizer adapts parameter updates using "
            "running estimates of both the first and second "
            "moments of the gradient?"
        ),
        "expected_sources": [
            "retrieval-eval_p49_c1",
        ],
    },
    {
        "id": "dropout_regularization",
        "question": (
            "Which regularization method randomly suppresses "
            "neural activations during training?"
        ),
        "expected_sources": [
            "retrieval-eval_p51_c1",
        ],
    },

    # --------------------------------------------------
    # Climate / science cases
    # --------------------------------------------------

    {
        "id": "co2_year_values",
        "question": (
            "What source contains the synthetic carbon dioxide "
            "measurements of 400 ppm in 2015 and 420 ppm in 2025?"
        ),
        "expected_sources": [
            "retrieval-eval_p46_c1",
        ],
    },
    {
        "id": "photosynthesis_semantic",
        "question": (
            "Which process stores incoming light energy in "
            "newly synthesized organic molecules?"
        ),
        "expected_sources": [
            "retrieval-eval_p43_c1",
        ],
    },
]