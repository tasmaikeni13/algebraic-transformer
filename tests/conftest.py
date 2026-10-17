import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "tier1: Acceptance tests for Tier 1 (Cleanliness & Architecture Decoupling)")
    config.addinivalue_line("markers", "tier2: Acceptance tests for Tier 2 (Tokenizer & Architecture)")
    config.addinivalue_line("markers", "tier3: Acceptance tests for Tier 3 (Cross-Module & Kernels)")
    config.addinivalue_line("markers", "tier4: Acceptance tests for Tier 4 (Distributed Multi-Worker Smoke Tests & Audit)")
    config.addinivalue_line("markers", "e2e: End-to-end publication acceptance tests")

