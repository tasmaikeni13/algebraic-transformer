import json

from scripts.run_verify_primitives import CPU_MEASUREMENT_FILES, hardware_evidence, reusable_cpu_evidence


def test_missing_hardware_cannot_pass(tmp_path):
    assert not hardware_evidence(tmp_path/"missing.json",{})["passed"]


def test_stale_hardware_cannot_pass(tmp_path):
    path=tmp_path/"metrics.json"
    path.write_text(json.dumps({"passed":True,"environment":{"source_sha256":{"src/primitives.py":"old"}}}))
    assert not hardware_evidence(path,{"src/primitives.py":"new"})["passed"]


def test_incomplete_inventory_cannot_pass(tmp_path):
    path=tmp_path/"metrics.json"
    path.write_text(json.dumps({"passed":True,"environment":{"source_sha256":{}},"device_count":16,"process_count":4,
                               "devices":[{"kind":"TPU v4"}]*16}))
    assert not hardware_evidence(path,{})["passed"]


def test_cpu_reuse_rejects_changes_to_actual_numerics():
    import copy
    env={k:"pinned" for k in ("jax","jaxlib","numpy","scipy")}
    env["source_sha256"]={name:"same" for name in CPU_MEASUREMENT_FILES}
    record={"status":"CPU_VERIFIED_TPU_PENDING","environment":copy.deepcopy(env),
            "numerical":{"passed":True},"monte_carlo":{"passed":True,"samples_per_scale":1_000_000},
            "deep":{"passed":True,"trials_per_depth":10_000,"width":128}}
    assert reusable_cpu_evidence(record,env)
    for name in CPU_MEASUREMENT_FILES:
        changed=copy.deepcopy(env)
        changed["source_sha256"][name]="changed"
        assert not reusable_cpu_evidence(record,changed)
    record["deep"]["passed"]=False
    assert not reusable_cpu_evidence(record,env)
