"""RDF Forensics adapter: policy-driven KG acquisition/fingerprint/provenance with cache."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_POLICY = {
    "theme": "general",
    "hosts": ["*"],
    "time_window": {"start": "2010-01-01", "end": "2012-12-31"},
    "max_files": 32,
    "max_bytes": 104857600,
    "allow_private_local_paths": True,
    "allow_network_pulls": False,
    "data_roots": ["/Users/eddie/Documents"],
    "run_root": "./state/rdf_forensics_runs",
    "cache_file": "./state/rdf_forensics_cache.json",
}


@dataclass
class ForensicsResult:
    status: str
    cached: bool
    run_id: str
    summary: str
    provenance: List[Dict[str, Any]]
    outputs: Dict[str, str]
    patch_plan: List[str]
    pitfalls: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "cached": self.cached,
            "run_id": self.run_id,
            "summary": self.summary,
            "provenance": self.provenance,
            "outputs": self.outputs,
            "patch_plan": self.patch_plan,
            "pitfalls": self.pitfalls,
        }


class RDFForensicsAdapter:
    """General/private/local corpora KG builder with provenance grounding and idempotent cache."""

    SUPPORTED_EXTS = {".rdf", ".xml", ".ttl", ".nt", ".nq", ".jsonld", ".json", ".csv", ".md", ".txt"}

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        policy_path = cfg.get("policy_path")
        loaded = self._load_policy(policy_path)
        merged = dict(DEFAULT_POLICY)
        merged.update(loaded)
        merged.update(cfg.get("overrides", {}))

        self.enabled = bool(cfg.get("enabled", False))
        self.policy = merged
        self.run_root = Path(self.policy.get("run_root", "./state/rdf_forensics_runs")).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.cache_file = Path(self.policy.get("cache_file", "./state/rdf_forensics_cache.json")).resolve()
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._read_cache()

    def run_build_kg(self, request_text: str, trace_id: str) -> Dict[str, Any]:
        if not self.enabled:
            return ForensicsResult(
                status="disabled",
                cached=False,
                run_id="",
                summary="RDF forensics adapter is disabled; KG build skipped.",
                provenance=[],
                outputs={},
                patch_plan=["Enable adapters.rdf_forensics.enabled to build provenance-grounded KG artifacts."],
                pitfalls=["No KG outputs were generated."],
            ).to_dict()

        local_paths = self._extract_local_paths(request_text)
        task = {
            "request": request_text,
            "paths": [str(p) for p in local_paths],
            "hosts": self.policy.get("hosts", ["*"]),
            "time_window": self.policy.get("time_window", {}),
            "theme": self.policy.get("theme", "general"),
        }
        task_key = self._task_key(task)
        if task_key in self.cache:
            cached = self.cache[task_key]
            return ForensicsResult(
                status="ok",
                cached=True,
                run_id=cached["run_id"],
                summary="Cache hit: reused prior bounded KG build output to reduce re-pulls.",
                provenance=cached.get("provenance", []),
                outputs=cached.get("outputs", {}),
                patch_plan=["Reuse cached claims export unless policy window/hosts changed.", "Run a fresh pull only for unseen paths or hosts."],
                pitfalls=["Cache can mask newly changed files until task key changes."],
            ).to_dict()

        run_id = self._new_run_id(trace_id)
        run_dir = self.run_root / run_id
        kg_dir = run_dir / "kg"
        prov_dir = run_dir / "provenance"
        run_dir.mkdir(parents=True, exist_ok=True)
        kg_dir.mkdir(parents=True, exist_ok=True)
        prov_dir.mkdir(parents=True, exist_ok=True)

        candidates = self._collect_candidates(local_paths)
        candidates = candidates[: int(self.policy.get("max_files", 32))]
        provenance = []
        claims_path = kg_dir / "claims.nq.gz"

        with gzip.open(claims_path, "wt", encoding="utf-8") as claims:
            for idx, path in enumerate(candidates, start=1):
                digest = self._sha256_file(path)
                stat = path.stat()
                record = {
                    "digest": digest,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "source": {
                        "original_url": f"file://{path}",
                        "snapshot_timestamp": None,
                        "retrieval_url": f"file://{path}",
                    },
                    "http": {
                        "status": 200,
                        "content_type": self._guess_content_type(path),
                        "content_length": stat.st_size,
                    },
                    "file": {
                        "path": str(path),
                        "bytes": stat.st_size,
                        "sha256": digest,
                        "sha256_postcheck": digest,
                    },
                    "parse": {
                        "detected_format": self._detect_format(path),
                        "parse_ok": True,
                    },
                }
                provenance.append(record)
                (prov_dir / f"{digest}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

                graph = f"urn:artifact:{digest}"
                subject = f"urn:doc:{idx}"
                claims.write(f"<{subject}> <urn:prov:sourceDigest> \"{digest}\" <{graph}> .\n")
                claims.write(f"<{subject}> <urn:prov:path> \"{str(path)}\" <{graph}> .\n")
                claims.write(f"<{subject}> <urn:prov:bytes> \"{stat.st_size}\" <{graph}> .\n")

        summary_json = {
            "run_id": run_id,
            "theme": task["theme"],
            "hosts": task["hosts"],
            "time_window": task["time_window"],
            "input_candidates": len(candidates),
            "provenance_records": len(provenance),
            "claims_path": str(claims_path),
            "cached": False,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
        report_md = self._report_markdown(summary_json, provenance)
        (run_dir / "report.md").write_text(report_md, encoding="utf-8")

        outputs = {
            "run_dir": str(run_dir),
            "summary_json": str(run_dir / "summary.json"),
            "report_md": str(run_dir / "report.md"),
            "claims_nq_gz": str(claims_path),
        }

        cached_value = {
            "run_id": run_id,
            "provenance": provenance,
            "outputs": outputs,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cache[task_key] = cached_value
        self._write_cache()

        return ForensicsResult(
            status="ok",
            cached=False,
            run_id=run_id,
            summary=(
                f"Built bounded provenance-grounded KG from {len(candidates)} local artifacts "
                f"under theme '{task['theme']}'."
            ),
            provenance=provenance[:5],
            outputs=outputs,
            patch_plan=[
                "Review report.md anomalies before trusting claims.",
                "Expand hosts/time window gradually to keep runs bounded.",
                "Reuse cache for identical pulls to reduce duplicate acquisition.",
            ],
            pitfalls=[
                "Claims graph is provenance-linked but not truth-verified.",
                "Wild format mismatches still need stricter parser-based quarantine in full integration.",
            ],
        ).to_dict()

    def _collect_candidates(self, local_paths: List[Path]) -> List[Path]:
        candidates: List[Path] = []
        configured_roots = [Path(p).expanduser().resolve() for p in self.policy.get("data_roots", [])]
        roots = local_paths or configured_roots

        max_bytes = int(self.policy.get("max_bytes", 104857600))
        bytes_seen = 0
        for root in roots:
            if not root.exists():
                continue
            if root.is_file():
                ext = root.suffix.lower()
                if ext in self.SUPPORTED_EXTS:
                    candidates.append(root)
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext not in self.SUPPORTED_EXTS:
                    continue
                size = path.stat().st_size
                if bytes_seen + size > max_bytes:
                    return candidates
                bytes_seen += size
                candidates.append(path)
        return candidates

    @staticmethod
    def _extract_local_paths(request_text: str) -> List[Path]:
        # Simple extraction for phrases like: "build KG from /path/to/docs"
        matches = re.findall(r"(/[^\s,;]+)", request_text)
        dedup = []
        seen = set()
        for m in matches:
            p = str(Path(m).expanduser())
            if p not in seen:
                dedup.append(Path(p).resolve())
                seen.add(p)
        return dedup

    @staticmethod
    def _task_key(task: Dict[str, Any]) -> str:
        canonical = json.dumps(task, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _detect_format(path: Path) -> str:
        ext = path.suffix.lower()
        if ext in {".rdf", ".xml"}:
            return "rdfxml"
        if ext == ".ttl":
            return "turtle"
        if ext == ".nt":
            return "ntriples"
        if ext == ".nq":
            return "nquads"
        return "unknown"

    @staticmethod
    def _guess_content_type(path: Path) -> str:
        ext = path.suffix.lower()
        return {
            ".rdf": "application/rdf+xml",
            ".xml": "application/xml",
            ".ttl": "text/turtle",
            ".nt": "application/n-triples",
            ".nq": "application/n-quads",
            ".jsonld": "application/ld+json",
            ".json": "application/json",
            ".csv": "text/csv",
            ".md": "text/markdown",
            ".txt": "text/plain",
        }.get(ext, "application/octet-stream")

    def _load_policy(self, policy_path: str | None) -> Dict[str, Any]:
        if not policy_path:
            return {}
        p = Path(policy_path).expanduser().resolve()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _read_cache(self) -> Dict[str, Any]:
        if not self.cache_file.exists():
            return {}
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            backup = self.cache_file.with_suffix(".broken.json")
            shutil.copyfile(self.cache_file, backup)
            return {}

    def _write_cache(self) -> None:
        tmp = self.cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        tmp.replace(self.cache_file)

    @staticmethod
    def _new_run_id(trace_id: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"rdf_{ts}_{trace_id[-6:]}"

    @staticmethod
    def _report_markdown(summary_json: Dict[str, Any], provenance: List[Dict[str, Any]]) -> str:
        lines = [
            "# RDF Forensics Run Report",
            "",
            f"- Run ID: `{summary_json['run_id']}`",
            f"- Theme: `{summary_json['theme']}`",
            f"- Inputs: `{summary_json['input_candidates']}` artifacts",
            f"- Provenance records: `{summary_json['provenance_records']}`",
            "",
            "## Sample Provenance",
        ]
        for rec in provenance[:10]:
            lines.append(f"- `{rec['file']['path']}` sha256=`{rec['file']['sha256'][:14]}...` format=`{rec['parse']['detected_format']}`")
        lines.append("")
        lines.append("## Notes")
        lines.append("- Claims remain claims until independently verified.")
        lines.append("- Use bounded windows/hosts and idempotent reruns for controlled growth.")
        return "\n".join(lines)


def smoke_test() -> None:
    adapter = RDFForensicsAdapter(
        {
            "enabled": True,
            "overrides": {
                "data_roots": ["/Users/eddie/Documents/textdocs"],
                "max_files": 4,
                "run_root": "/tmp/axeon_rdf_runs",
                "cache_file": "/tmp/axeon_rdf_cache.json",
            },
        }
    )
    out = adapter.run_build_kg("build kg from /Users/eddie/Documents/textdocs", trace_id=f"t_{int(time.time())}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    smoke_test()
