#!/usr/bin/env python3
# Copyright 2026 Zhemin Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit a release tree for private artifacts, credentials, and provenance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


_EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "install",
    "log",
}
_FORBIDDEN_FILENAMES = {
    "conference_101719.tex",
    "MUJOCO_LOG.TXT",
}
_FORBIDDEN_SUFFIXES = {
    ".bag",
    ".db3",
    ".mcap",
    ".mov",
    ".mp4",
    ".pyc",
}
_TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".in",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".urdf",
    ".xacro",
    ".xml",
    ".yaml",
    ".yml",
}
_CONTENT_PATTERNS = {
    "user home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "private IPv4 address": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "generic bearer token": re.compile(
        r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"
    ),
}
_REQUIRED_PROVENANCE = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "mujoco/SOURCE.yml",
    "mujoco/SHA256SUMS",
    "mujoco/vendor/robotiq_2f85/LICENSE",
    "mujoco/vendor/universal_robots_ur5e/LICENSE",
    "docs/ur5e_robotiq_model_provenance.md",
)


@dataclass(frozen=True)
class AuditFinding:
    """One actionable repository audit failure."""

    path: str
    reason: str


def _repository_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def audit_repository(root: Path) -> list[AuditFinding]:
    """Return every privacy, artifact, credential, or provenance finding."""
    repository = root.resolve()
    findings: list[AuditFinding] = []
    for required in _REQUIRED_PROVENANCE:
        if not (repository / required).is_file():
            findings.append(AuditFinding(required, "required provenance missing"))

    for path in _repository_files(repository):
        relative = path.relative_to(repository).as_posix()
        lower_parts = {part.lower() for part in path.relative_to(repository).parts}
        if path.name in _FORBIDDEN_FILENAMES:
            findings.append(AuditFinding(relative, "forbidden private artifact"))
        if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            findings.append(AuditFinding(relative, "forbidden binary/log artifact"))
        if "nac_logs" in lower_parts:
            findings.append(AuditFinding(relative, "private experiment log tree"))
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(AuditFinding(relative, "unexpected binary text file"))
            continue
        for reason, pattern in _CONTENT_PATTERNS.items():
            if pattern.search(content):
                findings.append(AuditFinding(relative, reason))
    return findings


def main() -> int:
    """Audit a supplied repository root and print concise findings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    findings = audit_repository(args.root)
    for finding in findings:
        print(f"{finding.path}: {finding.reason}")
    if findings:
        print(f"Release audit failed with {len(findings)} finding(s).")
        return 1
    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
