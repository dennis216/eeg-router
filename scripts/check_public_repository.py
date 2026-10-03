"""Check tracked/new source syntax and local Markdown link targets without writes."""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    errors = []
    python_count = link_count = 0
    for name in sorted(set(filter(None, listed))):
        path = ROOT / name
        if path.suffix not in {".py", ".md"}:
            continue
        content = path.read_text(encoding="utf-8-sig")
        if path.suffix == ".py":
            python_count += 1
            try:
                ast.parse(content, filename=name)
            except SyntaxError as exc:
                errors.append(f"{name}: {exc}")
            continue
        # Documentation uses inline Markdown links; fenced examples are not links.
        content = re.sub(r"```.*?```", "", content, flags=re.S)
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", content):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            link_count += 1
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f"{name}: missing/outside local link {target}")
                continue
            # Git paths are case-sensitive even when the local filesystem is not.
            current = ROOT
            for part in resolved.relative_to(ROOT).parts:
                if part not in {entry.name for entry in current.iterdir()}:
                    errors.append(f"{name}: case mismatch in {target}")
                    break
                current = current / part
    print(f"Checked {python_count} Python files and {link_count} local Markdown links.")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Syntax and local link targets: PASS")


if __name__ == "__main__":
    main()
