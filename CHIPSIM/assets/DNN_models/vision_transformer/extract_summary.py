#!/usr/bin/env python3
import re
from pathlib import Path


def parse_instance_mapping(text: str) -> dict:
    """Parse a simple inline mapping like "{A: 1, B: 2}" into a dict.
    Values are converted to int when possible.
    """
    instance: dict = {}
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return instance
    inner = text[1:-1].strip()
    if not inner:
        return instance
    # Split on commas not inside potential quotes (though we don't expect quotes here)
    parts = [p for p in inner.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value_str = value.strip()
        # Try to cast to int
        if re.fullmatch(r"[-+]?\d+", value_str or ""):
            instance[key] = int(value_str)
        else:
            instance[key] = value_str
    return instance


def extract_problem_fields(file_path: Path) -> dict:
    """Extract only the requested fields from the problem block of a YAML file
    without fully parsing YAML (to avoid custom anchors/includes).
    Returns a dict with keys: file, name, dnn_name, notes, instance (dict).
    """
    result = {"file": file_path.name}
    try:
        with file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return result

    in_problem = False
    base_indent = None
    # We stop scanning after we reach histograms
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not in_problem:
            if stripped == "problem:" or stripped.startswith("problem:"):
                in_problem = True
                base_indent = len(line) - len(line.lstrip(" "))
            continue

        # If we reached a new top-level (same or less indent) key other than desired block, keep going
        # but our stop signal is specifically histograms.
        if stripped.startswith("histograms:"):
            break

        # Capture fields when present
        if stripped.startswith("instance:"):
            after_colon = line.split(":", 1)[1].strip()
            instance_map = parse_instance_mapping(after_colon)
            if instance_map:
                result["instance"] = instance_map
            continue

        for key in ("name", "dnn_name", "notes"):
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                value = stripped.split(":", 1)[1].strip()
                # remove optional quotes
                if (value.startswith("\"") and value.endswith("\"")) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                result[key] = value
                break

    return result


def write_summary(output_path: Path, entries: list) -> None:
    """Write a simple YAML list with the extracted entries using only stdlib."""
    with output_path.open("w", encoding="utf-8") as out:
        out.write("# Auto-generated summary of problem metadata for vision_transformer\n")
        out.write("# Contains: file, name, dnn_name, notes, instance\n")
        for entry in entries:
            out.write("- file: \"%s\"\n" % entry.get("file", ""))
            for key in ("name", "dnn_name", "notes"):
                if key in entry:
                    out.write(f"  {key}: {entry[key]}\n")
            # Instance mapping
            instance = entry.get("instance")
            if isinstance(instance, dict):
                out.write("  instance:\n")
                for k, v in instance.items():
                    out.write(f"    {k}: {v}\n")


def main() -> None:
    here = Path(__file__).resolve().parent
    output_file = here / "vision_transformer_summary.yaml"

    yaml_files = sorted(
        [p for p in here.glob("*.yaml") if p.name != output_file.name],
        key=lambda p: p.name,
    )

    entries = []
    for yf in yaml_files:
        info = extract_problem_fields(yf)
        # Only include if we captured at least a name or instance
        if any(k in info for k in ("name", "instance", "dnn_name", "notes")):
            entries.append(info)

    write_summary(output_file, entries)


if __name__ == "__main__":
    main()


