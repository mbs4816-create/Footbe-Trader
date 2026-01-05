#!/usr/bin/env python3
"""Generate field catalog documentation from the field registry.

This script generates docs/field_catalog.md from the Python field catalog,
ensuring documentation is always in sync with the code.

Usage:
    python scripts/generate_field_catalog.py

    # Or regenerate from live API samples (requires API key):
    python scripts/generate_field_catalog.py --fetch-samples
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.football.field_catalog import (
    FIELD_CATALOG,
    FieldAvailability,
    FieldInfo,
    get_catalog_summary,
    get_endpoints,
    get_fields_by_endpoint,
)


def generate_markdown() -> str:
    """Generate markdown documentation from field catalog."""
    lines = []

    # Header
    lines.append("# API-Football Field Catalog")
    lines.append("")
    lines.append("This document catalogs all fields from API-Football endpoints used for EPL match prediction modeling.")
    lines.append("")
    lines.append("## Field Availability Labels")
    lines.append("")
    lines.append("| Label | Description |")
    lines.append("|-------|-------------|")
    lines.append("| 🟢 `PRE_MATCH_SAFE` | Available before kickoff reliably. Safe for pre-match predictions. |")
    lines.append("| 🟡 `PRE_MATCH_UNCERTAIN` | May be available before kickoff but often missing (e.g., lineups ~1hr before). |")
    lines.append("| 🔴 `POST_MATCH_ONLY` | Only available after match completion. **DO NOT use for pre-match predictions.** |")
    lines.append("")

    # Summary
    summary = get_catalog_summary()
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Fields Cataloged:** {summary['total_fields']}")
    lines.append(f"- **PRE_MATCH_SAFE:** {summary['pre_match_safe']} fields")
    lines.append(f"- **PRE_MATCH_UNCERTAIN:** {summary['pre_match_uncertain']} fields")
    lines.append(f"- **POST_MATCH_ONLY:** {summary['post_match_only']} fields")
    lines.append("")
    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*")
    lines.append("")

    # Table of contents
    lines.append("## Endpoints")
    lines.append("")
    endpoints = get_endpoints()
    for endpoint in endpoints:
        anchor = endpoint.replace("/", "").replace(" ", "-").lower()
        lines.append(f"- [{endpoint}](#{anchor})")
    lines.append("")

    # Fields by endpoint
    lines.append("---")
    lines.append("")

    for endpoint in endpoints:
        lines.append(f"## {endpoint}")
        lines.append("")

        fields = get_fields_by_endpoint(endpoint)

        # Group by availability
        safe_fields = [f for f in fields if f.availability == FieldAvailability.PRE_MATCH_SAFE]
        uncertain_fields = [f for f in fields if f.availability == FieldAvailability.PRE_MATCH_UNCERTAIN]
        post_fields = [f for f in fields if f.availability == FieldAvailability.POST_MATCH_ONLY]

        # Table header
        lines.append("| Field Path | Type | Availability | Description | Example |")
        lines.append("|------------|------|--------------|-------------|---------|")

        # Pre-match safe first (green)
        for field in sorted(safe_fields, key=lambda x: x.path):
            avail = "🟢 `PRE_MATCH_SAFE`"
            nullable = " (nullable)" if field.nullable else ""
            example = _format_example(field.example_value)
            lines.append(f"| `{field.path}` | `{field.python_type}`{nullable} | {avail} | {field.description} | {example} |")

        # Uncertain (yellow)
        for field in sorted(uncertain_fields, key=lambda x: x.path):
            avail = "🟡 `PRE_MATCH_UNCERTAIN`"
            nullable = " (nullable)" if field.nullable else ""
            example = _format_example(field.example_value)
            lines.append(f"| `{field.path}` | `{field.python_type}`{nullable} | {avail} | {field.description} | {example} |")

        # Post-match only (red)
        for field in sorted(post_fields, key=lambda x: x.path):
            avail = "🔴 `POST_MATCH_ONLY`"
            nullable = " (nullable)" if field.nullable else ""
            example = _format_example(field.example_value)
            lines.append(f"| `{field.path}` | `{field.python_type}`{nullable} | {avail} | {field.description} | {example} |")

        lines.append("")

    # Usage notes
    lines.append("---")
    lines.append("")
    lines.append("## Usage Guidelines")
    lines.append("")
    lines.append("### For Pre-Match Prediction Models")
    lines.append("")
    lines.append("Only use fields marked with 🟢 `PRE_MATCH_SAFE` for features in pre-match prediction models.")
    lines.append("")
    lines.append("Fields marked 🟡 `PRE_MATCH_UNCERTAIN` (like lineups) can be used with proper handling:")
    lines.append("- Check if data is available before using")
    lines.append("- Have fallback logic when data is missing")
    lines.append("- Document the uncertainty in your model")
    lines.append("")
    lines.append("### Validation")
    lines.append("")
    lines.append("The `field_catalog.py` module provides validation functions:")
    lines.append("")
    lines.append("```python")
    lines.append("from footbe_trader.football.field_catalog import (")
    lines.append("    is_field_pre_match_safe,")
    lines.append("    validate_pre_match_fields,")
    lines.append(")")
    lines.append("")
    lines.append("# Check single field")
    lines.append("assert is_field_pre_match_safe('standings.points')  # True")
    lines.append("assert not is_field_pre_match_safe('goals.home')     # False - post match only")
    lines.append("")
    lines.append("# Validate multiple fields")
    lines.append("violations = validate_pre_match_fields(['standings.points', 'goals.home'])")
    lines.append("if violations:")
    lines.append("    raise ValueError(f'Post-match fields in pre-match model: {violations}')")
    lines.append("```")
    lines.append("")
    lines.append("### Adding New Fields")
    lines.append("")
    lines.append("When adding new fields to the catalog:")
    lines.append("")
    lines.append("1. Add to `src/footbe_trader/football/field_catalog.py`")
    lines.append("2. Run `python scripts/generate_field_catalog.py` to regenerate this doc")
    lines.append("3. Run tests to ensure no pre-match models use POST_MATCH_ONLY fields")
    lines.append("")

    return "\n".join(lines)


def _format_example(value: Any) -> str:
    """Format example value for markdown table."""
    if value is None:
        return "`null`"
    if isinstance(value, str):
        # Truncate long strings
        if len(value) > 30:
            return f"`\"{value[:27]}...\"`"
        return f"`\"{value}\"`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    return f"`{value}`"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate field catalog documentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/field_catalog.md",
        help="Output file path (default: docs/field_catalog.md)",
    )
    parser.add_argument(
        "--fetch-samples",
        action="store_true",
        help="Fetch fresh sample payloads from API (requires API key)",
    )

    args = parser.parse_args()

    if args.fetch_samples:
        print("⚠️  --fetch-samples not implemented yet. Using existing catalog.")

    # Generate markdown
    markdown = generate_markdown()

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)

    print(f"✅ Generated {output_path}")

    # Print summary
    summary = get_catalog_summary()
    print(f"   Total fields: {summary['total_fields']}")
    print(f"   PRE_MATCH_SAFE: {summary['pre_match_safe']}")
    print(f"   PRE_MATCH_UNCERTAIN: {summary['pre_match_uncertain']}")
    print(f"   POST_MATCH_ONLY: {summary['post_match_only']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
