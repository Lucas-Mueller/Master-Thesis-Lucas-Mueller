#!/usr/bin/env python3
"""
Script to extract Mermaid diagrams from markdown files and render them to PNG.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

def extract_mermaid_blocks(markdown_file):
    """Extract all mermaid code blocks from a markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)

    return matches

def render_mermaid_to_png(mermaid_code, output_path, diagram_name):
    """Render a mermaid diagram to PNG using mmdc CLI."""
    # Create temporary .mmd file
    temp_mmd = output_path / f"{diagram_name}_temp.mmd"

    try:
        # Write mermaid code to temporary file
        with open(temp_mmd, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)

        # Render to PNG
        png_output = output_path / f"{diagram_name}.png"

        cmd = [
            'mmdc',
            '-i', str(temp_mmd),
            '-o', str(png_output),
            '-b', 'transparent',  # Transparent background
            '-w', '1920',  # Width
            '-H', '1080',  # Height (will auto-scale)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Rendered: {diagram_name}.png")
            return True
        else:
            print(f"❌ Failed to render {diagram_name}: {result.stderr}")
            return False

    finally:
        # Clean up temporary file
        if temp_mmd.exists():
            temp_mmd.unlink()

def main():
    """Main function to render all diagrams."""
    # Get the diagrams directory
    script_dir = Path(__file__).parent
    rendered_dir = script_dir / "rendered"

    # Ensure rendered directory exists
    rendered_dir.mkdir(exist_ok=True)

    # List of markdown files to process
    diagram_files = [
        ("01_experiment_overview.md", "01_experiment_overview"),
        ("02_system_context.md", "02_system_context"),
        ("03_phase1_architecture.md", "03_phase1_architecture"),
        ("04_phase2_services.md", "04_phase2_services"),
        ("05_data_model_core.md", "05_data_model_core"),
        ("06_discussion_sequence.md", "06_discussion_sequence"),
        ("07_voting_sequence.md", "07_voting_sequence"),
        ("08_memory_flow.md", "08_memory_flow"),
        ("09_payoff_calculation.md", "09_payoff_calculation"),
    ]

    success_count = 0
    total_diagrams = 0

    for md_file, diagram_name in diagram_files:
        md_path = script_dir / md_file

        if not md_path.exists():
            print(f"⚠️  File not found: {md_file}")
            continue

        print(f"\n📄 Processing {md_file}...")

        # Extract mermaid blocks
        mermaid_blocks = extract_mermaid_blocks(md_path)

        if not mermaid_blocks:
            print(f"  No mermaid diagrams found in {md_file}")
            continue

        # Render each block (usually just one per file)
        for idx, mermaid_code in enumerate(mermaid_blocks):
            total_diagrams += 1

            # If multiple diagrams, add suffix
            if len(mermaid_blocks) > 1:
                name = f"{diagram_name}_{idx+1}"
            else:
                name = diagram_name

            if render_mermaid_to_png(mermaid_code, rendered_dir, name):
                success_count += 1

    print(f"\n{'='*60}")
    print(f"✨ Rendering complete!")
    print(f"   Successfully rendered: {success_count}/{total_diagrams} diagrams")
    print(f"   Output directory: {rendered_dir}")
    print(f"{'='*60}\n")

    return 0 if success_count == total_diagrams else 1

if __name__ == "__main__":
    sys.exit(main())
