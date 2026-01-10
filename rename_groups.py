import os
from pathlib import Path

BASE = Path("experiment_execution/sensitivity_analysis_llm")
SUBDIRS = ["configs", "results", "terminal_outputs", "transcripts"]
MAPPING = {
    "american": "group_1",
    "chinese": "group_2"
}

def rename_dirs_and_files():
    print("Starting process...")
    for sub in SUBDIRS:
        sub_path = BASE / sub
        if not sub_path.exists():
            print(f"Skipping {sub}: not found")
            continue
            
        # Rename directories
        for old_name, new_name in MAPPING.items():
            old_dir = sub_path / old_name
            new_dir = sub_path / new_name
            
            if old_dir.exists():
                print(f"Renaming directory {old_dir} -> {new_dir}")
                # Handle case where new_dir might already exist (unlikely but safe)
                if new_dir.exists():
                    print(f"Target {new_dir} already exists. Merging/Overwriting...")
                    # Basic merge: move contents? No, just warn.
                    # Assuming clean state, rename is fine.
                old_dir.rename(new_dir)
            elif new_dir.exists():
                 print(f"Directory {new_dir} already exists (already renamed?). Processing files inside.")

            
            # Now process files inside new_dir
            target_dir = new_dir
            if not target_dir.exists():
                continue

            # Recursively rename files
            # Note: rglob returns a generator. If we rename files, we might confuse it?
            # Better to collect list first.
            files = list(target_dir.rglob("*"))
            for file_path in files:
                if file_path.is_file():
                    fname = file_path.name
                    if old_name in fname:
                        new_fname = fname.replace(old_name, new_name)
                        print(f"Renaming file {fname} -> {new_fname}")
                        try:
                            file_path.rename(file_path.with_name(new_fname))
                        except Exception as e:
                            print(f"Error renaming {fname}: {e}")

if __name__ == "__main__":
    rename_dirs_and_files()
