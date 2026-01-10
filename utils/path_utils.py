from pathlib import Path

def get_project_root() -> Path:
    """
    Returns the project root directory.
    Searches upwards for a marker file/directory (like 'main.py' or '.git').
    """
    current_path = Path.cwd().resolve()
    
    # Try finding main.py or .git in current or parent directories
    for p in [current_path] + list(current_path.parents):
        if (p / 'main.py').exists() or (p / '.git').exists():
            return p
            
    # Fallback to current working directory if not found (though unlikely in this repo)
    return current_path
