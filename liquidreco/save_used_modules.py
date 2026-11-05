from pathlib import Path

def log_module_name(module_name, file_path="modules.txt", header="Modules run:\n"):
    path = Path(file_path)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a") as f:
        if is_new:
            f.write(header)
        f.write(module_name + "\n")