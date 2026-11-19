"""Copy the documents folder into ./docs so the web version of the app has them.

Run before pushing to GitHub:   python sync_docs.py
Adds new files, updates changed ones, and removes files you've deleted from the source folder.
"""
import filecmp
import shutil
from pathlib import Path

SOURCE = None   # ITA app: docs/ is maintained by hand. Set a folder here to sync from it instead.
TARGET = Path(__file__).parent / "docs"
TYPES = {".pdf", ".docx"}


def main():
    if SOURCE is None or not Path(SOURCE).is_dir():
        print("No SOURCE folder set — edit files in docs/ directly. Nothing changed.")
        return
    TARGET.mkdir(exist_ok=True)
    wanted = {}
    for f in SOURCE.rglob("*"):
        if f.is_file() and f.suffix.lower() in TYPES and not f.name.startswith("~$"):
            wanted[f.relative_to(SOURCE)] = f
    for rel, src in sorted(wanted.items()):
        dst = TARGET / rel
        if dst.exists() and filecmp.cmp(src, dst, shallow=False):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print("copied ", rel)
    for f in sorted(TARGET.rglob("*")):
        if f.is_file() and f.relative_to(TARGET) not in wanted:
            f.unlink()
            print("removed", f.relative_to(TARGET))
    print(f"docs/ now has {len(wanted)} files.")


if __name__ == "__main__":
    main()
