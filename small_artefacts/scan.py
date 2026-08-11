from pathlib import Path

# --------------------------------------------------
# CHANGE THIS TO THE FOLDER YOU WANT TO SCAN
# --------------------------------------------------
ROOT_FOLDER = Path(r"E:\Dissertation - Code\HPC -Files")

# Output text file
OUTPUT_FILE = Path("file_list.txt")


def scan_folder(root_folder):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        f.write(f"Root folder: {root_folder}\n")
        f.write("=" * 100 + "\n\n")

        file_count = 0

        # rglob("*") recursively searches all folders/subfolders
        for path in root_folder.rglob("*"):

            if path.is_file():
                file_count += 1

                # Folder relative to the root directory
                relative_folder = path.parent.relative_to(root_folder)

                f.write(
                    f"Filename: {path.name}\n"
                    f"Subfolder: {relative_folder}\n"
                    f"{'-' * 100}\n"
                )

        f.write(f"\nTotal files found: {file_count}\n")

    print(f"Done.")
    print(f"Total files found: {file_count}")
    print(f"Results saved to: {OUTPUT_FILE.resolve()}")


scan_folder(ROOT_FOLDER)