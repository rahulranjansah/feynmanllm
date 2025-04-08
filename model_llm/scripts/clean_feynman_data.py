import re
from pathlib import Path

def read_feynman_chapter(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Get all chapter directories
    base_dir = Path("data/feynman_content/volume_TOC")
    chapter_dirs = sorted(base_dir.glob('chapter_*'))

    # Read and combine all chapter texts
    text_sequence = ""
    for chapter_dir in chapter_dirs:
        chapter_file = chapter_dir / 'full_chapter.txt'
        if chapter_file.exists():
            text_sequence += read_feynman_chapter(chapter_file) + "\n\n"

    print(f"Total text length: {len(text_sequence)} characters")

    # Save the combined text to a file
    output_file = Path("output/feynman_combined_text.txt")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(text_sequence)

    print(f"Saved combined text to {output_file}")

if __name__ == "__main__":
    main()