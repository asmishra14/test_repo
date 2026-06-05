from pathlib import Path

path = Path("player_count.py")

text = path.read_text(encoding="utf-8")

# Replace common problematic Unicode whitespace characters
text = (
    text.replace("\u00A0", " ")  # NO-BREAK SPACE
        .replace("\u2007", " ")  # FIGURE SPACE
        .replace("\u202F", " ")  # NARROW NO-BREAK SPACE
)

path.write_text(text, encoding="utf-8")