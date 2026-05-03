import re


def strip_coauthored(message):
    text = message.decode("utf-8")
    # Remove trailing Co-Authored-By Claude lines (with or without trailing newline)
    text = re.sub(r"\n\nCo-Authored-By: Claude[^\n]*\n?", "\n", text)
    text = re.sub(r"\nCo-Authored-By: Claude[^\n]*\n?", "", text)
    # Clean up any double trailing newlines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.rstrip("\n") + "\n"
    return text.encode("utf-8")
