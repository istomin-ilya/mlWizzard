from pathlib import Path


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def parse_metadata(text: str, path: Path) -> dict:
    metadata = {
        "ticker": path.stem.split("_")[0].upper(),
        "company_name": "Unknown",
        "document_type": "educational synthetic company brief",
        "fiscal_period": "2025",
        "source_file": path.name,
    }

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "ticker":
            metadata["ticker"] = value.upper()
        elif key == "company":
            metadata["company_name"] = value
        elif key == "document type":
            metadata["document_type"] = value
        elif key == "fiscal period":
            metadata["fiscal_period"] = value
        elif key == "generated for demo":
            metadata["generated_for_demo"] = value

    return metadata


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - chunk_overlap

    return chunks


def read_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = clean_text(text)
    metadata = parse_metadata(text, path)

    return {
        "text": text,
        "metadata": metadata,
    }


def load_reports(reports_dir: str = "data/reports") -> list[dict]:
    directory = Path(reports_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Reports directory not found: {directory}")

    reports = []

    for path in sorted(directory.glob("*.txt")):
        reports.append(read_report(path))

    return reports


if __name__ == "__main__":
    reports = load_reports()

    print(f"Reports loaded: {len(reports)}")

    for report in reports:
        metadata = report["metadata"]
        chunks = chunk_text(report["text"])

        print(
            f"{metadata['ticker']} | "
            f"{metadata['company_name']} | "
            f"chunks: {len(chunks)}"
        )