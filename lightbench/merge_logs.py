import io
from datetime import datetime
from pathlib import Path
from typing import List

import markdown
import numpy as np
import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_enable=True)


def nanmean(x):
    return np.mean(x[np.isfinite(x)])


def merge_markdown_files(file_paths):
    """Merge multiple markdown files by taking the union of detail experiments and computing new summary stats."""
    latest_timestamp = datetime.min
    union_details = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            content = f.read()
        content = content.split("## Details")[1].split("## ")[0].strip()
        html = markdown.markdown(content, extensions=["tables"])
        union_details.append(pd.read_html(io.StringIO(html))[0])

    union_details = pd.concat(union_details)

    # For duplicate rows (i.e., Benchmark, Optimizer, Cautious, Mars == same), take the best run (i.e., Successful, Runtime, Attempts)
    union_details["Success"] = union_details["Success"] == "✓"
    union_details["Runtime"] = union_details["Runtime"].str.replace("s", "")
    union_details["Runtime"] = pd.to_numeric(union_details["Runtime"], errors="coerce")
    union_details["Attempts"] = pd.to_numeric(union_details["Attempts"], errors="coerce")
    union_details["Loss"] = pd.to_numeric(union_details["Loss"], errors="coerce")

    union_details = union_details.sort_values(by=["Success", "Runtime", "Attempts"], ascending=[False, True, True])
    union_details = union_details.drop_duplicates(keep="first", subset=["Benchmark", "Optimizer", "Cautious", "Mars"])

    configs = union_details[["Optimizer", "Cautious", "Mars"]].drop_duplicates().to_dict(orient="records")

    new_summary = []

    for config in configs:
        config_details = union_details[
            (union_details["Optimizer"] == config["Optimizer"])
            & (union_details["Cautious"] == config["Cautious"])
            & (union_details["Mars"] == config["Mars"])
        ]
        new_summary.append({
            **config,
            "Attempts": nanmean(config_details["Attempts"]),
            "Success": f"{int(np.sum(config_details['Success']))}/{len(config_details)}",
            "Average Runtime": f"{nanmean(config_details['Runtime']):.1f}s",
        })

    new_summary = pd.DataFrame(new_summary)
    new_summary.sort_values(by=["Optimizer", "Cautious", "Mars"], inplace=True)

    union_details["Runtime"] = [f"{x:.1f}s" for x in union_details["Runtime"]]
    union_details["Success"] = ["✓" if x else "✗" for x in union_details["Success"]]

    # Generate merged content with updated summary based on union of experiments
    merged_content = f"""# Benchmark Results

Generated: {latest_timestamp}
Last updated: {latest_timestamp}

## Summary

{new_summary.to_markdown(index=False)}

## Details

{union_details.to_markdown(index=False)}
"""

    return merged_content


@app.command()
def main(path: List[str] = typer.Option([], help="Markdown files to merge")):
    files = [Path(p) for p in path]

    # Generate merged content
    merged_content = merge_markdown_files(files)

    # Write to output file
    output_path = Path("merged_results.md")
    with open(output_path, "w") as f:
        f.write(merged_content)


if __name__ == "__main__":
    app()
