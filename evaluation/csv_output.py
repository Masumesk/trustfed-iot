import csv
import os


def save_round_to_csv(file_path, row):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    file_exists = os.path.exists(
        file_path
    )

    with open(
        file_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)