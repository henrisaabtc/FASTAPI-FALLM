import csv


def read_and_sort_csv(
    file_path, delimiter=";", encoding="utf-8", row_delimiter="\n", column_delimiter=";"
):
    """Read the csv"""
    with open(file_path, "r", encoding=encoding) as file:
        csv_reader = csv.reader(file, delimiter=delimiter)

        header = next(csv_reader)

        data = list(csv_reader)

        data_sorted = sorted(data, key=lambda row: tuple(row))

        csv_list = data_sorted

        formatted_lines = []

        for row in csv_list:
            formatted_row = "; ".join(
                f"{name}: {value}" for name, value in zip(header, row)
            )
            formatted_lines.append(formatted_row + ";")

        text = row_delimiter.join(formatted_lines)

        return text


file_path = "test/conversion/ListTest.csv"
sorted_data = read_and_sort_csv(file_path)
print(sorted_data)
