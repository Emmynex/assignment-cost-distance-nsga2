import numpy as np

def load_data(path, n=100):
    # method to load data into the list.
    rows = []
    current_row = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.isdigit() and int(line) == n and len(rows) == 0:
                continue

            #
            nums = list(map(int, line.split()))
            current_row.extend(nums)

            if len(current_row) == n:
                rows.append(current_row)
                current_row = []

    return np.array(rows)