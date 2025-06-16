class DataFrame:
    def __init__(self, data):
        # store data as dict of lists
        self.data = {k: list(v) for k, v in data.items()}
        self.columns = list(data.keys())

    def __repr__(self):
        return f"DataFrame({self.data})"

def get_dummies(df, drop_first=False):
    if not isinstance(df, DataFrame):
        raise TypeError("df must be a DataFrame")
    new_data = {}
    for col in df.columns:
        values = df.data[col]
        categories = sorted(set(values))
        if drop_first:
            categories = categories[1:]
        for cat in categories:
            name = f"{col}_{cat}"
            new_data[name] = [1 if v == cat else 0 for v in values]
    return DataFrame(new_data)

