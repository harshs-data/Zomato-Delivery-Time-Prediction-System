import json

with open("notebooks/finalEstimator.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("notebooks/finalEstimator.py", "w", encoding="utf-8") as out:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            out.write("".join(cell["source"]) + "\n\n")
