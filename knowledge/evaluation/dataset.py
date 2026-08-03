from pathlib import Path
import json


class EvaluationDataset:

    def __init__(self, path):

        self.path = Path(path)

    def load(self):

        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)