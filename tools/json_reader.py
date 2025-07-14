import json
from pathlib import Path
from abc import ABC, abstractmethod


class Reader(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """To be overriden by the descendant class"""


# 读取jsonl文件
class JSONLReader(Reader):
    def parse_file(file_path: Path) -> list:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            # text = '\n'.join([str(line) for line in lines])
        return lines  # text

    def parse(file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
            text = "\n".join([str(line) for line in lines])
        return text


# 读取json文件
class JSONReader(Reader):
    def parse_file(file_path: Path) -> list:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # text = str(data)
            return data  # text
        except:
            return []

    def parse(self, file_path: Path) -> str:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                text = str(data)
            return text
        except:
            return ""
