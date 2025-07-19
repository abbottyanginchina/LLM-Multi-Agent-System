import json
from typing import Union, List, Literal, Any, Dict
from abc import ABC


class Query(ABC):
    def __init__(
        self,
        split: Union[Literal["dev"], Literal["val"], Literal["test"]],
    ) -> None:

        self._split = split

        data_path = f"dataset/query/"
        self._total_data: List[Dict[str, Any]] = self._load_data(data_path)

    @staticmethod
    def _load_data(
        data_path: str,
    ) -> List[Dict[str, Any]]:

        json_files = []
        try:
            import glob

            json_paths = glob.glob(data_path + "*.json")
            json_paths = sorted(json_paths)
            print("Number of JSON files: ", len(json_paths))

            total_data = []
            for path in json_paths:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        total_data.extend(data)
                    else:
                        total_data.append(data)
        except Exception as e:
            print(f"Error loading JSON files: {e}")
            total_data = []

        print("Total number of tasks: ", len(total_data))

        return total_data

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._total_data[index]
        assert isinstance(record, dict)
        return record

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        task_data = record.get("task", {})

        # 验证必需字段
        required_fields = ["theme", "genre", "length"]
        for field in required_fields:
            if not task_data.get(field):
                raise ValueError(
                    f"Required field '{field}' is missing or empty in task data"
                )

        # 构建完整的任务描述
        task_description = f"""Create a story with the following specifications:

Theme: {task_data['theme']}
Genre: {task_data['genre']}
Length: {task_data['length']}"""

        # 添加可选字段
        if task_data.get("setting"):
            task_description += f"\nSetting: {task_data['setting']}"

        if task_data.get("tone"):
            task_description += f"\nTone: {task_data['tone']}"

        if task_data.get("constraints"):
            constraints = task_data["constraints"]
            if isinstance(constraints, list) and constraints:
                task_description += "\nConstraints:"
                for constraint in constraints:
                    task_description += f"\n- {constraint}"

        input_dict = {"task": task_description}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")

        # 对于故事生成任务，直接返回答案，不需要特殊的后处理
        return answer.strip()
