import re


def data_process(dataset):
    # extract the question, step and answer
    list_data_dict = []
    for data in dataset:
        item = {"task": data["question"]}
        raw_answer = data["answer"]
        raw_answer_list = raw_answer.split("\n####")
        item["step"] = raw_answer_list[0].strip()
        item["answer"] = raw_answer_list[-1].replace(",", "").strip()
        list_data_dict.append(item)

    return list_data_dict


def get_predict(pred_str):
    """从预测字符串中提取答案 - 支持MMLU选择题和数学题"""
    if not isinstance(pred_str, str):
        pred_str = str(pred_str)

    # 先尝试提取MMLU格式的选择题答案 (A, B, C, D)
    # 查找单独的字母答案
    choice_pattern = r"\b[ABCD]\b"
    choice_matches = re.findall(choice_pattern, pred_str)
    if choice_matches:
        return choice_matches[0]  # 返回第一个找到的选择

    # 查找行首的字母答案
    lines = pred_str.split("\n")
    for line in lines:
        line = line.strip()
        if line and line[0] in "ABCD" and (len(line) == 1 or line[1] in ". ):"):
            return line[0]

    # 如果没找到选择题答案，尝试数学题格式
    if "The answer is " in pred_str:
        pred = pred_str.split("The answer is ")[-1].strip()
    elif "the answer is " in pred_str:
        pred = pred_str.split("the answer is ")[-1].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a
    else:
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str)
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""

    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]

    pred = _strip_string(pred)

    if "boxed" in pred:
        ans = pred.split("boxed")[-1]
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a

    if pred.isdigit():
        return pred
    else:
        matches = re.findall(r"\d+", pred)
        return matches[-1] if matches else None


def _strip_string(string):
    """清理字符串中的特殊字符"""
    string = str(string).strip()
    # remove inverse spaces
    string = string.replace("\\!", "")
    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit square
    string = string.replace("\\text{", "")
    string = string.replace("}", "")

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # \sqrt2 --> \sqrt{2}
    string = _fix_sqrt(string)

    return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string
