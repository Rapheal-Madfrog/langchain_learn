# -*- coding: utf-8 -*-
# author = 'ty'
import os
import yaml
import re
from typing import Dict, Any, List
from utils.log_util import logger
from dataclasses import asdict, fields

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_mysql_config(env, mysql_name='creative'):
    """获取对应mysql配置信息"""
    with open(f"{script_dir}/../resources/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["mysql"][env][mysql_name]


def load_yaml_with_env(
        path=None,
        data=None,
        tag='!ENV',
        loader=yaml.SafeLoader,
        encoding='utf-8'
):
    pattern = re.compile(r'\$\{([^}]+)\}')

    def constructor_env_variables(loader, node):
        value = loader.construct_scalar(node)
        env_vars = pattern.findall(value)
        for var in env_vars:
            env_value = os.environ.get(var, "N/A")
            value = value.replace(f'${{{var}}}', env_value)
        return value

    loader.add_constructor(tag, constructor_env_variables)

    if path:
        with open(path, encoding=encoding) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError('Either a path or data should be defined as input')


def dataclass_to_yaml(data_obj, additional_indent):
    data_dict = asdict(data_obj)
    # eliminate default values
    for f in fields(data_obj):
        if (
                data_dict[f.name] == f.default is None
        ):
            del data_dict[f.name]
    if len(data_dict) == 0:
        return None
    yaml_content = yaml.dump(data_dict)
    if additional_indent > 0:
        additional = ' ' * additional_indent
        return additional + re.sub('\n', '\n' + additional, yaml_content)
    else:
        return yaml_content


def flatten_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    递归展开嵌套字典，将所有内层的非字典值提取到一个新的字典中。

    参数:
    d (Dict[Any, Any]): 一个可能包含嵌套字典的字典。

    返回:
    Dict[Any, Any]: 一个展开后的字典，其中所有非字典值被保留下来。
    """
    result = {}  # 存放展开后的字典

    # 遍历字典的每一个键值对
    for key, value in d.items():
        if isinstance(value, dict):
            # 如果值是字典类型，则递归调用 flatten_dict 进行展开
            result.update(flatten_dict(value))
        else:
            # 如果值不是字典类型，直接将键值对加入结果字典
            result[key] = value

    return result


def partial_flatten_dict(d: Dict[Any, Any], flatten_layer: int) -> Dict[Any, Any]:
    """
    根据设定的层数展开字典，保留到达指定层数的字典结构。

    参数:
    d (Dict[Any, Any]): 输入的嵌套字典。
    flatten_layer (int): 要展开的字典层数。

    返回:
    Dict[Any, Any]: 展开后的字典，保留指定层数的嵌套字典。
    """
    if flatten_layer == 0:
        # 如果flatten_layer为0，则不展开，直接返回当前字典
        return d

    result = {}

    # 遍历字典中的每个键值对
    for key, value in d.items():
        if isinstance(value, dict):
            # 如果值是字典类型并且还需要展开，则递归展开到下一层
            result.update(partial_flatten_dict(value, flatten_layer - 1))
        else:
            # 如果值不是字典类型，直接加入结果
            result[key] = value

    return result


def flatten_dict_to_layer(d: Dict[Any, Any], flatten_layer: int) -> Dict[Any, Any]:
    """
    根据设定的层数展开字典，仅保留对应层数的键值对。

    参数:
    d (Dict[Any, Any]): 输入的嵌套字典。
    flatten_layer (int): 要展开到的目标层数。

    返回:
    Dict[Any, Any]: 展开后的字典，仅保留指定层数的键值对。
    """
    if flatten_layer == 0:
        # 到达层级0时，停止展开并返回原字典（保留该层所有内容）
        return d

    result = {}

    # 遍历字典中的每个键值对
    for key, value in d.items():
        if isinstance(value, dict):
            # 如果值是字典，且还没有达到flatten_layer，则继续递归
            result.update(flatten_dict_to_layer(value, flatten_layer - 1))

    return result


def merge_dict(dict_: dict, dict_candidate: List[dict]):
    for d in dict_candidate:
        for k, v in d.items():
            dict_[k] = v


if __name__ == '__main__':
    print(load_yaml_with_env('F:\project\ctr\examples\ctr_torch_test\config_.yml'))
