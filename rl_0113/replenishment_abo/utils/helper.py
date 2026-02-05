from pathlib import Path

def get_conf_path(args):
    "根据 base_dir, task_name, data_ver, para_ver 拼接 config 的 json path"
    base_path = Path(args.base_dir)
    outs_path = base_path.joinpath("output", args.task_name, args.data_ver, args.para_ver)
    conf_path = outs_path.joinpath("model.json")
    return str(conf_path)

def create_path_with_suffix(path, suffix):
    "给 path 增加后缀, 用于 predict 新数据"
    file, ext = path.rsplit(".", 1)
    newpath = ".".join([file, suffix, ext])
    return newpath