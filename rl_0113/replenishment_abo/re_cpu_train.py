
import logging
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from pathlib import Path
from trainer import TrainerConfig
from task import task_dict
from utils.io import read_json
from utils.log import logging_once
from a_refactor_train import Trainer


def setup_cpu(rank, world_size, master_addr="localhost", master_port="12355", socket_ifname="lo0"):
    """
    在CPU上初始化分布式环境
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["GLOO_SOCKET_IFNAME"] = socket_ifname  # 加这行，防止在多节点训练时，不同节点之间通信失败
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    """
    清理分布式环境
    """
    dist.destroy_process_group()


def run_one_process(rank, world_size, config, master_addr="127.0.0.1", master_port="12355"):
    """
    在CPU上运行分布式训练
    """
    try:
        # 初始化分布式环境
        setup_cpu(rank, world_size, master_addr, master_port)
        config.distributed = True if world_size > 1 else False
        # 在这里已经讲rank写进config了，每个进程的rank都不一样的
        config.rank = rank
        # 将device设置为cpu
        config.device = "cpu"
        config.world_size = world_size
        # 创建训练器并训练
        trainer = Trainer(config)
        trainer.train()   
    except Exception as e:
        logging_once(f"Error in process {rank}: {e}", logging.CRITICAL)
        raise e
    finally:
        cleanup()


def str2bool(str):
    return True if str.lower() == "true" else False


def arg_parser():
    # 分布式相关设置
    parser = argparse.ArgumentParser(description="CPU Distributed ReplenishAgent Training")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes per node")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Master node address")
    parser.add_argument("--master_port", type=str, default="12355", help="Master node port")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of nodes")

    # 训练版本相关设置
    parser.add_argument("--task_name", type=str, required=True, help="任务名")
    parser.add_argument("--data_ver",type=str,required=True,help="数据版本,任务名+数据版本=数据文件夹,所有生成的数据都放在数据文件夹下面")
    parser.add_argument("--para_ver", type=str, required=True, help="实验版本,所有生成的数据在数据文件夹下面,由实验版本作为开头")
    parser.add_argument("--json_path", type=str, default="", help="某些task可能需要从json中解析需要的内容")
    # parser.add_argument(
    # "--json_path_ls", 
    # type=lambda x: [s.strip() for s in x.split(",") if s.strip()],
    # default=[],
    # help="逗号分隔的 JSON 路径列表，如: path1.json,path2.json"
    # )
    # config原有参数，不传入则使用默认值
    parser.add_argument("--data_path",type=str,default=argparse.SUPPRESS,help="原始训练数据,如果是文件名就在数据文件夹查找,是绝对路径就")
    parser.add_argument("--valid_data_path",type=str,default=argparse.SUPPRESS,help="测试集数据")
    parser.add_argument("--model_name", type=str, default=argparse.SUPPRESS, help="本次训练使用的模型名称base_ppo,ppo_continue_action,dqn")
    parser.add_argument("--optim_name", type=str, default=argparse.SUPPRESS, help="优化器设置")
    parser.add_argument("--loss_name", type=str, default=argparse.SUPPRESS, help="policy损失函数的名称")
    parser.add_argument("--value_loss_name", type=str, default=argparse.SUPPRESS, help="value模型的损失函数名称")
    # 训练参数,若不传入,则采用trainer/trainer_conf中的默认值
    parser.add_argument("--k_epochs", type=int, default=argparse.SUPPRESS, help="每个episode中采用的epoch的轮次")
    parser.add_argument("--max_episodes", type=int, default=argparse.SUPPRESS, help="最大episodes轮次")
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS, help="学习率")
    parser.add_argument("--norm_clip", type=float, default=argparse.SUPPRESS, help="归一化时的梯度裁剪")
    parser.add_argument("--clip_grad", type=float, default=argparse.SUPPRESS, help="反向传播的梯度裁剪")
    parser.add_argument("--clip_grad_decay", type=float, default=argparse.SUPPRESS, help="梯度裁剪的衰减")
    parser.add_argument("--print_every", type=int, default=argparse.SUPPRESS, help="loss打印的间隔")
    parser.add_argument("--sample", type=float, default=argparse.SUPPRESS, help="数据的采样比例")
    parser.add_argument("--l2", type=float, default=argparse.SUPPRESS, help="l2正则的数")
    parser.add_argument("--save_every_eposide", type=int, default=argparse.SUPPRESS, help="保存模型的间隔")

    parser.add_argument("--use_state_norm", type=str2bool, default=argparse.SUPPRESS, help="是否使用state归一化")
    parser.add_argument("--use_discount_reward_norm", type=str2bool, default=argparse.SUPPRESS, help="是否使用reward归一化")
    parser.add_argument("--center", type=str2bool, default=argparse.SUPPRESS, help="reward归一化相关参数")
    parser.add_argument("--scale", type=str2bool, default=argparse.SUPPRESS, help="scale归一化相关参数")
    parser.add_argument("--use_checkpoint", type=str2bool, default=False, help="是否采用checkpoint训练")
    parser.add_argument("--checkpoint_path", type=str, default=argparse.SUPPRESS, help="checkpoint路径")
    parser.add_argument("--continue_mode", type=str, default='pretrain', help="继续训练模式，resume或pretrain")
    # parser.add_argument("--checkpoint_name_value", type=str, default='', help="value的checkpoint路径")
    parser.add_argument("--action_ls",type=lambda x: list(map(float, x.split(","))),default=argparse.SUPPRESS,help="标签的映射关系,一般都比较长,不建议在args中传入")

    args = parser.parse_args()
    args.base_dir = str(Path(__file__).parents[0])  # project dir
    print(str(Path(__file__).parents[0]))
    print(args.base_dir)
    return args

def run_one_process_curriculum(rank, world_size, config_ls, master_addr, master_port):
    """课程学习版本的训练入口"""
    try:
        setup_cpu(rank, world_size, master_addr, master_port)
        
        # 为所有 config 设置分布式参数
        for config in config_ls:
            config.distributed = world_size > 1
            config.rank = rank
            config.device = "cpu"
            config.world_size = world_size
        
        trainer = Trainer(config_ls)
        trainer.train_curriculum()
    except Exception as e:
        logging_once(f"Error in process {rank}: {e}", logging.CRITICAL)
        raise e
    finally:
        cleanup()

def run_one_process_curriculum_new(rank, world_size, config, master_addr, master_port):
    """课程学习版本的训练入口"""
    try:
        setup_cpu(rank, world_size, master_addr, master_port)
        
        # 为所有 config 设置分布式参数
        
        config.distributed = world_size > 1
        config.rank = rank
        config.device = "cpu"
        config.world_size = world_size
        
        trainer = Trainer(config)
        trainer.train_curriculum()
    except Exception as e:
        logging_once(f"Error in process {rank}: {e}", logging.CRITICAL)
        raise e
    finally:
        cleanup()

def main():
    # # 初始化args参数
    # args = arg_parser()
    # # 初始化task参数
    # config = task_dict[args.task_name](args.json_path)
    # if args.json_path:
    #     config.update(read_json(args.json_path), priority="high")
    # config.update(args, priority="high")
    # config.update(TrainerConfig(), priority="low")
    # config.update(modelconfig_dict[config.model_name](), priority="low")

    # config.initialize()
    # # processes_per_node = args.num_processes  # 每个节点的进程数
    # config.world_size = config.num_nodes * config.num_processes  # 总进程数 = 节点数 × 每节点进程数
    # config.node_rank_start = config.node_rank * config.num_processes  # 当前节点的起始进程rank

    # print(f"Node {config.node_rank}/{config.num_nodes}: Starting training with {config.num_processes} processes")
    # print(f"Global ranks from {config.node_rank_start} to {config.node_rank_start + config.num_processes - 1}")
    # print(f"Process num {config.world_size} using CPU")
    # # 打印config
    # logging_once(config, logging.CRITICAL)
    # print(f"😺😺state维度:{config.state_dim}")

    # if config.world_size == 1:
    #     run_one_process(0, 1, config, config.master_addr, config.master_port)  # 单进程训练
    # else:
    #     processes = []  # 多进程训练 - 每个节点只启动自己负责的那部分进程
    #     for local_rank in range(config.num_processes):
    #         global_rank = config.node_rank_start + local_rank  # 计算全局rank
    #         p = mp.Process(
    #             target=run_one_process,
    #             args=(global_rank, config.world_size, config, config.master_addr, config.master_port),
    #         )
    #         p.start()
    #         processes.append(p)

    #     for p in processes:  # 等待所有进程完成
    #         p.join()
    
    # # 初始化args参数
    # args = arg_parser()
    # # 课程学习：解析多个 JSON 文件生成 config 列表
    # config_ls = []
    # for idx, json_path in enumerate(args.json_path_ls):
    #     config = task_dict[args.task_name](json_path)
    #     if json_path:
    #         config.update(read_json(json_path), priority="high")
    #     config.update(args, priority="high")
    #     config.update(TrainerConfig(), priority="low")
    #     config.update(modelconfig_dict[config.model_name](), priority="low")
        
    #     # 为每个阶段设置标识
    #     config.stage_idx = idx
    #     config.total_stages = len(args.json_path_ls)
        
    #     config.initialize()
    #     config_ls.append(config)
    
    # # 使用第一个 config 初始化分布式相关参数
    # base_config = config_ls[0]
    # base_config.world_size = base_config.num_nodes * base_config.num_processes
    # base_config.node_rank_start = base_config.node_rank * base_config.num_processes

    # if base_config.world_size == 1:
    #     run_one_process_curriculum(0, 1, config_ls, base_config.master_addr, base_config.master_port)
    # else:
    #     processes = []
    #     for local_rank in range(base_config.num_processes):
    #         global_rank = base_config.node_rank_start + local_rank
    #         p = mp.Process(
    #             target=run_one_process_curriculum,
    #             args=(global_rank, base_config.world_size, config_ls, 
    #                   base_config.master_addr, base_config.master_port),
    #         )
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    

    # 初始化args参数
    args = arg_parser()
    # 初始化task参数
    config = task_dict[args.task_name](args.json_path)
    if args.json_path:
        config.update(read_json(args.json_path), priority="high")
    config.update(args, priority="high")
    config.update(TrainerConfig(), priority="low")

    if args.data_path:
        for stage in config.curriculum_stages:
            stage['data_path'] = args.data_path
    

    config.initialize()
    # processes_per_node = args.num_processes  # 每个节点的进程数
    config.world_size = config.num_nodes * config.num_processes  # 总进程数 = 节点数 × 每节点进程数
    config.node_rank_start = config.node_rank * config.num_processes  # 当前节点的起始进程rank

    print(f"Node {config.node_rank}/{config.num_nodes}: Starting training with {config.num_processes} processes")
    print(f"Global ranks from {config.node_rank_start} to {config.node_rank_start + config.num_processes - 1}")
    print(f"Process num {config.world_size} using CPU")
    # 打印config
    logging_once(config, logging.CRITICAL)
    print(f"😺😺state维度:{config.state_dim}")

    if config.world_size == 1:
        run_one_process_curriculum_new(0, 1, config, config.master_addr, config.master_port)  # 单进程训练
    else:
        processes = []  # 多进程训练 - 每个节点只启动自己负责的那部分进程
        for local_rank in range(config.num_processes):
            global_rank = config.node_rank_start + local_rank  # 计算全局rank
            p = mp.Process(
                target=run_one_process_curriculum_new,
                args=(global_rank, config.world_size, config, config.master_addr, config.master_port),
            )
            p.start()
            processes.append(p)

        for p in processes:  # 等待所有进程完成
            p.join()


if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method("spawn", force=True)
    main()