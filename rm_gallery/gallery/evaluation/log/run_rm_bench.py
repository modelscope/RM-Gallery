#!/usr/bin/env python3
"""
RM-Bench评估运行脚本
支持DeepSeek-Chat模型在RM-Bench上的评估
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluations.rm_bench_evaluator import RMBenchEvaluator


def setup_logging(log_level="INFO"):
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 创建日志文件
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rm_bench_{timestamp}.log"
    
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB"
    )
    
    return str(log_file)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行RM-Bench评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认运行DeepSeek-Chat评估
  python scripts/evaluation/run_rm_bench.py
  
  # 指定模型配置文件
  python scripts/evaluation/run_rm_bench.py --model configs/models/gpt-3.5-turbo.yaml
  
  # 运行少量样本测试
  python scripts/evaluation/run_rm_bench.py --max-samples 10
  
  # 使用8个线程并行评估
  python scripts/evaluation/run_rm_bench.py --max-workers 8
  
  python scripts/evaluation/run_rm_bench.py --max-workers 32 --model configs/models/qwen2.5-14b-instruct.yaml
  
  # 指定输出目录
  python scripts/evaluation/run_rm_bench.py --output-dir data/outputs/rm_bench_test
        """
    )
    
    parser.add_argument(
        "--model",
        default="configs/models/deepseek-chat.yaml",
        help="模型配置文件路径 (默认: configs/models/deepseek-chat.yaml)"
    )
    
    parser.add_argument(
        "--data",
        default="benchmarks/RM-Bench/total_dataset.json",
        help="数据集路径 (默认: benchmarks/RM-Bench/total_dataset.json)"
    )
    
    parser.add_argument(
        "--template",
        default="rm_bench",
        help="Prompt模板名称 (默认: rm_bench)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data/outputs/rm_bench",
        help="输出目录 (默认: data/outputs/rm_bench)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大评估样本数 (默认: 全部样本)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="最大线程数 (默认: 4)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式，只加载数据和模型，不进行实际评估"
    )
    
    return parser.parse_args()


def validate_paths(args):
    """验证路径是否存在"""
    errors = []
    
    # 检查模型配置文件
    if not Path(args.model).exists():
        errors.append(f"模型配置文件不存在: {args.model}")
    
    # 检查数据文件
    if not Path(args.data).exists():
        errors.append(f"数据文件不存在: {args.data}")
    
    # 检查模板目录
    template_dir = Path("templates/benchmarks")
    if not template_dir.exists():
        errors.append(f"模板目录不存在: {template_dir}")
    
    if errors:
        logger.error("路径验证失败:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True


def print_config(args):
    """打印配置信息"""
    print("=" * 60)
    print("🚀 RM-Bench评估配置")
    print("=" * 60)
    print(f"模型配置:    {args.model}")
    print(f"数据集:      {args.data}")
    print(f"模板:        {args.template}")
    print(f"输出目录:    {args.output_dir}")
    print(f"最大样本数:  {args.max_samples or '全部'}")
    print(f"最大线程数:  {args.max_workers}")
    print(f"日志级别:    {args.log_level}")
    print(f"干运行:      {args.dry_run}")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    log_file = setup_logging(args.log_level)
    logger.info(f"日志文件: {log_file}")
    
    # 打印配置
    print_config(args)
    
    # 验证路径
    if not validate_paths(args):
        sys.exit(1)
    
    try:
        # 创建评估器
        logger.info("初始化RM-Bench评估器...")
        evaluator = RMBenchEvaluator(
            model_config_path=args.model,
            template_name=args.template,
            data_path=args.data,
            max_workers=args.max_workers
        )
        
        logger.info("✅ 评估器初始化成功")
        
        # 如果是干运行，就此结束
        if args.dry_run:
            logger.info("🏁 干运行模式，评估器加载成功")
            return
        
        # 运行评估
        logger.info("🔥 开始RM-Bench评估...")
        
        results = evaluator.evaluate(
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
        
        logger.info("✅ 评估完成")
        
        # 打印最终结果摘要
        print("\n" + "=" * 60)
        print("📊 评估结果摘要")
        print("=" * 60)
        
        overall = results["overall_accuracy"]
        print(f"模型: {results['model']}")
        print(f"Hard Accuracy:   {overall['hard_acc']:.4f}")
        print(f"Normal Accuracy: {overall['normal_acc']:.4f}")
        print(f"Easy Accuracy:   {overall['easy_acc']:.4f}")
        print(f"Overall Accuracy: {overall['overall_acc']:.4f}")
        
        details = results["evaluation_details"]
        print(f"\n评估统计:")
        print(f"总样本数: {details['total_samples']}")
        print(f"成功评估: {details['successful_samples']}")
        print(f"成功率: {details['success_rate']:.2%}")
        
        # 按领域显示结果
        if results["domain_accuracy"]:
            print(f"\n按领域准确率:")
            for domain, acc in results["domain_accuracy"].items():
                print(f"  {domain}: Hard={acc['hard_acc']:.4f}, Normal={acc['normal_acc']:.4f}, Easy={acc['easy_acc']:.4f}")
        
        print("=" * 60)
        
        # 保存结果到JSON文件
        output_path = Path(args.output_dir) / "rm_bench_summary.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 详细结果已保存到: {output_path}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断评估")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 评估失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 