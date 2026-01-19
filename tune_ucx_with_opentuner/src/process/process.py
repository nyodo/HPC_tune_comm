#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理脚本：将原始通信日志数据转换为统计汇总数据

功能：
1. 读取 raw_file_dir 目录下所有 log-*.csv 文件
2. 计算 total_size = sendsize * sendcount
3. 按 comm_type 和 total_size 分组统计出现次数（count）
4. 在同一 comm_type 下按 total_size 排序并添加 index
5. 输出到指定的 csv_file
"""

import pandas as pd
import glob
import os
import json
import sys


def read_all_log_files(raw_file_dir):
    """
    读取指定目录下所有 log-*.csv 文件
    
    Args:
        raw_file_dir: 原始数据目录路径
        
    Returns:
        合并后的 DataFrame
    """
    print(f">>> Reading data from: {raw_file_dir}")
    
    # 查找所有 log-*.csv 文件
    pattern = os.path.join(raw_file_dir, "log-*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No log-*.csv files found in {raw_file_dir}")
    
    print(f">>> Found {len(files)} log files")
    
    # 读取所有文件并合并
    dfs = []
    for file in sorted(files):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"    - Loaded: {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            print(f"    ! Error reading {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid data loaded from log files")
    
    # 合并所有数据
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f">>> Total rows loaded: {len(merged_df)}")
    
    return merged_df


def calculate_total_size(df):
    """
    计算 total_size = sendsize * sendcount
    
    Args:
        df: 原始数据 DataFrame
        
    Returns:
        添加了 total_size 列的 DataFrame
    """
    print(">>> Calculating total_size = sendsize * sendcount")
    
    # 计算 total_size
    df['total_size'] = df['sendsize'] * df['sendcount']
    
    print(f">>> total_size range: [{df['total_size'].min()}, {df['total_size'].max()}]")
    
    return df


def aggregate_by_comm_type_and_size(df):
    """
    按 comm_type 和 total_size 分组统计
    
    Args:
        df: 包含 total_size 的 DataFrame
        
    Returns:
        聚合后的 DataFrame，包含 comm_type, total_size, count
    """
    print(">>> Aggregating by comm_type and total_size")
    
    # 按 comm_type 和 total_size 分组统计
    aggregated = df.groupby(['comm_type', 'total_size']).size().reset_index(name='count')
    
    print(f">>> Unique comm_types: {aggregated['comm_type'].nunique()}")
    print(f">>> Total unique (comm_type, total_size) pairs: {len(aggregated)}")
    
    return aggregated


def add_index_per_comm_type(df):
    """
    在每个 comm_type 内按 total_size 排序并添加 index
    
    Args:
        df: 聚合后的 DataFrame
        
    Returns:
        添加了 index 列的 DataFrame
    """
    print(">>> Adding index within each comm_type")
    
    # 按 comm_type 分组，然后按 total_size 排序
    df = df.sort_values(['comm_type', 'total_size'])
    
    # 在每个 comm_type 内添加 index（从 1 开始）
    df['index'] = df.groupby('comm_type').cumcount() + 1
    
    # 调整列顺序：comm_type, index, total_size, count
    df = df[['comm_type', 'index', 'total_size', 'count']]
    
    # 显示每个 comm_type 的统计信息
    for comm_type in df['comm_type'].unique():
        count = len(df[df['comm_type'] == comm_type])
        print(f"    - comm_type {comm_type}: {count} unique total_sizes")
    
    return df


def save_processed_data(df, output_file):
    """
    保存处理后的数据到 CSV 文件
    
    Args:
        df: 处理后的 DataFrame
        output_file: 输出文件路径
    """
    print(f">>> Saving processed data to: {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"    - Created directory: {output_dir}")
    
    # 保存到 CSV
    df.to_csv(output_file, index=False)
    print(f">>> Successfully saved {len(df)} rows")
    print(f">>> Output columns: {list(df.columns)}")


def process_raw_data(raw_file_dir, output_file):
    """
    完整的数据处理流程
    
    Args:
        raw_file_dir: 原始数据目录
        output_file: 输出文件路径
    """
    print("="*70)
    print("Starting data processing pipeline")
    print("="*70)
    
    try:
        # 步骤 1: 读取所有 log 文件
        df = read_all_log_files(raw_file_dir)
        
        # 步骤 2: 计算 total_size
        df = calculate_total_size(df)
        
        # 步骤 3: 按 comm_type 和 total_size 聚合
        df = aggregate_by_comm_type_and_size(df)
        
        # 步骤 4: 添加 index 并排序
        df = add_index_per_comm_type(df)
        
        # 步骤 5: 保存处理后的数据
        save_processed_data(df, output_file)
        
        print("="*70)
        print("Data processing completed successfully!")
        print("="*70)
        
        return df
        
    except Exception as e:
        print(f"\n!!! Error during processing: {e}")
        import traceback
        traceback.print_exc()
        raise


def process_from_config(config_path="config/config.json"):
    """
    从配置文件读取参数并处理数据
    
    Args:
        config_path: 配置文件路径
    """
    print(f">>> Loading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 从配置文件读取参数
    benchmark_cfg = config.get('benchmark', {})
    raw_file_dir = benchmark_cfg.get('raw_file_dir')
    csv_file = benchmark_cfg.get('csv_file')
    
    if not raw_file_dir:
        raise ValueError("'raw_file_dir' not found in config file")
    
    if not csv_file:
        raise ValueError("'csv_file' not found in config file")
    
    print(f">>> Raw data directory: {raw_file_dir}")
    print(f">>> Output CSV file: {csv_file}")
    print()
    
    # 执行数据处理
    result_df = process_raw_data(raw_file_dir, csv_file)
    
    return result_df


def main():
    """
    主函数：支持命令行调用
    
    用法：
        python process.py                          # 使用默认配置文件
        python process.py config/custom.json       # 使用指定配置文件
        python process.py <raw_dir> <output_file>  # 直接指定输入输出路径
    """
    if len(sys.argv) == 1:
        # 无参数：使用默认配置文件
        process_from_config()
        
    elif len(sys.argv) == 2:
        # 一个参数：配置文件路径
        config_path = sys.argv[1]
        process_from_config(config_path)
        
    elif len(sys.argv) == 3:
        # 两个参数：原始数据目录和输出文件
        raw_file_dir = sys.argv[1]
        output_file = sys.argv[2]
        process_raw_data(raw_file_dir, output_file)
        
    else:
        print("Usage:")
        print("  python process.py")
        print("  python process.py <config_file>")
        print("  python process.py <raw_data_dir> <output_csv>")
        sys.exit(1)


if __name__ == '__main__':
    main()
