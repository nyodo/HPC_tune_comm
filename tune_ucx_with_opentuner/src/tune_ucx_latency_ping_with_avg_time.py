#!/usr/bin/env python
# -*- coding: utf-8 -*-

import opentuner
from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter
from opentuner.measurement import MeasurementInterface
from opentuner.resultsdb.models import Result
import subprocess
import re
import os
import glob
import sys
import time
import json
import datetime

# 数据处理函数 - 使用 subprocess 调用
def process_raw_data(raw_file_dir, output_file):
    """通过 subprocess 调用 process.py 处理数据"""
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'process', 'process.py')
        
        if not os.path.exists(script_path):
            print(f"Warning: Process script not found at {script_path}")
            return None
        
        print(f">>> Calling: {sys.executable} {script_path}")
        result = subprocess.run(
            [sys.executable, script_path, raw_file_dir, output_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # 显示处理输出
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"Warning: Data processing failed (exit code: {result.returncode})")
            if result.stderr:
                print(result.stderr)
            return None
    except Exception as e:
        print(f"Warning: Could not process data: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 新增：双重日志输出辅助类 ---
class DualLogger(object):
    """
    将标准输出同时定向到 控制台 和 文件
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class UCXTraceTuner(MeasurementInterface):
    def __init__(self, *args, **kwargs):
        # 初始化父类
        super(UCXTraceTuner, self).__init__(*args, **kwargs)
        
        # === 1. 读取配置文件并初始化所有配置 ===
        config_path = os.path.join("config", "config.json")
        
        # 设置默认值
        self.script_path = "src/scripts/run_latency_2_intra-Blade.sh"
        self.result_dir = "result/result_2_Intra-Blade"
        self.opentuner_log_dir = "tune_result_avg"
        self.max_tests = None
        self.save_opentuner_log = True
        self.save_shell_output = True
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # === 自动处理原始数据 ===
                benchmark_cfg = config.get("benchmark", {})
                raw_file_dir = benchmark_cfg.get("raw_file_dir")
                csv_file = benchmark_cfg.get("csv_file")
                
                if raw_file_dir and csv_file:
                    # 检查是否需要处理数据
                    need_process = False
                    
                    # 如果输出文件不存在，需要处理
                    if not os.path.exists(csv_file):
                        need_process = True
                        print(f"[Init] Output CSV not found, will process raw data")
                    # 如果原始数据目录比输出文件新，需要重新处理
                    elif os.path.exists(raw_file_dir):
                        raw_mtime = os.path.getmtime(raw_file_dir)
                        csv_mtime = os.path.getmtime(csv_file)
                        if raw_mtime > csv_mtime:
                            need_process = True
                            print(f"[Init] Raw data is newer than CSV, will reprocess")
                    
                    if need_process:
                        print(f"\n{'='*70}")
                        print("[Init] Auto-processing raw data...")
                        print(f"{'='*70}")
                        try:
                            process_raw_data(raw_file_dir, csv_file)
                            print(f"{'='*70}")
                            print("[Init] Data processing completed!")
                            print(f"{'='*70}\n")
                        except Exception as e:
                            print(f"[Init] Warning: Data processing failed: {e}")
                            print(f"[Init] Will try to use existing CSV file if available")
                    else:
                        print(f"[Init] Using existing processed data: {csv_file}")
                
                # 读取 script_shell 配置
                script_shell_cfg = config.get("script_shell", {})
                self.script_path = script_shell_cfg.get("script_path", self.script_path)
                self.result_dir = script_shell_cfg.get("result_dir", self.result_dir)
                self.save_shell_output = script_shell_cfg.get("save_shell_output", self.save_shell_output)
                
                print(f"[Init] Script path: {self.script_path}")
                print(f"[Init] Result directory: {self.result_dir}")
                print(f"[Init] Shell output: {'enabled' if self.save_shell_output else 'disabled'}")
                
                # 读取 opentuner 配置
                opentuner_cfg = config.get("opentuner", {})
                self.max_tests = opentuner_cfg.get("test_limit", None)
                self.opentuner_log_dir = opentuner_cfg.get("opentuner_log_dir", self.opentuner_log_dir)
                self.save_opentuner_log = opentuner_cfg.get("save_opentuner_log", self.save_opentuner_log)
                
                if self.max_tests and self.max_tests > 0:
                    print(f"[Init] Strict test limit enabled: {self.max_tests} tests")
                
                print(f"[Init] OpenTuner log directory: {self.opentuner_log_dir}")
                print(f"[Init] OpenTuner log: {'enabled' if self.save_opentuner_log else 'disabled'}")
                
            except Exception as e:
                print(f"[Init] Warning: Could not read config.json: {e}")
                print(f"[Init] Using default values")
        
        # === 2. 配置日志系统 ===
        if self.save_opentuner_log:
            self.setup_logging()
        else:
            print("[Init] OpenTuner logging to console only (file logging disabled)")

        # 初始化计数器
        self.call_count = 0
        
        # 基准测试相关变量
        self.default_time = None
        self.best_time_so_far = float('inf')
        self.best_round = -1
        
        # 在初始化时直接运行一次基准测试
        print("\n[Init] Running Baseline Test (Default Parameters)...")
        self.run_baseline()

    def setup_logging(self):
        """
        根据脚本名称创建目录并重定向输出
        """
        # 1. 提取场景名称: run_latency_2_intra-Blade.sh -> 2_intra-Blade
        script_name = os.path.basename(self.script_path)
        # 移除前缀 run_latency_ 和后缀 .sh (简单的字符串处理)
        scenario_name = script_name.replace("run_latency_", "").replace(".sh", "")
        # 或者更简单的: 直接用文件名作为目录名
        # scenario_name = os.path.splitext(script_name)[0]

        # 2. 构建目录路径: 从配置文件读取基础目录
        base_dir = self.opentuner_log_dir
        log_dir = os.path.join(base_dir, scenario_name)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 3. 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"opentuner_log_{timestamp}.txt"
        log_filepath = os.path.join(log_dir, log_filename)

        # 4. 重定向 sys.stdout
        # 只有在还没有重定向过的时候才执行，防止多次初始化导致嵌套
        if not isinstance(sys.stdout, DualLogger):
            sys.stdout = DualLogger(log_filepath)
        
        print(f"--> Log file created at: {log_filepath}")

    def run_baseline(self):
        """
        运行不带任何 UCX 环境变量的基准测试
        """
        # 创建一个干净的环境变量副本
        base_env = os.environ.copy()
        keys_to_remove = ['UCX_RNDV_THRESH', 'UCX_BCOPY_THRESH', 'UCX_ZCOPY_THRESH']
        for key in keys_to_remove:
            if key in base_env:
                del base_env[key]
        
        avg_time = self._run_benchmark(base_env, is_baseline=True)
        
        if avg_time and avg_time != float('inf'):
            self.default_time = avg_time
            print(f"[Init] Baseline (Default) Time: {self.default_time:.2f} us")
            print("-----------------------------------------------------------\n")
        else:
            print("[Init] Error: Failed to measure baseline time! Optimization ratio will be N/A.")
            self.default_time = None

    def _run_benchmark(self, run_env, is_baseline=False):
        """
        内部辅助函数：执行 Shell 脚本并解析结果
        """
        # 使用配置文件中的 result_dir
        log_dir_parse = self.result_dir
        
        if not os.path.exists(log_dir_parse):
            os.makedirs(log_dir_parse, exist_ok=True)
            
        execution_times = []
        
        # 运行 5 次取最小值/平均值
        for i in range(5):
            try:
                # 根据配置决定是否保存 Shell 日志
                run_env_with_flag = run_env.copy()
                if not self.save_shell_output:
                    # 设置环境变量通知 Shell 脚本不保存日志
                    run_env_with_flag['UCX_DISABLE_SHELL_LOG'] = '1'
                
                subprocess.run(["bash", self.script_path], env=run_env_with_flag, check=True, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 解析最新日志
                list_of_files = glob.glob(f'{log_dir_parse}/*.log')
                if not list_of_files:
                    return float('inf')
                
                latest_file = max(list_of_files, key=os.path.getctime)
                time_val = self.parse_log_file(latest_file)
                
                if time_val is not None:
                    execution_times.append(time_val)
                else:
                    execution_times.append(float('inf'))
                
                # 如果禁用了日志保存，解析完后立即删除临时文件
                if not self.save_shell_output:
                    try:
                        os.remove(latest_file)
                    except Exception as e:
                        pass  # 删除失败也不影响主流程
                    
            except Exception as e:
                if not is_baseline:
                    print(f"Error: {e}")
                return float('inf')

        valid_times = [t for t in execution_times if t != float('inf')]
        if valid_times:
            # 策略：保持使用 Min Time (最小值) 作为优化依据
            return min(valid_times) 
        else:
            return float('inf')

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        # === 使用指数范围 (Exponent) ===
        manipulator.add_parameter(IntegerParameter('UCX_RNDV_THRESH_EXP', 10, 19))
        manipulator.add_parameter(IntegerParameter('UCX_BCOPY_THRESH_EXP', 8, 17))
        manipulator.add_parameter(IntegerParameter('UCX_ZCOPY_THRESH_EXP', 9, 18))
        return manipulator

    def run(self, desired_result, input, limit):
        self.call_count += 1
        
        # 严格限制检查：如果超过配置的 test_limit，立即停止
        if self.max_tests and self.max_tests > 0 and self.call_count > self.max_tests:
            print(f"\n===========================================================")
            print(f">>> STRICT LIMIT: Test #{self.call_count} skipped (limit: {self.max_tests})")
            print(f">>> OpenTuner may try more tests, but they will all be skipped.")
            print(f"===========================================================")
            return Result(time=float('inf'), state='TIMEOUT')
        
        cfg = desired_result.configuration.data
        
        # 将指数还原为实际数值
        val_rndv = 1 << cfg['UCX_RNDV_THRESH_EXP']
        val_bcopy = 1 << cfg['UCX_BCOPY_THRESH_EXP']
        val_zcopy = 1 << cfg['UCX_ZCOPY_THRESH_EXP']

        # 设置环境变量
        run_env = os.environ.copy()
        run_env['UCX_RNDV_THRESH'] = str(val_rndv)
        run_env['UCX_BCOPY_THRESH'] = str(val_bcopy)
        run_env['UCX_ZCOPY_THRESH'] = str(val_zcopy)
        
        print(f"===========================================================")
        print(f"[Round {self.call_count}] Testing: RNDV={val_rndv}, BCOPY={val_bcopy}, ZCOPY={val_zcopy}")

        # 调用通用测试函数
        min_time = self._run_benchmark(run_env)
        
        if min_time != float('inf'):
            # === 计算优化率 ===
            improvement_str = "N/A"
            if self.default_time:
                # 优化率公式
                ratio = (self.default_time - min_time) / self.default_time * 100.0
                if ratio > 0:
                    improvement_str = f"+{ratio:.2f}% (Faster)"
                else:
                    improvement_str = f"{ratio:.2f}% (Slower)"

            # 更新当前最佳记录
            if min_time < self.best_time_so_far:
                self.best_time_so_far = min_time
                self.best_round = self.call_count
            
            print(f"   -> Time: {min_time:.2f} us | vs Default: {improvement_str}")
            
            # 检查是否达到限制（在完成当前测试后）
            if self.max_tests and self.max_tests > 0 and self.call_count >= self.max_tests:
                print(f"\n{'='*60}")
                print(f">>> Test limit reached! Completed {self.call_count}/{self.max_tests} tests.")
                print(f">>> Best result found in Round {self.best_round}: {self.best_time_so_far:.2f} us")
                print(f"{'='*60}\n")
            
            return Result(time=min_time)
        else:
            print("   -> Error in execution")
            return Result(time=float('inf'), state='ERROR')

    def parse_log_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                match = re.search(r"Total Execution Time:\s+([\d\.]+)\s+us", content, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except:
            pass
        return None

    def save_final_config(self, configuration):
        """保存最佳配置并输出对比报告"""
        cfg = configuration.data
        
        final_values = {
            "UCX_RNDV_THRESH": 1 << cfg['UCX_RNDV_THRESH_EXP'],
            "UCX_BCOPY_THRESH": 1 << cfg['UCX_BCOPY_THRESH_EXP'],
            "UCX_ZCOPY_THRESH": 1 << cfg['UCX_ZCOPY_THRESH_EXP'],
            "Raw_Exponents": cfg 
        }
        
        print("\n\n###########################################################")
        print("                 OPTIMIZATION SUMMARY                      ")
        print("###########################################################")
        print(f" Total Tests Executed    : {self.call_count}")
        if self.max_tests and self.max_tests > 0:
            print(f" Configured Limit        : {self.max_tests}")
        print(f" Best Parameters Found in [Round {self.best_round}]:")
        print(f"   RNDV : {final_values['UCX_RNDV_THRESH']}")
        print(f"   BCOPY: {final_values['UCX_BCOPY_THRESH']}")
        print(f"   ZCOPY: {final_values['UCX_ZCOPY_THRESH']}")
        print("-----------------------------------------------------------")
        
        # 安全处理 None 值和无效值
        if self.default_time is not None:
            print(f" Baseline (Default) Time : {self.default_time:.2f} us")
        else:
            print(f" Baseline (Default) Time : N/A (no baseline data)")
        
        if self.best_time_so_far != float('inf'):
            print(f" Optimized (Best) Time   : {self.best_time_so_far:.2f} us")
        else:
            print(f" Optimized (Best) Time   : N/A (no valid results)")
        
        # 计算改进指标
        speedup = None
        improvement = None
        if self.default_time is not None and self.best_time_so_far != float('inf') and self.best_time_so_far > 0:
            speedup = self.default_time / self.best_time_so_far
            improvement = (self.default_time - self.best_time_so_far) / self.default_time * 100.0
            print(f" Speedup                 : {speedup:.2f}x")
            print(f" Improvement             : {improvement:.2f}%")
        else:
            print(f" Speedup                 : N/A")
            print(f" Improvement             : N/A")
        
        print("###########################################################\n")
        
        final_values['metrics'] = {
            'default_time_us': self.default_time if self.default_time is not None else None,
            'optimized_time_us': self.best_time_so_far if self.best_time_so_far != float('inf') else None,
            'improvement_percent': improvement if improvement is not None else None,
            'speedup': speedup if speedup is not None else None,
            'best_round': self.best_round,
            'total_tests': self.call_count,
            'test_limit': self.max_tests if self.max_tests else 'unlimited'
        }
        
        # 将 JSON 结果也保存到日志目录中一份，方便归档
        # 获取当前 DualLogger 中的文件路径有点麻烦，所以这里直接保存在当前目录
        # 也可以修改 DualLogger 让它暴露 log path
        
        os.makedirs("outputs", exist_ok=True)
        with open(os.path.join("outputs", "ucx_best_params.json"), "w") as f:
            json.dump(final_values, f, indent=4)

if __name__ == '__main__':
    if not os.environ.get("SLURM_JOB_ID"):
        # 不在 SLURM 环境中，需要申请节点
        print(">>> Not in SLURM environment, allocating nodes...")
        
        config_path = os.path.join("config", "config.json")
        if not os.path.exists(config_path):
            print(f"Error: Missing {config_path}")
            sys.exit(1)
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        cluster_cfg = config.get("cluster", {})
        partition = cluster_cfg.get("partition", "thcp3")
        nodes = cluster_cfg.get("nodes", [])
        
        if len(nodes) < 2:
            print("Error: config.json must define at least 2 nodes")
            sys.exit(1)
        
        nodelist = ",".join(nodes[:2])
        print(f">>> Requesting nodes: {nodelist} on partition {partition}")
        
        # 从 config.json 读取 OpenTuner 参数
        opentuner_cfg = config.get("opentuner", {})
        test_limit = opentuner_cfg.get("test_limit", None)
        no_dups = opentuner_cfg.get("no_dups", False)
        
        # 构建命令行参数
        extra_args = []
        
        # 添加 test_limit 参数
        if test_limit and test_limit > 0:
            extra_args.extend(["--test-limit", str(test_limit)])
            print(f">>> Config: test_limit = {test_limit}")
        else:
            print(f">>> Config: test_limit = unlimited")
        
        # 添加 no_dups 参数
        if no_dups:
            extra_args.append("--no-dups")
            print(f">>> Config: no_dups = enabled")
        else:
            print(f">>> Config: no_dups = disabled")
        
        # 构建 salloc 命令
        cmd = [
            "salloc",
            "-p",
            partition,
            "-N",
            "2",
            "--nodelist",
            nodelist,
            sys.executable,
            os.path.abspath(__file__),
        ] + extra_args
        
        print(f">>> Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    
    # 在 SLURM 环境中运行 OpenTuner
    print(f">>> Running OpenTuner in SLURM job {os.environ.get('SLURM_JOB_ID')}")
    print(f">>> Allocated nodes: {os.environ.get('SLURM_JOB_NODELIST', 'N/A')}")
    
    argparser = opentuner.default_argparser()
    UCXTraceTuner.main(argparser.parse_args())
