"""
MT3000 平台配置加载模块
供所有模块使用
"""
import os
import json


# 配置文件路径
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

# 缓存配置，避免重复读取
_config_cache = None


def load_config():
    """加载配置文件（带缓存）"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    
    try:
        with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
            _config_cache = json.load(f)
            return _config_cache
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {_CONFIG_FILE}")
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {e}")


def get_mt3000_client_config():
    """获取 MT3000 客户端配置"""
    config = load_config()
    return config.get('mt3000_client', {})


def get_collection_config(module_name):
    """
    获取指定数据采集模块的配置
    
    Args:
        module_name: 模块名称，如 'io', 'communication', 'compute', 'memory'
    
    Returns:
        dict: 模块配置字典
    """
    config = load_config()
    collection_config = config.get('collection', {})
    return collection_config.get(module_name, {})


def get_remote_paths_config(module_name):
    """
    获取指定模块的远程路径配置
    
    Args:
        module_name: 模块名称，如 'io', 'communication', 'compute', 'memory'
    
    Returns:
        dict: 远程路径配置字典
    """
    module_config = get_collection_config(module_name)
    return module_config.get('remote_paths', {})


def reload_config():
    """重新加载配置文件（清除缓存）"""
    global _config_cache
    _config_cache = None
    return load_config()
