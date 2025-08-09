#!/usr/bin/env python3
"""
Claude Voice Assistant 安装脚本
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# 确保 Python 版本兼容性
if sys.version_info < (3, 9):
    sys.exit("Claude Voice Assistant 需要 Python 3.9 或更高版本")

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 读取 README 文件
README_FILE = ROOT_DIR / "README.md"
if README_FILE.exists():
    with open(README_FILE, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Claude Voice Assistant - 基于 Claude Code Agents 模式的智能语音助手"

# 读取版本信息
VERSION_FILE = ROOT_DIR / "src" / "__init__.py"
if VERSION_FILE.exists():
    with open(VERSION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break
    else:
        version = "1.0.0"
else:
    version = "1.0.0"

# 读取依赖文件
def read_requirements(filename):
    """读取依赖文件"""
    requirements_file = ROOT_DIR / filename
    if not requirements_file.exists():
        return []
    
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # 跳过注释、空行和内置模块
            if (line and 
                not line.startswith("#") and 
                not line.startswith("-") and
                ";" not in line and  # 跳过平台特定依赖说明
                not any(builtin in line for builtin in [
                    "asyncio", "os", "sys", "json", "time", "datetime",
                    "pathlib", "uuid", "hashlib", "base64", "subprocess",
                    "threading", "multiprocessing", "queue", "platform",
                    "re", "collections", "itertools", "functools",
                    "traceback", "warnings", "pickle", "tempfile",
                    "shutil", "glob", "ctypes", "signal", "gc", "mmap",
                    "concurrent.futures", "weakref", "copy", "operator",
                    "abc", "enum", "dataclasses", "contextlib",
                    "argparse", "unittest", "urllib", "socket",
                    "gzip", "zipfile", "string", "math", "statistics",
                    "random", "secrets", "environ"
                ])):
                # 处理平台特定依赖
                if "platform_system" in line:
                    if 'platform_system=="Windows"' in line and os.name == "nt":
                        requirements.append(line.split(";")[0].strip())
                    elif 'platform_system=="Darwin"' in line and sys.platform == "darwin":
                        requirements.append(line.split(";")[0].strip())
                    elif 'platform_system=="Linux"' in line and sys.platform.startswith("linux"):
                        requirements.append(line.split(";")[0].strip())
                else:
                    requirements.append(line)
    
    return requirements

# 基础依赖
install_requires = read_requirements("requirements.txt")

# 开发依赖
dev_requires = read_requirements("requirements-dev.txt")

# 可选依赖组
extras_require = {
    "dev": dev_requires,
    "test": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.12.0",
        "coverage>=7.3.0"
    ],
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0"
    ],
    "performance": [
        "memory-profiler>=0.61.0",
        "line-profiler>=4.1.0",
        "py-spy>=0.3.14"
    ],
    "gui": [
        "tkinter",  # 通常内置，但某些Linux发行版需要单独安装
        "PyQt6>=6.5.0",
        "PySide6>=6.5.0"
    ]
}

# 所有可选依赖
extras_require["all"] = [
    dep for deps in extras_require.values() 
    for dep in deps if isinstance(dep, str)
]

# 控制台脚本
console_scripts = [
    "claude-voice=src.main:main",
    "claude-voice-gui=src.gui:main",  # 如果有GUI版本
    "claude-voice-config=src.config_tool:main",  # 配置工具
]

# 项目分类
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: User Interfaces",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Office/Business",
    "Topic :: System :: System Shells",
    "Topic :: Utilities",
    "Natural Language :: Chinese (Simplified)",
    "Natural Language :: English",
    "Environment :: Console",
    "Environment :: Win32 (MS Windows)",
    "Framework :: AsyncIO",
]

# 项目关键词
keywords = [
    "voice assistant", "speech recognition", "AI", "automation",
    "claude code", "agents", "voice control", "productivity",
    "语音助手", "语音识别", "人工智能", "自动化", "语音控制", "生产力"
]

# 项目URL
project_urls = {
    "Homepage": "https://github.com/your-username/claude-voice-assistant",
    "Bug Reports": "https://github.com/your-username/claude-voice-assistant/issues",
    "Source": "https://github.com/your-username/claude-voice-assistant",
    "Documentation": "https://claude-voice-assistant.readthedocs.io/",
    "Changelog": "https://github.com/your-username/claude-voice-assistant/blob/main/CHANGELOG.md",
}

# 数据文件
package_data = {
    "src": [
        "config/*.yaml",
        "data/*.json",
        "templates/*.txt",
        "locale/*/*.po",
        "assets/*.png",
        "assets/*.ico",
    ]
}

# 包含的数据文件
data_files = [
    ("config", ["config/default.yaml"]),
    ("docs", ["docs/README.md"]),
]

setup(
    # 基础信息
    name="claude-voice-assistant",
    version=version,
    description="基于 Claude Code Agents 模式的智能语音助手",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # 作者信息
    author="Your Name",
    author_email="your.email@example.com",
    maintainer="Your Name", 
    maintainer_email="your.email@example.com",
    
    # 项目URL
    url="https://github.com/your-username/claude-voice-assistant",
    project_urls=project_urls,
    
    # 许可证
    license="MIT",
    
    # 包信息
    packages=find_packages(where=".", exclude=["tests*", "docs*", "backup*"]),
    package_dir={"": "."},
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,
    
    # 依赖
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # 控制台脚本
    entry_points={
        "console_scripts": console_scripts,
    },
    
    # 分类和关键词
    classifiers=classifiers,
    keywords=", ".join(keywords),
    
    # 打包选项
    zip_safe=False,
    
    # 平台特定选项
    platforms=["Windows", "macOS", "Linux"],
    
    # 测试套件
    test_suite="tests",
    tests_require=extras_require["test"],
    
    # setuptools 特定选项
    setup_requires=[
        "setuptools>=45",
        "wheel>=0.29.0",
    ],
    
    # 构建配置
    options={
        "build_ext": {
            "inplace": True,
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
        "bdist_wheel": {
            "universal": False,  # 不是纯Python包（有平台特定依赖）
        },
    },
    
    # 命令类（如果需要自定义命令）
    cmdclass={},
    
    # 资源文件处理
    resource_files=[
        ("share/claude-voice-assistant/config", ["config/default.yaml"]),
        ("share/claude-voice-assistant/docs", ["docs/*.md"]),
    ] if os.name != "nt" else [],  # Windows 不需要这些
)

# 安装后提示信息
def print_post_install_message():
    """打印安装后信息"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   Claude Voice Assistant                     ║
    ║                      安装完成！                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  快速开始:                                                   ║
    ║    claude-voice --help                  # 查看帮助          ║
    ║    claude-voice-config                  # 配置工具          ║
    ║    claude-voice                         # 启动助手          ║
    ║                                                              ║
    ║  配置文件位置:                                               ║
    ║    ~/.claude-voice-assistant/config.yaml                    ║
    ║                                                              ║
    ║  文档: https://claude-voice-assistant.readthedocs.io/       ║
    ║  问题反馈: https://github.com/your-username/issues          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    # 如果是直接运行安装
    if "install" in sys.argv:
        import atexit
        atexit.register(print_post_install_message)