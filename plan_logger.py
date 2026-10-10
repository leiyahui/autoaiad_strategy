"""为离线训练任务提供与主项目兼容的轻量日志接口。"""

import logging
from typing import Optional


_process_name = "mateval"
_tier_prefix = ""
_logger = logging.getLogger("mateval")
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)


def set_process_name(name: str) -> None:
    """设置日志中的任务名称。"""
    global _process_name
    _process_name = name or "mateval"


def set_tier_prefix(tier: str) -> None:
    """保留主项目的分级日志接口，便于代码复用。"""
    global _tier_prefix
    _tier_prefix = "[{}] ".format(tier) if tier else ""


def _message(content: object, phase: Optional[str] = None) -> str:
    """组合进程、分级和阶段信息。"""
    phase_prefix = "[{}] ".format(phase) if phase else ""
    return "[{}] {}{}{}".format(
        _process_name,
        _tier_prefix,
        phase_prefix,
        content,
    )


def log_debug(content: object, plan=None, phase: Optional[str] = None) -> None:
    """记录调试信息，plan参数仅用于兼容主项目调用。"""
    _logger.debug(_message(content, phase))


def log_info(content: object, plan=None, phase: Optional[str] = None) -> None:
    """记录普通信息，plan参数仅用于兼容主项目调用。"""
    _logger.info(_message(content, phase))


def log_warn(content: object, plan=None, phase: Optional[str] = None) -> None:
    """记录警告信息，plan参数仅用于兼容主项目调用。"""
    _logger.warning(_message(content, phase))


def log_error(content: object, plan=None, phase: Optional[str] = None) -> None:
    """记录错误信息，plan参数仅用于兼容主项目调用。"""
    _logger.error(_message(content, phase))
