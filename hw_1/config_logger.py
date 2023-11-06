import logging

# 配置日志记录
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# 创建一个全局日志记录器
logger = logging.getLogger("debug")
