import os
from datetime import datetime


TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

# 从环境变量获取 START_TIME，如果未设置则使用当前时间
START_TIME_STR = os.getenv("START_TIME", datetime.now().strftime(TIME_FORMAT))