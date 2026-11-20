from server import *
from args import parse
from data import TinyImageNetValDataset  # noqa: F401
import warnings
warnings.filterwarnings("ignore")
import logging
import os


args = parse.parse_args()
data, num_client, alpha, backbone = args.data_name, args.num_clients, args.alpha, args.backbone_name

logger_name = args.save_result + "/logs/" + f"data_{data} client_{num_client} alpha_{alpha} backbone_{backbone}"
os.makedirs(args.model_save_path, exist_ok=True)
os.makedirs(args.data_save_path, exist_ok=True)
os.makedirs(args.save_result + "/logs", exist_ok=True)
os.makedirs(args.save_result + "/detail", exist_ok=True)

logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

file_handler = logging.FileHandler(f"{logger_name}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

Server(parse=parse, logger=logger).train()
