import re
import torch

def validate_pytorch2(torch_version: str = None):
    torch_version = torch.__version__ if torch_version is None else torch_version

    pattern = r"^2\.\d+(\.\d+)*"

    return True if re.match(pattern, torch_version) else False