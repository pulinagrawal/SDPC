import torch

def discover_device():
  if torch.cuda.is_available():
    return 'cuda'
  if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
      print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
      return 'cpu'
    else:
      print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
      return 'cpu'
  else:
    return 'mps'

torch.discover_device = discover_device  
  

