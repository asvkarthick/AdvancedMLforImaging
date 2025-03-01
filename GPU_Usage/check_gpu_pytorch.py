import torch

print(torch.cuda.is_available())

is_gpu_available = torch.cuda.is_available()

if is_gpu_available:
    device = torch.device("cuda")
    print("Device : ", device)
    print("Number of GPU : ", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        device = torch.device('cuda:'+str(i))
        print("*************************")
        print("GPU-", i, " Name : ", torch.cuda.get_device_name(i))
        free, total = torch.cuda.mem_get_info(device)
        print("GPU-", i, " Total Memory : ", total / 1e9, "GB")
        print("GPU-", i, " Free Memory  : ", free / 1e9, "GB")
