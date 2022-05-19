from get_module import get_module
import torch


model = get_module()

if __name__ == '__main__':
    test_tensor = [torch.rand(size=(1, 256, 184, 184)),
                   torch.rand(size=(1, 512, 92, 92)),
                   torch.rand(size=(1, 1024, 46, 46)),
                   torch.rand(size=(1, 2048, 23, 23))]

    results = model(test_tensor)
    for result in results:
        print(result.shape)