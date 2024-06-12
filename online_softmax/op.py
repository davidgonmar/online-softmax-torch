# add build path to sys.path
import torch
import online_softmax_cu



def online_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)

    online_softmax_cu.forward(x, out)

    return out



if __name__ == '__main__':
    def reference(x):
        return torch.nn.functional.softmax(x, dim=-1)


    # benchmarks
    import torch
    import time

    x = torch.randn(500, 3000, device='cuda', dtype=torch.float32)

    # warmup
    for _ in range(10):
        online_softmax(x)
        reference(x)

    
    # online softmax
    start = time.time()
    for _ in range(10000):
        online_softmax(x)
        torch.cuda.synchronize()
    print('online softmax:', time.time() - start)

    
    # reference softmax
    start = time.time()
    for _ in range(10000):
        reference(x)
        torch.cuda.synchronize()
    print('reference softmax:', time.time() - start)
