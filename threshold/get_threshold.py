def get_thresholds(net_name, d_eps, i_eps, net_idx1=None, net_idx2=None):
    if 'mnist-net_256x4' in net_name:
        file_path = f"threshold/mnist4/d{d_eps}_e{i_eps}.txt"
    elif 'mnist_conv' in net_name:
        file_path = f"threshold/mnist-conv/d{d_eps}_e{i_eps}.txt"
    elif 'cifar10' in net_name:
        file_path = f"threshold/cifar10/d{d_eps}_e{i_eps}.txt"
    elif 'acasxu' in net_name:
        file_path = f"threshold/acasxu/net_{net_idx1}_{net_idx2}_d_{d_eps}.txt"

    return read_thresholds(file_path)

def get_thresholds_bs(net_name, thr_id=None):
    if thr_id is None:
        raise ValueError("thr_id should be None when calling get_thresholds_bs, as the function is designed to read from a single file for each dataset.")
    if 'mnist-net_256x4' in net_name:
        file_path = f"threshold/bs/mnist4_{thr_id}.txt"
    elif 'mnist_conv' in net_name:
        file_path = f"threshold/bs/mnist-conv_{thr_id}.txt"
    elif 'cifar10' in net_name:
        file_path = f"threshold/bs/cifar10_{thr_id}.txt"
    elif 'acasxu' in net_name:
        file_path = f"threshold/bs/acasxu_{thr_id}.txt"

    return read_thresholds(file_path)


def read_thresholds(file_path):
    """
    e.g.,
    0.002742
    0.001629
    0.006227
    0.010318
    0.00135755
    """

    thresholds = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                thresholds.append(float(s))
            except ValueError:
                # skip lines that don't contain a single float
                continue
    return thresholds
