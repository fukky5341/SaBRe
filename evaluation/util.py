
def get_time_budget(df, net_name):
    if net_name == 'acasxu':
        time_budget = 420  # seconds
    elif net_name == 'mnist4' or net_name == 'mnist-conv':
        time_budget = 600  # seconds
    elif net_name == 'cifar10':
        time_budget = df['name'].apply(lambda x: 1800 if x.startswith('1_') \
                                       else 3600 if x.startswith('2_') else 7200)
    elif net_name == 'gtsrb':
        time_budget = df['name'].apply(lambda x: 1800 if x.startswith('1_') \
                                       else 3600 if x.startswith('2_') else 7200)
    else:
        raise ValueError("Unknown net_name")
    return time_budget


method_to_time_col = {
    'NS': 'NS time',
    'NSInd': 'NSInd time',
    'DS_dual_Z': 'DSZ time',
    'DS_random_Z': 'RndZ time',
    'Random': 'Random time',
    'Dual': 'Dual time',
    'RS_dual_Z': 'RSZ time',
    'RS_random_Z': 'RSRndZ time',
    'IS_dual': 'IS time',
    'IS_dual_ind': 'ISInd time'
}

method_to_status_col = {
    'NS': 'NS status',
    'NSInd': 'NSInd status',
    'DS_dual_Z': 'DSZ status',
    'DS_random_Z': 'RndZ status',
    'Random': 'Random status',
    'Dual': 'Dual status',
    'RS_dual_Z': 'RSZ status',
    'RS_random_Z': 'RSRndZ status',
    'IS_dual': 'IS status',
    'IS_dual_ind': 'ISInd status'
}

method_to_dsns_name = {
    'NS_dual': 'NS',
    'NS_ind_dual': 'NSInd',
    'DS_dual_Z': 'DS_dual_Z',
    'DS_random_Z': 'DS_random_Z',
    'Random': 'Random',
    'Dual': 'Dual',
    'RS_dual_Z': 'RS_dual_Z',
    'RS_random_Z': 'RS_random_Z',
    'IS_dual': 'IS_dual',
    'IS_dual_ind': 'IS_dual_ind'
}

method_name_map = {
    'NS': 'IS',
    'NSInd': 'IS individual',
    'DS_dual_Z': 'RS',
    'DS_random_Z': 'RS random Z',
    'Random': 'RS Random',
    'Dual': 'RS Dual',
    'RS_dual_Z': 'RS',
    'RS_random_Z': 'RS random Z',
    'IS_dual': 'IS',
    'IS_dual_ind': 'IS individual'
}
