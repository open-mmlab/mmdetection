import json

model_zoo = dict()

with open('single_valid_results.json', 'rb') as res:
    for line in res:
        line = json.loads(line)
        config_path = line['config'].split('/')
        model_family = config_path[1]
        if model_family not in model_zoo:
            model_zoo[model_family] = dict()

        # parse a single model
        model_dict = dict()
        config_name = config_path[2]  # *.py
        model = config_name.split('.')[0]
        train_mem = line['train_Mem']
        model_dict['Mem'] = train_mem

        metrics = ['bbox_mAP', 'segm_mAP', 'AR@1000']
        for m in metrics:
            if 'valid_' + m in line:
                if m != 'AR@1000':
                    train_metric = line['train_{}'.format(m)]
                    valid_metric = line['valid_{}'.format(m)]
                    if round(train_metric, 3) != valid_metric:
                        print('model: {}, train_{}: {} != valid_{}: {}'.format(
                            model, m, train_metric, m, valid_metric))
                else:
                    valid_metric = line['valid_{}'.format(m)]
                model_dict[m] = valid_metric
        model_dict['Inf'] = line['inf_speed']
        model_zoo[model_family][model] = model_dict

metrics = ['bbox_mAP', 'segm_mAP', 'AR@1000']
for model_family in sorted(model_zoo.keys()):
    model_family_dict = model_zoo[model_family]
    print('-' * 10, model_family, '-' * 10)
    for model in sorted(model_family_dict.keys()):
        print(model)
        model_dict = model_family_dict[model]
        mem = model_dict['Mem']
        inf_speed = model_dict['Inf']
        metric_values = []
        res = 'Mem: {}, Inf: {}, '.format(mem, inf_speed)
        for m in metrics:
            if m in model_dict:
                res += '{}: {}, '.format(m, model_dict[m])
                res = res[:-2]
        print(res)
