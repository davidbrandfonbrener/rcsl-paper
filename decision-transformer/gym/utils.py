import csv
import os
import wandb

class Logger():
    def __init__(self, config, dir, seed,
                    name=None, group=None):
        
        wandb.init(
            name=name,
            group=group,
            config=config,
        )

        csv_path = os.path.join(dir, 'csv', str(seed))
        if not os.path.isdir(csv_path):
            os.makedirs(csv_path)

        self.types = ["train"]
        self.csv_files = {}
        self.csv_init = {}
        for t in self.types:
            self.csv_files.update({t: 
                                open(os.path.join(csv_path, f'{t}.csv'),
                                'w', newline='')})
            self.csv_init.update({t: False})
        self.csv_writers = {}

    def write(self, dict, type, step):

        # write to wandb
        wandb_dict = {}
        for k, v in dict.items():
            wandb_dict.update({f'{type}/{k}': v})
        wandb.log(wandb_dict)

        # write to csvs
        assert type in self.types
        dict.update({'step': step})
        if not self.csv_init[type]:
            fieldnames = dict.keys()
            self.csv_writers.update({type: 
                                        csv.DictWriter(
                                            self.csv_files[type], 
                                            fieldnames=fieldnames)})
            self.csv_writers[type].writeheader()
            self.csv_init[type] = True
        
        self.csv_writers[type].writerow(dict)
        self.csv_files[type].flush()

    def close(self):
        for t in self.types:
            self.csv_files[t].close()


def generate_name(config):
    name = 'dt'
    keys = ['dataset']

    for k in keys:
        if k in config.keys():
            name += f'_{k[:3]}={config[k]}'
    return name