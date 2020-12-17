#!/usr/bin/env python
import glob
import os.path as osp
import re

url_prefix = 'https://github.com/open-mmlab/mmdetection/blob/master/'

files = sorted(glob.glob('../configs/*/README.md'))

stats = []
titles = []
num_ckpts = 0
num_configs = 0

for f in files:
    url = osp.dirname(f.replace('../', url_prefix))

    with open(f, 'r') as content_file:
        content = content_file.read()

    title = content.split('\n')[0].replace('# ', '')

    titles.append(title)

    ckpts = set(x.lower().strip()
                for x in re.findall(r'\[model\]\((https?.*)\)', content))

    configs = set(x.lower().strip()
                  for x in re.findall(r'\[config\]\((https?.*)\)', content))

    num_ckpts += len(ckpts)
    num_configs += len(configs)

    statsmsg = f"""
\t* [{title}]({url}) ({len(ckpts)} ckpts)
"""
    stats.append((title, ckpts, statsmsg))

msglist = '\n'.join(x for _, _, x in stats)

modelzoo = f"""
# Model Zoo Statistics

* Number of papers: {len(titles)}
* Number of configs: {num_configs}
* Number of checkpoints: {num_ckpts}
{msglist}
"""

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo)
