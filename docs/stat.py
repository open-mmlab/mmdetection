#!/usr/bin/env python
import glob
import os.path as osp
import re

url_prefix = 'https://github.com/open-mmlab/mmdetection/blob/master/'

files = sorted(glob.glob('../configs/*/README.md'))

stats = []
titles = []
num_ckpts = 0

for f in files:
    url = osp.dirname(f.replace('../', url_prefix))

    with open(f, 'r') as content_file:
        content = content_file.read()

    title = content.split('\n')[0].replace('# ', '')

    titles.append(title)

    ckpts = set(x.lower().strip()
                for x in re.findall(r'\[model\]\((https?.*)\)', content))

    num_ckpts += len(ckpts)

    statsmsg = f"""
\t* [{title}]({url}) ({len(ckpts)} ckpts)
"""
    stats.append((title, ckpts, statsmsg))

msglist = '\n'.join(x for _, _, x in stats)

modelzoo = f"""
# Model Zoo Statistics

* Number of papers: {len(titles)}
* Number of checkpoints: {num_ckpts}
{msglist}
"""

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo)
