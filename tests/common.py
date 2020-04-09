# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

def replace_text_in_file(path, replace_what, replace_by):
    with open(path) as read_file:
        content = '\n'.join([line.rstrip() for line in read_file.readlines()])
        if content.find(replace_what) == -1:
            return False
        content = content.replace(replace_what, replace_by)
    with open(path, 'w') as write_file:
        write_file.write(content)
    return True


def collect_ap(path):
    ap = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                ap.append(float(line.replace(beginning, '')))
    return ap
