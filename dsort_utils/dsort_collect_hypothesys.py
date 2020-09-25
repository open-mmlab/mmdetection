import os
import sys
import numpy as np


def update_ids_next_chunk(hyp_lst_chunk, matches, max_frame_idx):

    for frame_num in range(1, max_frame_idx+1):
        frame_key = str(frame_num)

        if frame_num == max_frame_idx:
            for trset in hyp_lst_chunk[frame_key]:
                for m in matches:
                    if trset[0] == m[1]:
                        print ('old id -> %d new id -> %d' % (trset[0], m[0]))

def combine_tracks(hyp_lst, last_frame_indices, num_chunks):
    
    matches = []

    for trset1 in hyp_lst[0][last_frame_indices[0]]:
        for trset2 in hyp_lst[1]['1']:
            if trset1[1] == trset2[1] and trset1[2] == trset2[2]:
                matches.append((trset1[0], trset2[0]))
    
    for m in matches:
        print (m)

    print ('number of id mismatches between chunks: %d' % (len(hyp_lst[0][last_frame_indices[0]])-len(matches)))

    update_ids_next_chunk(hyp_lst[1], matches, int(last_frame_indices[1]))

def collect_hypothesys(path, n_chunks):
    hyp_lst = []
    last_frame_indices = []

    for n in range(1, n_chunks+1):
        hyp_name = 'hypotheses_%d.txt' % (n)
        file_path = os.path.join(path, hyp_name)

        cur_dict = {}
        prev_frame_num = '-1'

        with open(file_path, "r") as fp:
            for line in fp:
                currentline = line.split(",")

                frame_num = currentline[0]
                trid = currentline[1]
                coord = (currentline[2], currentline[3])

                if prev_frame_num != frame_num:
                    cur_dict[frame_num] = []
                    prev_frame_num = frame_num

                cur_dict[frame_num].append([int(trid), float(currentline[2]), float(currentline[3])])
        
        last_frame_indices.append(prev_frame_num)

        print ('done processing hypothesys #%d' % (n))
        hyp_lst.append(cur_dict)
    
    return hyp_lst, last_frame_indices

if __name__ == '__main__':
    data_path = sys.argv[1]
    num_chunks = int(sys.argv[2])
    
    hyp_lst, last_frame_indices = collect_hypothesys(data_path, num_chunks)
    combine_tracks(hyp_lst, last_frame_indices, num_chunks)