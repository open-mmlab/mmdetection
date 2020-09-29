import os
import sys
import numpy as np

def update_ids_next_chunk(hyp_lst_chunk, matches, max_frame_idx, cid):
    
    # create new file with updated track ids
    filename = "./data/new_hypotheses_%s.txt" % (cid+1)
    fp = open(filename, "w")

    for frame_num in range(1, max_frame_idx+1):
        frame_key = str(frame_num)

        for trset in hyp_lst_chunk[frame_key]:

            for m in matches:
                if trset[0] == m[1]:
                    trset[0] = m[0]

            newstr = '%d, %d, %.2f, %.2f, %.2f, %.2f, 1,-1,-1, %d \n' % (int(frame_key)-1, trset[0], trset[1], trset[2], trset[3], trset[4], 1)

            # don't write first frame (first frame is same as previous chunk's tracks)
            if frame_key != '1':
                fp.write(newstr)

def combine_tracks(hyp_lst, last_frame_indices, num_chunks):
    for cid in range(num_chunks-1):
            matches = []

            for trset1 in hyp_lst[cid][last_frame_indices[cid]]:
                for trset2 in hyp_lst[cid+1]['1']:
                    if trset1[1] == trset2[1] and trset1[2] == trset2[2]:
                        matches.append((trset1[0], trset2[0]))
            
            print ('matches between chunk indices #%d and #%d ==>' % (cid, cid+1))
            print ('number of id mismatches between chunks: %d' % (len(hyp_lst[cid][last_frame_indices[cid]])-len(matches)))

            update_ids_next_chunk(hyp_lst[cid+1], matches, int(last_frame_indices[cid+1]), cid+1)

    # writing result file
    print ("building final output ==>")
    filename = "./data/all_hypotheses.txt"
    fp_out = open(filename, "w")

    for cid in range(1, num_chunks+1):
        out_file = ""

        if cid == 1:
            out_file = "./data/hypotheses_%d.txt" % (cid)
        else:
            out_file = "./data/new_hypotheses_%s.txt" % (cid)

        offset = -1000

        if cid == 1:
            offset = 0
        elif cid == 2:
            offset = int(last_frame_indices[cid-2])
        else:
            offset = int(last_frame_indices[cid-2])-1

        with open(out_file, "r") as fp_in:
            for line in fp_in:
                currentline = line.split(",")

                frame_num = int(currentline[0])
                trid = int(currentline[1])
                x = float(currentline[2])
                y = float(currentline[3]) 
                w = float(currentline[4])
                h = float(currentline[5])

                newstr = '%d, %d, %.2f, %.2f, %.2f, %.2f, 1,-1,-1, %d \n' % (frame_num+offset*(cid-1), trid, x, y, w, h, 1)
                fp_out.write(newstr)

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

                if prev_frame_num != frame_num:
                    cur_dict[frame_num] = []
                    prev_frame_num = frame_num

                cur_dict[frame_num].append([int(trid), float(currentline[2]), float(currentline[3]), float(currentline[4]), float(currentline[5])])
        
        last_frame_indices.append(prev_frame_num)

        print ('done processing hypothesys #%d' % (n))
        hyp_lst.append(cur_dict)
    
    return hyp_lst, last_frame_indices

if __name__ == '__main__':
    data_path = sys.argv[1]
    num_chunks = int(sys.argv[2])
    
    hyp_lst, last_frame_indices = collect_hypothesys(data_path, num_chunks)
    combine_tracks(hyp_lst, last_frame_indices, num_chunks)