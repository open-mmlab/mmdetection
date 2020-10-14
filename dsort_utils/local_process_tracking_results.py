import sys

def main(out_path, in_path):
    frame_num = 0

    f_out = open(out_path, "w")
    f_in = open(in_path, "r")

    for line in f_in:
        currentline = line.split(",")
        frame_num = int(currentline[0])
        track_id = int(currentline[1])
        x = float(currentline[2])
        y = float(currentline[3])
        bb_w = float(currentline[4])
        bb_h = float(currentline[5])

        if x>0 and y>0 and bb_w>0 and bb_h>0:
            total = str(frame_num) + "," + str(track_id) + "," + "{0:.2f}".format(x) + "," + "{0:.2f}".format(y) + "," + "{0:.2f}".format(bb_w) + "," + "{0:.2f}".format(bb_h) + "\n"
            f_out.write(total)
        else:
            print(currentline)

if __name__ == '__main__':
    result_file = sys.argv[1]
    input_file = sys.argv[2]
    main(result_file, input_file)