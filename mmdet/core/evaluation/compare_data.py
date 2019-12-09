import numpy as np

def calculate_area(box_1, box_2, iou_threshold):
    probability_1 = box_1[4]
    probability_2 = box_2[4]
    location_1 = box_1
    location_2 = box_2
    x_min = max(location_1[0], location_2[0])
    y_min = max(location_1[1], location_2[1])
    x_max = min(location_1[2], location_2[2])
    y_max = min(location_1[3], location_2[3])
    if x_max > x_min and y_max > y_min:
        I = (x_max - x_min) * (y_max - y_min)
        U = (location_1[2] - location_1[0]) * (location_1[3] - location_1[1]) + (location_2[2] - location_2[0]) * (
                location_2[3] - location_2[1]) - I
        ratio = I / U
        #         print(ratio)
        if ratio > iou_threshold:
            if probability_1 > probability_2:
                return 1
            else:
                return 0
        else:
            return -1
    return -1


def compare_box_list(box_list_1, box_list_2, iou_threshold=0.5):
    box_pair = []
    length_1 = len(box_list_1)
    #     print(length_1)
    index_1 = 0
    while length_1 > 0:
        box_1 = box_list_1[index_1]
        #         print(box_1)
        index_2 = 0
        length_2 = len(box_list_2)
        while length_2 > 0:
            box_2 = box_list_2[index_2]
            area = calculate_area(box_1, box_2, iou_threshold)
            if area == -1:
                #                 print("go")
                length_2 -= 1
                index_2 += 1
            elif area == 1:
                #                 print("delete 2")
                box_list_2 = np.delete(box_list_2, index_2, 0)
                #                 box_list_2.pop(index_2)
                index_1 += 1
                length_1 -= 1
                break
            else:
                #                 print("delete")
                box_list_1 = np.delete(box_list_1, index_1, 0)
                length_1 -= 1
                break
        if length_2 <= 0:
            length_1 -= 1
            index_1 += 1
        # else:
        #     continue
    #             box_pair.append([box_list_1[index_1],box_list_2[index_2]])
    #     print(len(box_list_1))
    return box_list_1, box_list_2, box_pair

def delete_duplicate_box(result):

    result[0], result[1], _ = compare_box_list(result[0], result[1])
    result[1], result[2], _ = compare_box_list(result[1], result[2])
    result[1], result[3], _ = compare_box_list(result[1], result[3])
    result[2], result[3], _ = compare_box_list(result[2], result[3])
    result[0], result[2], _ = compare_box_list(result[0], result[2])
    result[0], result[3], _ = compare_box_list(result[0], result[3])

    return result
if __name__ == "__main__":
    pass
