import cv2
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
import json

color_map = {}

with open('resources/color_map.json') as cmp:
    color_map = json.load(cmp)


def get_name_of_color_lib(color_array):
    requested_colour = (int(color_array[0]), int(color_array[1]), int(color_array[2]))
    min_colours = {}
    
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        # r_c, g_c, b_c = key
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    
    return min_colours[min(min_colours.keys())]


def get_name_of_color_local(color_array):
    requested_colour = (int(color_array[0]), int(color_array[1]), int(color_array[2]))
    
    min_colours = {}
    
    # for key, name in webcolors.css3_hex_to_names.items():
    for key, name in color_map.items():
        r_c, g_c, b_c = name
        # if name == 'black':
        #     lol = 0
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = key
    
    return min_colours[min(min_colours.keys())]


# color_name = ''
# color_name = rgb_to_name()
# try:
# 	color_name = webcolors.rgb_to_name(requested_colour)
# except ValueError:
# 	color_name = closest_colour(requested_colour)
# color_name = closest_colour(requested_colour)
# 	# actual_name = None
# # return actual_name, closest_name
# return  color_name

def get_percentages(colors_count):
    percentages = []
    total_sum = sum([a[1] for a in colors_count])
    percentages = [(a[0], (a[1] * 100) / total_sum) for a in colors_count]
    return percentages


def get_colors(image, number_of_colors):
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    # counts = dict(sorted(counts.items()))
    counts = [(key, val) for key, val in counts.items()]
    
    counts = sorted(counts, key=lambda x: x[1], reverse=True)
    
    center_colors = clf.cluster_centers_
    
    ordered_cs = [(center_colors[i[0]], counts[i[0]][1]) for i in counts]
    
    # We get ordered colors by iterating through the keys
    ordered_colors = [(get_name_of_color_local(i[0]), i[1]) for i in ordered_cs]
    ordered_colors_lib = [(get_name_of_color_lib(i[0]), i[1]) for i in ordered_cs]
    percenatges = get_percentages(ordered_colors)
    temp = 0
    
    return ordered_colors


# hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
# rgb_colors = [ordered_colors[i] for i in counts.keys()]
#
# if (show_chart):
# 	plt.figure(figsize=(8, 6))
# 	plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)

# return rgb_colors

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def extract_color_scheme(img_path, coord_list):
    color_scheme = {}
    # get_image(img_path)
    im = get_image(img_path)
    i = 0
    for coord in coord_list:
        curr_img = im[coord[1]: coord[3], coord[0]: coord[2], :]
        
        ordered_colors = get_colors(curr_img, 8)
        # print('for ' + coord[4]  + ' color scheme is ' + ordered_colors)
        # try:
        #     # cv2.imwrite('lol_men_extracted_' + str(i) +'.jpg', curr_img)
        # except:
        #     print('exception')
        i = i + 1
    
    return color_scheme


