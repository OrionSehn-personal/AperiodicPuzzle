

from threading import local


def initialize_svg(svg_filename):
    file = open(svg_filename, "w")
    file.write(
'''
<svg width="1500" height="1500" xmlns="http://www.w3.org/2000/svg">
'''
        )
    file = open(svg_filename, "a")
    return file
    
def finalize_svg(file):
    file.write(
'''
</svg>
'''
    )
    file.close()
    return

# def draw_puzzle_svg(curve_list, svg_filename):
#     file = initialize_svg(svg_filename)
#     for curve in curve_list:
#         draw_curve(file)
#     finalize_svg(file)
#     return




def draw_curve(point_list, file):
    translation = 10
    scaling = 70
    local_point_list = point_list.copy()
    initial = local_point_list.pop(0)
    file.write(f"\t<path d=\"M {((initial[0]+translation) * scaling)} {((initial[1]+translation)*scaling)} ")
    if len(local_point_list)==2:
        file.write("Q")
    else:
        file.write("C")
    for point in local_point_list:
        file.write(f" {((point[0]+translation) * scaling)} {((point[1]+translation) * scaling)},")
    file.write("\" stroke=\"black\" fill=\"transparent\"/>\n")
    return


