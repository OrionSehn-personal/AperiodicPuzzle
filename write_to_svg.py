

from threading import local


def initialize_svg(svg_filename, size=1500):
    file = open(svg_filename, "w")
    file.write(
f'''
<svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
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




def draw_curve(point_list, file, size=1500):
    translation = size/2
    scaling = size/12
    local_point_list = point_list.copy()
    initial = local_point_list.pop(0)
    file.write(f"\t<path d=\"M {((initial[0]* scaling) + translation)} {((initial[1]* scaling) + translation)} ")
    if len(local_point_list)==2:
        file.write("Q")
    else:
        file.write("C")
    for point in local_point_list:
        file.write(f" {((point[0] * scaling) + translation)} {((point[1] * scaling) + translation)},")
    file.write("\" stroke=\"black\" fill=\"transparent\"/>\n")
    return


def draw_line(p1, p2, file, size=1):
    translation = size/2
    scaling = size/12
    file.write(f"\t<line x1=\"{((p1[0] * scaling) + translation)}\" y1=\"{((p1[1]*scaling)+translation)}\" x2=\"{((p2[0]*scaling)+translation)}\" y2=\"{((p2[1]*scaling)+translation)}\" stroke=\"black\" />\n")
    return

def draw_lines(lines, file, size=1):
    for line in lines:
        draw_line(line[0], line[1], file, size=size)


