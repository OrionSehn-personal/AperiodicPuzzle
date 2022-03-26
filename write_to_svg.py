

def initialize_svg(svg_filename):
    file = open(svg_filename, "wa")
    file.write(
        '''
        <svg width="1500" height="1500" xmlns="http://www.w3.org/2000/svg">
        '''
        )
    file.close()
def finalize_svg(file):
    file = open(file, "wa")
    file.write(
        '''
            </svg>
        '''
    )
    file.close()
    return

def draw_puzzle_svg(curve_list, svg_filename):
    file = initialize_svg(svg_filename)
    for curve in curve_list:
        draw_curve(file)
    finalize_svg(file)
    return




def draw_curve(point_list):
    test = open("sample.svg","wa")

    test.write(

    f'''
    <path d="M 100.15 100 C 20 20, 40 20, 50 10" stroke="black" fill="transparent"/>
    <path d="M 700 10 C 70 20, 110 20, 110 10" stroke="black" fill="transparent"/>
    <path d="M 1300 10 C 120 20, 180 20, 170 10" stroke="black" fill="transparent"/>
    <path d="M 10 60 C 20 80, 40 80, 50 60" stroke="black" fill="transparent"/>
    <path d="M 70 60 C 70 80, 110 80, 110 60" stroke="black" fill="transparent"/>
    <path d="M 1300 60 C 120 80, 180 80, 170 60" stroke="black" fill="transparent"/>
    <path d="M 10 110 C 20 140, 40 140, 50 110" stroke="black" fill="transparent"/>
    <path d="M 709.123123 110 C 70 140, 110 140, 110 110" stroke="black" fill="transparent"/>
    <path d="M 130.4s 110 C 120 140, 180 140, 170 110" stroke="black" fill="transparent"/>

    </svg>
    ''')

    test.close()

