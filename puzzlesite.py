import streamlit as st  
import base64
from fibbonacciTimesFibbonacciSubstitution import *
from Bezier import * 
from write_to_svg import *
import plotly.express as px

def draw_svg(svg_file):
    with open(svg_file, 'r') as f:
        svg = f.read()
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.title("Puzzle Generator") 
with st.expander("About"):
    st.markdown(
        r"""
        ## Introduction
        This project explores Bezier Curves, and their applications. In particular, applying restrictions to a set of Bezier Curves to generate jigsaw puzzles with non-standard bases. These non-standard bases come from Substitution Tilings. 
        
        The project is publically availible at: https://github.com/OrionSehn-personal/AperiodicPuzzle

        
        
        
        """
    )

grid_types = ["Penrose", "Ammann beenker", "Square", "Custom"]
selected_grid = st.selectbox("Select Grid Type", options=grid_types)

st.markdown("""---""")

col1, col2 = st.columns([1.5, 1])

size = 650


with col2:
    gen_type = st.selectbox("Edge Generation Type", ["Random_Uniform", "Random Binary", "Simplex Optimization"])
    

with col1:
    if selected_grid == "Penrose":

        iterations = st.number_input("Iterations", min_value=0, max_value=5, value=2)

        if iterations > 4:
            # st.warning("This may take a while")

            with st.spinner("Generating Puzzle - Warning: This may take a while"):
                penlines, border_list = penroseLines(iterations, maxradius=np.inf, init_scaling=0.2)
                svg_file = initialize_svg("puzzle.svg", size = size)
                param_set = bitwise_distribution(len(penlines))
                curveGen(penlines, param_set, flipTabs=True, svg_file=svg_file, size=size)
                draw_lines(border_list, svg_file, size=size)
                finalize_svg(svg_file)
                svg_file.close()

            draw_svg("puzzle.svg")

        else:
            with st.spinner("Generating Puzzle"):
                penlines, border_list = penroseLines(iterations, maxradius=np.inf)
                svg_file = initialize_svg("puzzle.svg", size = size)
                param_set = bitwise_distribution(len(penlines))
                curveGen(penlines, param_set, flipTabs=True, svg_file=svg_file, size=size)
                draw_lines(border_list, svg_file, size=size)
                finalize_svg(svg_file)
                svg_file.close()

            draw_svg("puzzle.svg")
    elif selected_grid == "Ammann beenker":
        st.write("Ammann beenker - not yet implemented")

    elif selected_grid == "Square":

        x = st.number_input("Squares in x axis", min_value=0, max_value=100, value=10)
        y = st.number_input("Squares in y axis", min_value=0, max_value=100, value=10)
        st.markdown(f"Number of Tiles: {x*y}")

        if 0<x<17 and 0<y<17:
            line_set, border_set = recGrid(x, y)
            svg_file = initialize_svg("puzzle.svg", size = size)
            param_set = bitwise_distribution(len(line_set))
            curveGen(line_set, random_distribution(len(line_set)), flipTabs=True, svg_file=svg_file, size=size)
            for line in border_set:
                draw_line(line[0], line[1], svg_file, size=size)
            finalize_svg(svg_file)
            svg_file.close()
            draw_svg("puzzle.svg")
        elif 17<=x<30 or 17<=y<30:
            with st.spinner("Generating Puzzle"):
                line_set, border_set = recGrid(x, y, scaling=0.3, translate=16)
                svg_file = initialize_svg("puzzle.svg", size = size)
                param_set = bitwise_distribution(len(line_set))
                curveGen(line_set, random_distribution(len(line_set)), flipTabs=True, svg_file=svg_file, size=size)
                for line in border_set:
                    draw_line(line[0], line[1], svg_file, size=size)
                finalize_svg(svg_file)
                svg_file.close()
                draw_svg("puzzle.svg")
        






    elif selected_grid == "Custom":
        st.write("Make your own puzzle by uploading a list of edges, and a list of the borders.")

        edges = st.file_uploader("Puzzle Edges", type="svg")
        border = st.file_uploader("Puzzle Border", type="svg")


with col2: 
    with open("puzzle.svg", "rb") as file:
        btn = st.download_button(
                label="Download SVG",
                data=file,
                file_name="puzzle.svg",
                mime="image/svg+xml",
            )
    differences = hamming_set_difference(param_set)
    st.markdown("### Puzzle Edge Difference Statistics: ")
    st.markdown("Each edge has a vector representation, and the difference between two edges is the hamming distance between two vectors. The following statistics are calculated from the hamming differences between the edges in the puzzle.")
    st.text(f"{differences.describe()}")
    fig = px.histogram(differences, nbins=18, title="Hamming Distance Distribution", width=600, range_x=[0, 18])
    fig.update_layout(showlegend=False, bargap = 0.1, xaxis_title="Hamming Distance", yaxis_title="Number of Edges")
    st.plotly_chart(fig)




    
# draw_svg("puzzle_sample.svg")
