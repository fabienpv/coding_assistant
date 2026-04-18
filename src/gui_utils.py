from streamlit_js_eval import streamlit_js_eval
import streamlit as st


def get_window_width(rerun_counter: int, sidebar: bool = False) -> int:
    page_width = streamlit_js_eval(
        js_expressions='window.innerWidth',
        key=f'WIDTH_{int(sidebar)}_{rerun_counter}',  
        want_output = True,
    )
    return page_width


def infer_height(width: int, width_sidebar: int) -> int:
    if width and width_sidebar:
        if width + width_sidebar > 1300:
            infered_height = 1150
        else:
            infered_height = 800
        print("width", width + width_sidebar, "container_height:", infered_height)
    else:
        return None
    return infered_height
