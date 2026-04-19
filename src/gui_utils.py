from streamlit_js_eval import streamlit_js_eval
import streamlit as st
import threading
import psutil
import pynvml
import time


# Check if pynvml is properly installed and GPU is available
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()


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


gpu_info = None

def hardware_monitoring(ss):
    global gpu_info
    gpu_info = ss.gpu_info

    if not ss.pynvml_placeholder:
        ss.pynvml_placeholder = st.empty()
        
        def monitor_gpu():
            global gpu_info

            while True:
                for i in range(device_count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_info.update({
                        "name": pynvml.nvmlDeviceGetName(h),
                        "GPU": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                        "VRAM": round(pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 3), 1),
                        "Temp.": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        "CPU": psutil.cpu_percent(),
                        "RAM": round(psutil.virtual_memory().percent * 0.32, 1)
                    })
                time.sleep(1)
        threading.Thread(target=monitor_gpu, daemon=True).start()

    with ss.pynvml_placeholder.container():
        @st.fragment(run_every=1)
        def update_metrics():
            for k, v in gpu_info.items():
                if k == 'GPU':
                    st.metric(k, f"{v} %", border=True)
                if k == 'VRAM':
                    st.metric(k, f"{v} GB", border=True)
                if k == 'Temp.':
                    st.metric(k, f"{v} °C", border=True)
                if k == 'CPU':
                    st.metric(k, f"{v} %", border=True)
                if k == 'RAM':
                    st.metric(k, f"{v} GB", border=True)
        update_metrics()
