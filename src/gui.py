import streamlit as st
import time
import pynvml
import threading

from src.utils import ChatBot, get_chat_history
from src.gui_utils import get_window_width, infer_height
from src import params


st.set_page_config(page_title="Local Llama Chatbot", layout="wide")

# Check if pynvml is properly installed and GPU is available
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()


ss = st.session_state

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State for History and UI

if 'chat_session' not in ss:
    ss.chat_session = ChatBot()

if 'list_conversations' not in ss:
    _chat_history_ = get_chat_history()
    ss.list_conversations = _chat_history_.list_conversations

if "model_params" not in ss:
    ss.model_params = None

if "max_tokens" not in ss:
    ss.max_tokens = 512

if "rerun_counter" not in ss:
    ss.rerun_counter = 0

if "chat_container_height" not in ss:
    ss.chat_container_height = 800

if "pynvml_placeholder" not in ss:
    ss.pynvml_placeholder = None

if "gpu_info" not in ss:
    ss.gpu_info = {}



# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    model_mode = st.selectbox(
        "Select a mode",
        options=list(params.QWEN35_PARAMS.keys()),
        index=0,
        key="select_model_mode"
    )
    ss.model_params = params.QWEN35_PARAMS[model_mode]
    ss.max_tokens = params.QWEN35_DEFAULT_MAX_TOKENS[model_mode]

    max_tokens_options = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    ss.max_tokens = st.selectbox(
        "max_tokens:",
        options=max_tokens_options,
        index=max_tokens_options.index(ss.max_tokens),
        key="select_max_tokens"
    )

    with st.expander("Generation parameters:"):
        for k, v in ss.model_params.items():
            st.write(k, ":", v)

    st.divider()

    if st.button("New Conversation"):
        ss.chat_session = ChatBot()
        st.rerun()
    
    # Save/Reset
    with st.expander(f":material/history: :green[{ss.chat_session.conversation_name}]"):
        selected_conv = st.selectbox(
            "Select conversation:",
            options=ss.list_conversations,
            key="select_conv"
        )
        if st.button("Load", key="btn_load_conv"):
            ss.chat_session = ChatBot(selected_conv)
            st.rerun()

        if st.button("Delete", key="btn_delete_conv"):
            _chat_history_ = get_chat_history()
            _chat_history_.delete_conversation(selected_conv)

        if st.button("Reset History"):
            _chat_history_.reset_history()
            st.info("History reset")
            
    st.divider()

    if st.button(":material/refresh: Refresh"):
        ss.rerun_counter += 1

    width_sidebar = get_window_width(
        rerun_counter=ss.rerun_counter,
        sidebar=True
    )



cols = st.columns([3, 1])

with cols[0]:
    width_main = get_window_width(rerun_counter=ss.rerun_counter)
    height = infer_height(width=width_main, width_sidebar=width_sidebar)
    if height:
        ss.chat_container_height = height
    else:
        height = ss.chat_container_height

    chat_container = st.container(height=height)

    with chat_container:
        # Display previous messages
        if ss.chat_session.conversation:
            for msg in ss.chat_session.conversation:
                role = msg['role']
                content = msg['content']
                if role == 'user':
                    with st.chat_message("user"):
                        st.markdown(content)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(content)

        # Chat Input
        prompt = st.chat_input("Type your message here...")

        if prompt:
            # Add user message to UI and History
            st.chat_message("user").markdown(prompt)

            # Stream Assistant Response
            with st.chat_message("assistant"):
                st.markdown("") # Placeholder
                streamer = ss.chat_session.get_response_stream(
                    query=prompt,
                    max_tokens=ss.max_tokens,
                    params=ss.model_params
                )
                print("streamer", streamer)
                
                full_text = ""
                # empty = st.empty()
                
                # Generator loop to display word-by-word
                with st.spinner(text="In progress..."):
                #     for sentence in streamer:
                #         full_text += sentence + "\n" # Restore linebreak
                #         # empty.markdown(full_text)
                #         # time.sleep(0.05) # Small delay for visual effect
                    st.markdown(streamer)
                # Add assistant message to History
                ss.chat_session.add_assistant_message(streamer.strip())


with cols[1]:

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
                        "Memory": pynvml.nvmlDeviceGetMemoryInfo(h).used // (1024 * 1024),
                        "Temp.": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        "Clock": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
                    })
                print(gpu_info)
                time.sleep(1)
        threading.Thread(target=monitor_gpu, daemon=True).start()

    with ss.pynvml_placeholder.container():
        @st.fragment(run_every=1)
        def update_metrics():
            for k, v in gpu_info.items():
                if k == 'GPU':
                    st.metric(k, f"{v} %")
                if k == 'Memory':
                    st.metric(k, f"{v} MB")
                if k == 'Temp.':
                    st.metric(k, f"{v} °C")
                if k == 'Clock':
                    st.metric(k, f"{v} MHz")
        update_metrics()