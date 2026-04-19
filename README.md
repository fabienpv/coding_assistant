### Key Features Explained

1.  **Streamlit GUI**: Uses `st.chat_message` and `st.chat_input` for a modern chat interface.
2.  **Word-by-Word Yielding**: The `get_response_stream` method accumulates the response text and yields it word-by-word. The UI updates using `st.empty().markdown()` to reflect the text in real-time without full page reloads.
3.  **Markdown Support**: All chat messages are rendered using `st.markdown()`.
4.  **Chat History Class (`SecureChatHistory`)**:
    *   Stores the conversation in memory.
    *   Encrypts the data using `cryptography.fernet` before writing to `chat_history.json`.
    *   Decrypts the data upon loading to retrieve the past conversation.
5.  **OpenAI API Wrapper**: Uses `openai.ChatCompletion.create` to connect to the local `llama_cpp` server.
6.  **Model Parameters**:
    *   Standard parameters (`temperature`, `top_p`, `top_k`, `presence_penalty`) are mapped directly.
    *   Custom parameters (`min_p`, `enable_thinking`) are passed via `**kwargs`. *Note: `enable_thinking` and `min_p` are not standard OpenAI API parameters. They rely on the specific `llama-cpp` server implementation exposing them.*
7.  **Security**: Uses Fernet encryption for the JSON file. A warning is included about key management.
8.  **Hardware Monitoring** the interface provides a overview of the hardware monitoring: VRAM, RAM, % GPU, % CPU, temperature of GPU.


### Data:

- markdowns: OCR outputs saved locally
- samples: a place where to store outputs when I test features
- spelling: contains the documents for spelling correction post OCR
- temp: contains the loaded documents during the session
- temp_img: contains the loaded and processed images during the session

### Prerequisites

You will need to install the following Python packages:

```bash
pip install -r requirements.txt
```

**Local Server Setup:**
Ensure your `llama-cpp-python` server is running and configured to accept OpenAI API requests. You can start it with something like:

```bash
llama-cpp-python --server-port 8000 --server-name 0.0.0.0 --model your_model.gguf --chat-template-kwargs enable_thinking=True
```
*Note: Ensure your specific server version supports the parameters `min_p` and `enable_thinking` via the API.*

### How to Run

1.  Run the Streamlit app:
    ```bash
    python -m streamlit run ./src/gui.py
    ```
2.  Open the link provided in your browser (usually `http://localhost:8501`).

### Security Note

For this demo, the encryption key is generated once and stored in the environment. In a production environment, you should retrieve this key from a secure secrets manager (e.g., AWS Secrets Manager, Kubernetes Secrets) or a dedicated `.env` file that is not committed to version control. The `chat_history.json` file will be encrypted, but the key must be secure to prevent unauthorized decryption.