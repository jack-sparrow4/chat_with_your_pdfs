# Streamlit chat with PDF Application

This Streamlit application allows users to interact with PDFs by asking questions related to the content. It uses open-source LLM models powered by HuggingFace and LangChain to generate concise and helpful answers.

## Features

- Upload multiple PDF files
- Process and split PDF content into searchable vectors
- Ask questions related to the PDF content
- Get concise and context-aware answers

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/jack-sparrow4/chat_with_your_pdfs.git
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run main.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your PDF files using the sidebar.

4. Ask questions related to the PDF content using the input box.

## Configuration

- Make sure to replace the HuggingFace API token in the code with your own token.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [HuggingFace](https://huggingface.co/)
- [LangChain](https://github.com/hwchase17/langchain)
