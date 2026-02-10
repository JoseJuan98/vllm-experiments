"""
Streamlit UI for OpenAI-compatible API Server

Interactive web interface for testing the vLLM API server.
"""

import streamlit as st
import requests
import json
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Initialize configuration
config = Config()


def query_api(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: str,
) -> dict:
    """Send a completion request to the API server."""
    url = f"{base_url}/v1/completions"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def query_chat_api(
    base_url: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model: str,
) -> dict:
    """Send a chat completion request to the API server."""
    url = f"{base_url}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def main():
    st.set_page_config(
        page_title="vLLM API Interface",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 vLLM OpenAI-Compatible API Interface")
    st.markdown("Interactive interface for testing the vLLM API server")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    api_config = config.get_api_config()
    api_url = st.sidebar.text_input(
        "API Server URL",
        value=f"http://localhost:{api_config['port']}",
        help="Base URL of the vLLM API server"
    )
    
    model_name = st.sidebar.text_input(
        "Model Name",
        value=config.default_model,
        help="Name of the model to use"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Generation Parameters")
    
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=1,
        max_value=2048,
        value=256,
        help="Maximum number of tokens to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Sampling temperature"
    )
    
    top_p = st.sidebar.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling parameter"
    )
    
    # Main content area - tabs for different modes
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Completion", "ℹ️ Info"])
    
    with tab1:
        st.header("Chat Completion")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Enter your message..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = query_chat_api(
                        api_url,
                        st.session_state.messages,
                        max_tokens,
                        temperature,
                        top_p,
                        model_name,
                    )
                    
                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                        assistant_message = f"Error: {response['error']}"
                    else:
                        assistant_message = response["choices"][0]["message"]["content"]
                        st.markdown(assistant_message)
                        
                        # Show metadata
                        with st.expander("Response Metadata"):
                            st.json(response)
            
            # Add assistant response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_message}
            )
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    with tab2:
        st.header("Text Completion")
        
        prompt_text = st.text_area(
            "Enter your prompt:",
            height=150,
            placeholder="Once upon a time..."
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("🚀 Generate", type="primary", use_container_width=True)
        
        if submit_button and prompt_text:
            with st.spinner("Generating completion..."):
                response = query_api(
                    api_url,
                    prompt_text,
                    max_tokens,
                    temperature,
                    top_p,
                    model_name,
                )
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    st.markdown("### Generated Text")
                    completion = response["choices"][0]["text"]
                    st.markdown(f"**Prompt:** {prompt_text}")
                    st.markdown(f"**Completion:** {completion}")
                    
                    st.markdown("### Response Details")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tokens Generated", 
                                response["usage"]["completion_tokens"])
                    with col2:
                        st.metric("Total Tokens", 
                                response["usage"]["total_tokens"])
                    with col3:
                        st.metric("Finish Reason", 
                                response["choices"][0]["finish_reason"])
                    
                    with st.expander("Full Response JSON"):
                        st.json(response)
    
    with tab3:
        st.header("Server Information")
        
        st.markdown(f"""
        ### Configuration
        - **API Server URL:** `{api_url}`
        - **Model:** `{model_name}`
        - **Max Tokens:** `{max_tokens}`
        - **Temperature:** `{temperature}`
        - **Top P:** `{top_p}`
        
        ### API Endpoints
        - **Chat Completions:** `{api_url}/v1/chat/completions`
        - **Completions:** `{api_url}/v1/completions`
        - **Models:** `{api_url}/v1/models`
        
        ### Usage Example (cURL)
        ```bash
        curl {api_url}/v1/completions \\
          -H "Content-Type: application/json" \\
          -d '{{
            "model": "{model_name}",
            "prompt": "Hello, world!",
            "max_tokens": {max_tokens},
            "temperature": {temperature}
          }}'
        ```
        
        ### Python Example
        ```python
        import openai
        
        openai.api_base = "{api_url}/v1"
        openai.api_key = "EMPTY"  # vLLM doesn't require API key
        
        completion = openai.Completion.create(
            model="{model_name}",
            prompt="Hello, world!",
            max_tokens={max_tokens},
            temperature={temperature}
        )
        
        print(completion.choices[0].text)
        ```
        """)
        
        # Test connection button
        if st.button("🔍 Test Server Connection"):
            with st.spinner("Testing connection..."):
                try:
                    response = requests.get(f"{api_url}/v1/models", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Server is reachable!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Server returned status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection failed: {str(e)}")


if __name__ == "__main__":
    main()
