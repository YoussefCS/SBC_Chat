# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from flask import Flask, request, jsonify
from flask_cors import CORS
from ChatRTX.chatrtx_rag import ChatRTXRag
from ChatRTX.model_manager.model_manager import ModelManager
import gc
import torch
import logging
import sys
import threading
import queue
from ChatRTX.logger import ChatRTXLogger
import time

# Initialize logger
ChatRTXLogger(log_level=logging.INFO)

# Define the directory where models will be downloaded
model_download_dir = "C:\\ProgramData\\NVIDIA Corporation\\ChatRTX"

# Initialize the model manager with the specified download directory
model_manager = ModelManager(model_download_dir)

# Get the list of available models and print it
model_list = model_manager.get_model_list()
print(f"Model list: {model_list}")

def jsonify_data(data):
    """Recursively convert data to JSON-serializable format."""
    if isinstance(data, (dict, list, str, int, float, bool, type(None))):
        return data
    elif hasattr(data, "__dict__"):
        # Convert objects with `__dict__` attribute (e.g., custom classes) to a dict
        return {key: jsonify_data(value) for key, value in data.__dict__.items()}
    elif isinstance(data, list):
        # Recursively process lists
        return [jsonify_data(item) for item in data]
    else:
        # Fallback: Convert unknown types (like NodeWithScore) to a string
        return str(data)

def handle_model_setup(model_id): 
    # Check if the specified model is downloaded, if not, download it
    if not model_manager.is_model_downloaded(model_id):
        status = model_manager.download_model(model_id)
        if not status:
            logging.error(f"Model download failed for the model: {model_id}")
            sys.exit(1)

    # Check if the specified model is installed, if not, install it
    if not model_manager.is_model_installed(model_id):
        print("Building TRT-LLM engine....")
        status = model_manager.install_model(model_id)
        if not status:
            logging.error(f"Model installation failed for the model: {model_id}")
            sys.exit(1)

# Define the model ID and handle setup
model_id = "llama2_13b_AWQ_INT4_chat"
handle_model_setup(model_id)

# Get the information about the models
model_info = model_manager.get_model_info()

# Initialize the ChatRTXRag object with the model information and download directory
chat_rtx_rag = ChatRTXRag(model_info, model_download_dir)

# Set up the request queue
request_queue = queue.Queue()

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Background worker function to process requests from the queue
def process_requests():
    while True:
        # Wait for a request from the queue
        request_data, response_queue = request_queue.get()
        if request_data is None:
            # If None is sent to the queue, terminate the worker
            break
        try:
            query_text = request_data.get('query')
            user_ip = request_data.get('user')
            print(f"Processing query from IP {user_ip}: {query_text}")

            # Add audience prompt context to the query
            audience_prompt = "Explain this to me as if I am not familiar with the material: "
            full_query = f"{audience_prompt} {query_text}"

            # Generate the response
            raw_answer = chat_rtx_rag.generate_response(full_query, engine)

            # Include document or filename context if available
            answer = jsonify_data(raw_answer)
            print(answer)
            file_name = answer.get('file_name', 'No file associated with the response')
            print(file_name)
            text = answer.get('response', 'No response generated.')

            # Put the response into the response queue for the request
            response_queue.put({
                "answer": text,
                "file_name": file_name  # Include filename for context if available
            })
            print(f"Response generated: {text} (from {file_name})")
        except Exception as e:
            logging.error(f"Unable to generate a response: {str(e)}")
            response_queue.put({"error": "Unable to generate a response"})
        finally:
            request_queue.task_done()

# Start the background worker thread
worker_thread = threading.Thread(target=process_requests, daemon=True)
worker_thread.start()

try:
    # Initialize the LlamaIndex LLM model with the specified model ID
    status = chat_rtx_rag.init_llamaIndex_llm(model_id)
    if not status:
        sys.exit(1)

    # Set the embedding model
    chat_rtx_rag.set_embedding_model("BAAI/bge-small-en-v1.5", 384)

    # Set the RAG settings
    chat_rtx_rag.set_rag_setting(chunk_size=512, chunk_overlap=200)

    # Generate a query engine for the specified data directory
    #engine = chat_rtx_rag.generate_query_engine("//sbcounty.gov/dpw/SDD/Admin/BAI")
    engine = chat_rtx_rag.generate_query_engine("C:\Coding\ChatRTX-release-0.4.0\ChatRTX_APIs\ChatRTX\datasets\BAI")

    @app.route('/query', methods=['POST'])
    def query():
        data = request.get_json()
        response_queue = queue.Queue()  # Each request has its own response queue

        # Add request data and response queue to the main request queue
        request_queue.put((data, response_queue))

        # Wait for the response from the worker
        response = response_queue.get()
        return jsonify(response), 200 if "answer" in response else 500

    # Run the Flask application
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=5001)

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")

finally:
    # Clear the CUDA cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()

    # Unload the current LLM model
    chat_rtx_rag.unload_llm()
