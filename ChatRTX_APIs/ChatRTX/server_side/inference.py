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

from flask import Flask, request, jsonify
from flask_cors import CORS 
from ChatRTX.chatrtx import ChatRTX
from ChatRTX.model_manager.model_manager import ModelManager
import logging
import sys
from ChatRTX.logger import ChatRTXLogger

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This allows all origins (including http://localhost:3000)

# If you want to restrict CORS to a specific origin (e.g., only allow http://localhost:3000), do:
# CORS(app, resources={r"/query": {"origins": "http://localhost:3000"}})

# Initialize logger
ChatRTXLogger(log_level=logging.INFO)
logger = ChatRTXLogger.get_logger()

# Define model manager and download directory
model_download_dir = "C:\\Coding\\ChatRTX-release-0.4.0\\ChatRTX_APIs\\ChatRTX\\model_manager"
model_manager = ModelManager(model_download_dir)

# Initialize the model manager and chat
model_list = model_manager.get_model_list()
model_id = "mistral_7b_AWQ_int4_chat"

if not model_manager.is_model_downloaded(model_id):
    status = model_manager.download_model(model_id)
    if not status:
        logger.error(f"Model download failed for the model: {model_id}")
        sys.exit(1)

if not model_manager.is_model_installed(model_id):
    print("Building TRT-LLM engine....")
    status = model_manager.install_model(model_id)
    if not status:
        logger.error(f"Model installation failed for the model: {model_id}")
        sys.exit(1)

model_info = model_manager.get_model_info()
chat_rtx = ChatRTX(model_info, model_download_dir)
status = chat_rtx.init_llm_model(model_id, add_special_tokens=True, use_py_session=True)
if not status:
    logger.error(f"Failed to load the model: {model_id}")
    sys.exit(1)

@app.route('/query', methods=['POST'])
def query():
    try:
        # Get user input from the request
        user_query = request.json.get('query')
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        # Generate a response from the model


        # store in a queue w/ id, query, timestamp
        #my_task_data = {}
        #return jsonify({'response': my_task_data}) # that data in response
    
        response = chat_rtx.generate_response(user_query)
        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({'error': 'Unable to generate response'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
