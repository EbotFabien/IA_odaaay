from flask import Flask, request, jsonify
from functions import embedder, get_conversation_chain, get_text_chunks

app = Flask(__name__)

# Dummy POST route 1: Add User
@app.route('/ai/embedding', methods=['POST'])
def embedding():
    """
    Expected JSON for this route:
    {
        "text": "my name is john",
       
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Extract data
    text = data.get('text')
   

    if not text :
        return jsonify({"error": "Missing required fields:'text'"}), 400

    # Process data
    vector_data=embedder(text)
    
    return jsonify({
        "message": "Order submitted successfully",
        "data": {
            "vector": vector_data,
        }
    }), 201

# Dummy POST route 2: Submit Order
@app.route('/ai/chunks', methods=['POST'])
def chunks():
    """
    This route provides chunks and embedding
    Expected JSON for this route:
    {
        "content": 'This is the content to be passed for chunks'
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Extract data
    content = data.get('content')
   

    if not content :
        return jsonify({"error": "Missing required fields"}), 400

    # Process data

    segments=get_text_chunks(content)
    vector_data=embedder(segments)
    
    return jsonify({
        "message": "Order submitted successfully",
        "data": {
            "vector": vector_data,
        }
    }), 201

# Dummy GET route: Get Items
@app.route('/ai/response', methods=['POST'])
def get_response():
    """
    Returns an AI response.
    Expected JSON for this route:
    {
        "document": 'This is the content to be passed for chunks',
        'text':'The searchable text'
    }
    """
    #done
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Extract data
    documents = data.get('document')
    text= data.get('text')
   
    response=get_conversation_chain(documents,text)

    return jsonify({
        "answer": str(response.memory.chat_memory)
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
