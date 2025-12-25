from flask import Flask, request, jsonify
import faiss
import numpy as np
import pandas as pd
import os
import traceback # Required for traceback.print_exc()

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
index_path = os.path.join(base_dir, 'faiss_indices', 'faiss_index.index')
csv_path = os.path.join(base_dir, 'faiss_indices', 'faiss_csv.csv')

# --- LOAD FAISS INDEX ---
try:
    index = faiss.read_index(index_path)
    # Print dimensions to ensure correct testing inputs
    print(f"==========================================")
    print(f"✅ INDEX LOADED SUCCESSFULLY!")
    print(f"👉 Dimension (Required): {index.d}") 
    print(f"👉 Total vectors: {index.ntotal}")
    print(f"==========================================")
except Exception as e:
    print(f"❌ Error loading index: {e}")
    index = None

# --- LOAD CSV MAPPING ---
try:
    csv_df = pd.read_csv(csv_path)
    # Create a dictionary mapping between Faiss index and IDs
    index_to_ids = dict(zip(csv_df['Index'], csv_df['ID']))
    print(f"✅ CSV LOADED SUCCESSFULLY! Mapped {len(index_to_ids)} IDs.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    index_to_ids = {}

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get data from the search API request
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({'error': 'Missing "queries" field'}), 400
        
        # Convert queries to numpy array (float32 is required by Faiss)
        queries = np.array(data['queries'], dtype='float32')
        k = int(data.get('k', 5)) # Default k=5 if not provided

        # Check if Index is loaded
        if index is None:
            return jsonify({'error': 'Index not loaded on server'}), 500

        # Validate Dimensions
        input_dim = queries.shape[1]
        required_dim = index.d

        if input_dim != required_dim:
            error_msg = f"Dimension mismatch: You sent {input_dim} dimensions, but Index requires {required_dim} dimensions."
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400

        # Search in the Faiss index
        distances, indices = index.search(queries, k)
        
        # Retrieve IDs corresponding to the found indices
        matched_IDs = []
        for row in indices:
            row_ids = []
            for idx in row:
                # idx = -1 means neighbor not found
                if idx != -1:
                    row_ids.append(index_to_ids.get(idx, str(idx)))
                else:
                    row_ids.append(None)
            matched_IDs.append(row_ids)

        # Return the response as JSON
        return jsonify(ids=matched_IDs, distances=distances.tolist())

    except Exception as e:
        print("❌ SERVER ERROR:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
