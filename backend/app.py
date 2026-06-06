import sys
import os


# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt

from agents.executor import agent_system
from tools import analysis_tool, visualization_tool, cleaning_tool, ml_tool

app = Flask(__name__)
CORS(app)

# Global dataset
current_df = None


# ─────────────────────────────────────────────
# 🔧 Helper: Normalize column names
# ─────────────────────────────────────────────
def normalize_column_labels(df):
    cols = [str(c) for c in df.columns]
    seen = {}
    new_cols = []

    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")

    df.columns = new_cols
    return df


# ─────────────────────────────────────────────
# 🔧 Helper: Convert plot to base64
# ─────────────────────────────────────────────
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ─────────────────────────────────────────────
# 📂 Upload API
# ─────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload_file():
    global current_df

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = None

        # Try multiple encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding, low_memory=False)
                break
            except Exception:
                continue

        if df is None:
            return jsonify({"error": "Failed to read CSV"}), 400

        df = normalize_column_labels(df)
        current_df = df

        # Set dataset to tools
        analysis_tool.set_dataset(df)
        visualization_tool.set_dataset(df)
        cleaning_tool.set_dataset(df)
        ml_tool.set_dataset(df)

        # Reset agent system
        agent_system.clear_memory()
        agent_system.load_dataset(df, name="uploaded")

        return jsonify({
            "message": "Dataset loaded successfully",
            "rows": df.shape[0],
            "cols": df.shape[1]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 💬 Chat API
# ─────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    global current_df

    data = request.json
    query = data.get("query")

    print("🔥 Received query:", query)

    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400

    try:
        result = agent_system.run(query)
        print("🔥 Result:", result)

        figure = result.get("figure")
        image = None

        if figure:
            image = fig_to_base64(figure)
            plt.close(figure)

        return jsonify({
            "response": result.get("final_response") or "⚠️ No response generated",
            "agent": result.get("selected_agent"),
            "intent": result.get("intent"),
            "image": image
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({
            "response": "❌ Error processing request",
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────
# 📊 Sidebar APIs
# ─────────────────────────────────────────────

@app.route("/summary", methods=["GET"])
def summary():
    global current_df
    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400

    return jsonify({
        "rows": int(current_df.shape[0]),
        "columns": int(current_df.shape[1]),
        "duplicates": int(current_df.duplicated().sum()),
        "missing": int(current_df.isnull().sum().sum())
    })


@app.route("/columns", methods=["GET"])
def columns():
    global current_df
    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400

    return jsonify(list(current_df.columns))


@app.route("/preview", methods=["GET"])
def preview():
    global current_df
    if current_df is None:
        return jsonify({"error": "No dataset loaded"}), 400

    return jsonify(current_df.head(5).to_dict(orient="records"))


# ─────────────────────────────────────────────
# 🏠 Health check
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return "✅ AutoML API Running"


# ─────────────────────────────────────────────
# ▶️ Run server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)