import React from "react";
import axios from "axios";

function UploadBox({ setDataset }) {
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post("http://localhost:5000/upload", formData);
    setDataset({ name: file.name, rows: res.data.rows, cols: res.data.cols });
  };

  return (
    <div className="upload-box">
      <input type="file" accept=".csv,.xlsx,.json" onChange={handleUpload} />
      <span className="upload-box-icon">⊕</span>
      <div className="upload-box-text">Click or drop a file</div>
      <div className="upload-box-hint">CSV · XLSX · JSON</div>
    </div>
  );
}

export default UploadBox;