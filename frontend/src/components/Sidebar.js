// import React, { useEffect, useState } from "react";
// import axios from "axios";
// import UploadBox from "./UploadBox";
// import "../App.css";

// function Sidebar({ dataset, setDataset }) {
//   const [summary, setSummary] = useState(null);
//   const [columns, setColumns] = useState([]);
//   const [preview, setPreview] = useState([]);

//   useEffect(() => {
//     if (!dataset) return;

//     axios.get("http://localhost:5000/summary").then((res) => {
//       setSummary(res.data);
//     });

//     axios.get("http://localhost:5000/columns").then((res) => {
//       setColumns(res.data);
//     });

//     axios.get("http://localhost:5000/preview").then((res) => {
//       setPreview(res.data);
//     });
//   }, [dataset]);

//   return (
//     <div className="sidebar">
//       <UploadBox setDataset={setDataset} />

//       {dataset && (
//         <>
//           <h4>{dataset.name}</h4>

//           {summary && (
//             <div>
//               <p>Rows: {summary.rows}</p>
//               <p>Columns: {summary.columns}</p>
//               <p>Duplicates: {summary.duplicates}</p>
//               <p>Missing: {summary.missing}</p>
//             </div>
//           )}

//           <details>
//             <summary>Columns</summary>
//             {columns.map((c, i) => (
//               <div key={i}>{c}</div>
//             ))}
//           </details>

//           <details>
//             <summary>Preview</summary>
//             {preview.map((row, i) => (
//               <div key={i}>{JSON.stringify(row)}</div>
//             ))}
//           </details>

//           <h4>Agents</h4>
//           <div>Analysis</div>
//           <div>Cleaning</div>
//           <div>Visualization</div>
//           <div>ML</div>
//         </>
//       )}
//     </div>
//   );
// }

// export default Sidebar;





import React, { useEffect, useState } from "react";
import axios from "axios";
import UploadBox from "./UploadBox";
import "../App.css";

const AGENTS = ["Analysis", "Cleaning", "Visualization", "ML"];

function Sidebar({ dataset, setDataset }) {
  const [summary, setSummary]   = useState(null);
  const [columns, setColumns]   = useState([]);
  const [preview, setPreview]   = useState([]);
  const [activeAgent, setActiveAgent] = useState("Analysis");

  useEffect(() => {
    if (!dataset) return;
    axios.get("http://localhost:5000/summary").then((r) => setSummary(r.data));
    axios.get("http://localhost:5000/columns").then((r) => setColumns(r.data));
    axios.get("http://localhost:5000/preview").then((r) => setPreview(r.data));
  }, [dataset]);

  return (
    <div className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">⬡</div>
        <span className="sidebar-brand-name">DataLens AI</span>
      </div>

      {/* Upload */}
      <div className="sidebar-upload-section">
        <div className="sidebar-upload-label">Dataset</div>
        <UploadBox setDataset={setDataset} />
      </div>

      {/* Scrollable body */}
      <div className="sidebar-body">

        {/* File stats card */}
        {dataset && summary && (
          <div className="sidebar-file-card">
            <div className="sidebar-file-name">
              <span className="sidebar-file-dot" />
              {dataset.name}
            </div>
            <div className="sidebar-stat">
              <span className="sidebar-stat-label">Rows</span>
              <span className="sidebar-stat-value">{summary.rows?.toLocaleString()}</span>
            </div>
            <div className="sidebar-stat">
              <span className="sidebar-stat-label">Columns</span>
              <span className="sidebar-stat-value">{summary.columns}</span>
            </div>
            <div className="sidebar-stat">
              <span className="sidebar-stat-label">Duplicates</span>
              <span className={`sidebar-stat-value ${summary.duplicates > 0 ? "warn" : "good"}`}>
                {summary.duplicates}
              </span>
            </div>
            <div className="sidebar-stat">
              <span className="sidebar-stat-label">Missing</span>
              <span className={`sidebar-stat-value ${summary.missing > 0 ? "warn" : "good"}`}>
                {summary.missing}
              </span>
            </div>
          </div>
        )}

        {/* Columns collapsible */}
        {dataset && columns.length > 0 && (
          <details className="sidebar-details">
            <summary>Columns ({columns.length})</summary>
            <div className="sidebar-details-body">
              {columns.map((c, i) => (
                <div key={i} className="sidebar-col-tag">{c}</div>
              ))}
            </div>
          </details>
        )}

        {/* Preview collapsible */}
        {dataset && preview.length > 0 && (
          <details className="sidebar-details">
            <summary>Preview</summary>
            <div className="sidebar-details-body">
              {preview.slice(0, 5).map((row, i) => (
                <div key={i} className="sidebar-preview-row">{JSON.stringify(row)}</div>
              ))}
            </div>
          </details>
        )}

        {/* Agents */}
        {dataset && (
          <>
            <div className="sidebar-section-label">Agents</div>
            <div className="agents-list">
              {AGENTS.map((name) => (
                <div
                  key={name}
                  className={`agent-item ${activeAgent === name ? "active" : ""}`}
                  onClick={() => setActiveAgent(name)}
                >
                  <span className="agent-dot" />
                  {name}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Sidebar;