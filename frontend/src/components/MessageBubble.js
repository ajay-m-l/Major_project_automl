import React from "react";
import "../App.css";

function MessageBubble({ msg }) {
  return (
    <div className={`message ${msg.role}`}>
      <div style={{ whiteSpace: "pre-wrap" }}>{msg.text}</div>

      {msg.image && (
        <img
          src={`data:image/png;base64,${msg.image}`}
          alt="Generated plot"
        />
      )}
    </div>
  );
}

export default MessageBubble;