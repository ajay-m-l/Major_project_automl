// import React, { useState } from "react";
// import axios from "axios";
// import MessageBubble from "./MessageBubble";
// import "../App.css";

// function ChatWindow({ chat, setChat, loading, setLoading, dataset }) {
//   const [input, setInput] = useState("");

//   const sendMessage = async () => {
//     if (!input || !dataset) return;

//     setChat((prev) => [...prev, { role: "user", text: input }]);
//     setLoading(true);

//     try {
//       const res = await axios.post("http://localhost:5000/chat", {
//         query: input,
//       });

//       setChat((prev) => [
//         ...prev,
//         {
//           role: "bot",
//           text: res.data.response,
//           image: res.data.image,
//         },
//       ]);
//     } catch (err) {
//       setChat((prev) => [
//         ...prev,
//         { role: "bot", text: "❌ Error connecting to backend" },
//       ]);
//     }

//     setInput("");
//     setLoading(false);
//   };

//   return (
//   <div className="chat-window">

//     {/* 🔥 STICKY HEADER */}
//     <div className="chat-header">
//       <div className="glass-card">
//         <h1>Automated Data Analysis</h1>
//         <p>
//           Upload your dataset and explore insights, clean data,
//           visualize patterns, and build models using AI.
//         </p>
//       </div>
//     </div>

//     {/* CHAT */}
//     <div className="chat-messages">
//       {chat.map((msg, i) => (
//         <div key={i} className="chat-row">
//           <MessageBubble msg={msg} />
//         </div>
//       ))}

//       {loading && (
//         <div className="chat-row">
//           <div className="message bot">Thinking...</div>
//         </div>
//       )}
//     </div>

//     {/* INPUT */}
//     <div className="chat-input">
//       <div className="input-inner">
//         <input
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           placeholder="Ask anything..."
//         />
//         <button onClick={sendMessage}>Send</button>
//       </div>
//     </div>
//   </div>
// );
// }

// export default ChatWindow;






import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import MessageBubble from "./MessageBubble";
import "../App.css";

function ChatWindow({ chat, setChat, loading, setLoading, dataset }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat, loading]);

  const sendMessage = async () => {
    if (!input.trim() || !dataset) return;

    const userMsg = input.trim();
    setChat((prev) => [...prev, { role: "user", text: userMsg }]);
    setLoading(true);
    setInput("");

    try {
      const res = await axios.post("http://localhost:5000/chat", { query: userMsg });
      setChat((prev) => [
        ...prev,
        { role: "bot", text: res.data.response, image: res.data.image },
      ]);
    } catch {
      setChat((prev) => [
        ...prev,
        { role: "bot", text: "Error connecting to backend. Please try again." },
      ]);
    }

    setLoading(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-window">

      {/* Sticky header */}
      <div className="chat-header">
        <div className="glass-card">
          <div className="chat-header-left">
            <h1>DataLens AI – Multi-Agent Analytics</h1>
            <p>Interact with AI agents to explore, clean, and model your data</p>
          </div>
          <div className="chat-header-badge">Multi-Agent</div>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {chat.length === 0 && !loading && (
          <div className="chat-intro">
            <div className="intro-card">
              <div className="intro-card-glyph">⬡</div>
              <h1>What would you like to explore?</h1>
              <p>
                {dataset
                  ? `${dataset.name} is loaded and ready. Ask anything about your data.`
                  : "Upload a dataset from the sidebar to get started."}
              </p>
              {dataset && (
                <div className="intro-chips">
                  {["give me the data summary", "show missing values", "plot price distribution", "train a regression model"].map((q) => (
                    <span key={q} className="intro-chip" onClick={() => setInput(q)}>{q}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {chat.map((msg, i) => (
          <div key={i} className={`chat-row ${msg.role}`}>
            <MessageBubble msg={msg} />
          </div>
        ))}

        {loading && (
          <div className="chat-row bot">
            <div className="message bot thinking">Thinking</div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="chat-input">
        <div className="input-inner">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={dataset ? "Ask anything about your dataset…" : "Upload a dataset first…"}
            disabled={!dataset}
          />
          <span className="input-hint"></span>
          <button onClick={sendMessage} disabled={!dataset || !input.trim()}>
            Send →
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatWindow;