import React, { useState } from "react";
import "./App.css";

import Sidebar from "./components/Sidebar";
import ChatWindow from "./components/ChatWindow";

function App() {
  const [dataset, setDataset] = useState(null);
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);

  return (
    <div className="app">
      {/* LEFT SIDEBAR */}
      <Sidebar dataset={dataset} setDataset={setDataset} />

      {/* RIGHT CHAT */}
      <ChatWindow
        chat={chat}
        setChat={setChat}
        loading={loading}
        setLoading={setLoading}
        dataset={dataset}
      />
    </div>
  );
}

export default App;