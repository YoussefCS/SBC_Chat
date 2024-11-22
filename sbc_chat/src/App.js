import React, { useState } from 'react';
import './App.css';
import Chat from './components/Chat';
import Sidebar from './components/Sidebar';
import logo from "./images/SBC.png";


function App() {
  const [activeTab, setActiveTab] = useState('IT');

  return (
    <div className="App">
      <div className="top-bar">
        <img src={logo} alt="Logo" className="top-bar-logo" />
        <div className="top-bar-divider"></div>
        <div className="top-bar-title">SBC Chat</div>
      </div>
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="chat-container">
        <Chat activeTab={activeTab} />
      </div>
    </div>
  );
}

export default App;
