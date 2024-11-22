import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../App.css';

const Chat = ({ activeTab }) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [userIp, setUserIp] = useState('');

  const apiUrls = {
    IT: 'http://172.26.11.12:5000/query',
    Transportation: 'http://172.26.11.12:5001/query',
    Flood: 'http://172.26.11.12:5002/query',
    'HR & Payroll': 'http://172.26.11.12:5003/query',
  };

  useEffect(() => {
    // Fetch the user's IP address on component mount
    const fetchUserIp = async () => {
      try {
        const res = await axios.get('https://api.ipify.org?format=json');
        setUserIp(res.data.ip);
      } catch (err) {
        console.error('Error fetching IP address:', err);
        setError('Could not fetch IP address.');
      }
    };
    fetchUserIp();
  }, []);

  useEffect(() => {
    // Clear chat messages when the active tab changes
    setMessages([]);
  }, [activeTab]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { type: 'user', text: query };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    setLoading(true);
    setQuery('');
    setError('');

    try {
      const res = await axios.post(apiUrls[activeTab], {
        query,
        user: userIp,
      });
      const aiMessage = { type: 'ai', text: res.data.answer };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (err) {
      setError('Error generating response. Please try again.');
      console.error('Error from API:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <div className="chat-header">{activeTab} Chat</div>

        <div className="message-container">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.type === 'user' ? 'user' : 'ai'}`}
            >
              <p>{message.text}</p>
            </div>
          ))}
          {loading && <p className="loading">Generating response...</p>}
        </div>

        <form className="form-container" onSubmit={handleSubmit}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Ask a question for ${activeTab}...`}
            rows="1"
          />
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? 'Processing...' : 'Send'}
          </button>
        </form>

        {error && <p className="error-message">{error}</p>}
      </div>
    </div>
  );
};

export default Chat;
