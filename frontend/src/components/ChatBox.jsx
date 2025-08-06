import React, { useState } from 'react';
import axios from 'axios';

function ChatBox() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!prompt.trim()) return;

    const newMessages = [...messages, { sender: 'user', text: prompt }];
    setMessages(newMessages);
    setPrompt('');
    setLoading(true);

    try {
      const res = await axios.post("https://c25456d7852a.ngrok-free.app/chat", { prompt });
      const botResponse = res.data.response || 'No response.';
      setMessages([...newMessages, { sender: 'bot', text: botResponse }]);
    } catch (err) {
      console.error(err);
      setMessages([...newMessages, { sender: 'bot', text: 'Something went wrong.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '70vh',
      border: '1px solid #ddd',
      borderRadius: '12px',
      overflow: 'hidden',
    }}>
      {/* Message area */}
      <div style={{
        flex: 1,
        padding: '1rem',
        overflowY: 'auto',
        backgroundColor: '#f9fafb'
      }}>
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              display: 'flex',
              justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: '0.75rem'
            }}
          >
            <div style={{
              maxWidth: '75%',
              padding: '0.75rem 1rem',
              borderRadius: '16px',
              backgroundColor: msg.sender === 'user' ? '#d1e7dd' : '#e4e6eb',
              color: '#111',
              whiteSpace: 'pre-wrap',
              boxShadow: '0 1px 4px rgba(0, 0, 0, 0.05)'
            }}>
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ color: '#999', fontStyle: 'italic' }}>Typing...</div>
        )}
      </div>

      {/* Input area */}
      <div style={{
        borderTop: '1px solid #ddd',
        padding: '1rem',
        backgroundColor: '#fff',
        display: 'flex',
        flexDirection: 'row',
        gap: '0.5rem'
      }}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={2}
          placeholder="Type your message..."
          style={{
            flex: 1,
            resize: 'none',
            borderRadius: '8px',
            border: '1px solid #ccc',
            padding: '0.5rem 0.75rem',
            fontFamily: 'inherit',
            fontSize: '1rem'
          }}
        />
        <button
          onClick={handleSend}
          disabled={loading}
          style={{
            padding: '0.5rem 1rem',
            borderRadius: '8px',
            border: 'none',
            backgroundColor: loading ? '#ccc' : '#10a37f',
            color: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default ChatBox;
