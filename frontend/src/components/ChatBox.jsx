import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

function ChatBox() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const API_ROOT = 'https://57a122d2323d.ngrok-free.app';

  const onboardingQuestions = [
    "Hi, before we get started, I have a couple of questions! Would you like me to speak formally or informally (please respond with the word 'formal' or 'informal')?",
    "Do you prefer learning in a more exploratory way or a more guided way (please respond with the word 'exploratory' or 'guided')?",
    "Ok perfect, go ahead and ask what you'd like about the CN Tower!"
  ];

  const [onboardingIndex, setOnboardingIndex] = useState(0);
  const [onboardingAnswers, setOnboardingAnswers] = useState([]);
  const [isOnboardingComplete, setIsOnboardingComplete] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    setMessages([{ sender: "bot", text: onboardingQuestions[0] }]);
  }, []);

  useEffect(() => {
    if (isOnboardingComplete) {
      axios.post(`${API_ROOT}/onboarding`, {
        answers: onboardingAnswers
      }).then(() => {
        console.log("Onboarding answers sent successfully.");
      }).catch((err) => {
        console.error("Failed to send onboarding answers:", err);
      });
    }
  }, [isOnboardingComplete]);

  const handleSend = async () => {
    if (!prompt.trim()) return;

    const newMessages = [...messages, { sender: "user", text: prompt }];

    if (!isOnboardingComplete && onboardingIndex < onboardingQuestions.length) {
      if (onboardingIndex < onboardingQuestions.length - 1) {
        setOnboardingAnswers([...onboardingAnswers, prompt]);
        setMessages([
          ...newMessages,
          { sender: "bot", text: onboardingQuestions[onboardingIndex + 1] }
        ]);
        setPrompt("");
        setOnboardingIndex(onboardingIndex + 1);

        if (onboardingIndex + 1 === onboardingQuestions.length) {
          setIsOnboardingComplete(true);
        }
        return;
      }

      setIsOnboardingComplete(true);
      setMessages(newMessages);
      setPrompt('');
      setLoading(true);

      try {
        const res = await axios.post(`${API_ROOT}/chat`, { prompt });
        setMessages([...newMessages, { sender: "bot", text: res.data.response }]);
      } catch (error) {
        setMessages([...newMessages, { sender: "bot", text: "Sorry, something went wrong." }]);
      }

      setLoading(false);
      return;
    }

    setMessages(newMessages);
    setPrompt('');
    setLoading(true);

    try {
      const res = await axios.post(`${API_ROOT}/chat`, { prompt });
      setMessages([...newMessages, { sender: "bot", text: res.data.response }]);
    } catch (error) {
      setMessages([...newMessages, { sender: "bot", text: "Sorry, something went wrong." }]);
    }

    setLoading(false);
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '60vh',
      width: '50vw',       
      maxWidth: '1200px',  
      margin: 'auto',
      border: '1px solid #ddd',
      borderRadius: '12px',
      overflow: 'hidden',
      fontFamily: 'Arial, sans-serif',
    }}>
      <div style={{
        flex: 1,
        padding: '1rem',
        overflowY: 'auto',
        backgroundColor: '#f9fafb',
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
        <div ref={messagesEndRef} />
      </div>

      <div style={{
        borderTop: '1px solid #ddd',
        padding: '1rem',
        backgroundColor: '#fff',
        display: 'flex',
        flexDirection: 'row',
        gap: '0.5rem',
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
            fontSize: '1rem',
          }}
          disabled={loading}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
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
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default ChatBox;
