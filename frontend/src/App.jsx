import React from 'react';
import ChatBox from './components/ChatBox';

function App() {
  return (
    <div style={{
      minHeight: '100vh',
      padding: '2rem',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      overflow: 'hidden',  // Prevent page scroll
    }}>
      <div style={{
        maxWidth: '800px',
        width: '100%',
        backgroundColor: '#ffffff',
        padding: '2rem',
        borderRadius: '16px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.05)',
        height: '90vh',           // Fix height to limit container size
        display: 'flex',
        flexDirection: 'column',
      }}>
        <h1 style={{ marginBottom: '0.5rem', fontSize: '2rem' }}>CN Tower Chatbot</h1>
        <h2 style={{
          fontWeight: '400',
          fontSize: '1rem',
          color: '#555',
          marginBottom: '1.5rem',
          lineHeight: '1.5'
        }}>
          Treat this chatbot like a peer or a teacher, but please note that it is simply a tool and not a replacement for educators.
        </h2>
        <ChatBox />
      </div>
    </div>
  );
}

export default App;
