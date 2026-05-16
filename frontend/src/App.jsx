import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Loader2, Sparkles, Briefcase, FileText, ChevronRight } from 'lucide-react';
import './App.css';

// API base URL – change to your deployed backend when ready
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (userMessage) => {
    if (!userMessage.trim()) return;
    const newMessages = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/chat`, {
        messages: newMessages
      });
      const { reply, recommendations, end_of_conversation } = response.data;
      setMessages([...newMessages, {
        role: 'assistant',
        content: reply,
        recommendations: recommendations || []
      }]);
      if (end_of_conversation) {
        // Optionally disable further input or show a message
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages([...newMessages, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        recommendations: []
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  const renderRecommendations = (recs) => {
    if (!recs || recs.length === 0) return null;
    return (
      <div className="recommendations">
        <div className="rec-header">
          <Sparkles size={16} />
          <span>Recommended assessments</span>
        </div>
        <div className="rec-list">
          {recs.map((rec, idx) => (
            <a key={idx} href={rec.url} target="_blank" rel="noopener noreferrer" className="rec-card">
              <div className="rec-name">{rec.name}</div>
              <div className="rec-type">{rec.test_type}</div>
              <ChevronRight size={16} className="rec-arrow" />
            </a>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="header">
          <div className="logo">
            <Briefcase size={24} />
            <span>SHL Assessment Advisor</span>
          </div>
          <div className="badge">AI‑powered</div>
        </div>
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <FileText size={48} />
              <h2>Hello, I’m your SHL assessment advisor</h2>
              <p>Describe the role you’re hiring for, and I’ll recommend the right SHL tests – from personality to technical knowledge.</p>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-content">
                <div className="text">{msg.content}</div>
                {msg.recommendations && renderRecommendations(msg.recommendations)}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g., I need a personality test for a senior sales manager"
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            {loading ? <Loader2 size={20} className="spin" /> : <Send size={20} />}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;