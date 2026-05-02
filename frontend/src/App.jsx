import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './index.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

function App() {
  const [stage, setStage] = useState('auth'); // 'auth' | 'dash' | 'chat'
  const [username, setUsername] = useState(null);
  const [messages, setMessages] = useState([]);

  // Auth View
  const AuthView = () => {
    const [isLogin, setIsLogin] = useState(true);
    const [user, setUser] = useState('');
    const [pass, setPass] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
      e.preventDefault();
      setError('');
      setLoading(true);
      try {
        const endpoint = isLogin ? '/auth/login' : '/auth/signup';
        const res = await axios.post(`${API_BASE}${endpoint}`, { username: user, password: pass });
        if (res.data.success) {
          if (isLogin) {
            setUsername(res.data.username);
            setStage('dash');
          } else {
            setIsLogin(true);
            setUser('');
            setPass('');
            setError('Account created safely. Please log in.');
          }
        }
      } catch (err) {
        setError(err.response?.data?.detail || 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    return (
      <div className="auth-container">
        <div className="glass-panel auth-box">
          <h1>{isLogin ? 'Welcome Back' : 'Create Account'}</h1>
          {error && (
             <div className={`alert ${error.includes('safely') ? 'alert-success' : 'alert-error'}`}>
               {error.replace(' safely', '')}
             </div>
          )}
          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <label>Username</label>
              <input type="text" value={user} onChange={(e) => setUser(e.target.value)} required />
            </div>
            <div className="input-group">
              <label>Password</label>
              <input type="password" value={pass} onChange={(e) => setPass(e.target.value)} required />
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? <div className="spinner" /> : (isLogin ? 'Login' : 'Sign Up')}
            </button>
          </form>
          <div className="toggle-auth">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <span onClick={() => setIsLogin(!isLogin)}>{isLogin ? 'Sign up' : 'Login'}</span>
          </div>
        </div>
      </div>
    );
  };

  // Dashboard View for File Upload
  const DashView = () => {
    const [uploading, setUploading] = useState(false);
    const [msg, setMsg] = useState('');

    const handleFileUpload = async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('username', username);
      formData.append('file', file);

      setUploading(true);
      setMsg('');
      try {
        const res = await axios.post(`${API_BASE}/docs/upload`, formData);
        if (res.data.success) {
          setStage('chat');
        }
      } catch (err) {
        setMsg(err.response?.data?.detail || 'Upload failed');
      } finally {
        setUploading(false);
      }
    };

    return (
      <div className="dashboard-container">
        <div className="header glass-panel" style={{ padding: '1rem 2rem', marginBottom: '2rem' }}>
          <h2>Hello, {username} 👋</h2>
          <div>
            <button className="btn btn-secondary" onClick={() => setStage('chat')} style={{ marginRight: '1rem' }}>
              Skip to Chat
            </button>
            <button className="btn btn-secondary" onClick={() => { setUsername(null); setStage('auth'); }}>Logout</button>
          </div>
        </div>
        <div className="glass-panel">
          <h3 style={{ marginBottom: '1.5rem', textAlign: 'center' }}>Knowledge Base Setup</h3>
          {msg && <div className="alert alert-error">{msg}</div>}
          <label className="upload-area" style={{ display: 'block' }}>
            {uploading ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div className="spinner" style={{ width: '40px', height: '40px', borderColor: 'var(--primary)', borderTopColor: 'transparent', marginBottom: '1rem' }} />
                <p>Indexing your document...</p>
              </div>
            ) : (
              <>
                <div className="upload-icon">📄</div>
                <p style={{ fontSize: '1.2rem', fontWeight: 500, marginBottom: '0.5rem' }}>Click to upload PDF</p>
                <p style={{ color: 'var(--text-muted)' }}>Securely isolated in your own vector space</p>
                <input type="file" accept=".pdf" onChange={handleFileUpload} style={{ display: 'none' }} />
              </>
            )}
          </label>
        </div>
      </div>
    );
  };

  // Chat View
  const ChatView = () => {
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const endRef = useRef(null);

    const scrollToBottom = () => {
      endRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
      scrollToBottom();
    }, [messages, loading]);

    const handleSend = async (e) => {
      e.preventDefault();
      if (!input.trim() || loading) return;

      const userMsg = input.trim();
      setInput('');
      const newHistory = [...messages, { role: 'user', content: userMsg }];
      setMessages(newHistory);
      setLoading(true);

      try {
        const res = await axios.post(`${API_BASE}/chat`, {
          username: username,
          question: userMsg,
          history: messages // pass prev history
        });
        
        if (res.data.success) {
          setMessages([...newHistory, { role: 'assistant', content: res.data.answer }]);
        }
      } catch (err) {
        setMessages([...newHistory, { role: 'assistant', content: "Error: " + (err.response?.data?.detail || err.message) }]);
      } finally {
        setLoading(false);
      }
    };

    return (
      <div className="chat-container">
        <div className="chat-header">
          <div>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ color: 'var(--primary)' }}>🧠</span> Neural Chat
            </h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Memory: {Math.floor(messages.length / 2)} exchanges</span>
          </div>
          <div>
            <button className="btn btn-secondary" onClick={() => setMessages([])} style={{ padding: '0.5rem 1rem', marginRight: '0.5rem' }}>Clear</button>
            <button className="btn btn-secondary" onClick={() => setStage('dash')} style={{ padding: '0.5rem 1rem' }}>Back</button>
          </div>
        </div>

        <div className="chat-messages">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: 'auto', marginBottom: 'auto' }}>
               <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>✨</div>
               <p>Ask me anything about your documents</p>
            </div>
          )}
          {messages.map((m, idx) => (
            <div key={idx} className={`message ${m.role}`}>
              <ReactMarkdown>{m.content}</ReactMarkdown>
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        <form className="chat-input-container" onSubmit={handleSend}>
          <input 
            type="text" 
            className="chat-input" 
            placeholder="Type your message..." 
            value={input} 
            onChange={e => setInput(e.target.value)}
          />
          <button type="submit" className="send-btn" disabled={!input.trim() || loading}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </form>
      </div>
    );
  };

  return (
    <>
      {stage === 'auth' && <AuthView />}
      {stage === 'dash' && <DashView />}
      {stage === 'chat' && <ChatView />}
    </>
  );
}

export default App;
