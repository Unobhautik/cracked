# Frontend Developer - Quick Start Guide

## 🚀 Getting Started

### Step 1: Create React App
```bash
npx create-react-app frontend --template typescript
cd frontend
npm install axios
npm install @heroicons/react  # For icons (optional)
```

### Step 2: Project Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx      # Main chat component
│   │   ├── MessageBubble.tsx      # Individual message
│   │   ├── AgentIndicator.tsx     # Shows which agent responded
│   │   ├── PDFUpload.tsx          # PDF upload component
│   │   └── EmergencyAlert.tsx     # Emergency warning banner
│   ├── services/
│   │   └── api.ts                 # API client
│   ├── types/
│   │   └── index.ts               # TypeScript types
│   ├── App.tsx
│   └── index.tsx
└── package.json
```

### Step 3: API Client Setup

Create `src/services/api.ts`:
```typescript
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface ChatRequest {
  message: string;
  user_id: string;
  conversation_id?: string;
}

export interface ChatResponse {
  response: string;
  agent_used?: string;
  risk_level?: string;
  is_emergency: boolean;
}

export const chatAPI = {
  sendMessage: async (message: string, userId: string): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/api/chat', {
      message,
      user_id: userId,
    });
    return response.data;
  },

  analyzeSymptoms: async (symptoms: string, userId: string): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/api/symptoms', {
      symptoms,
      user_id: userId,
    });
    return response.data;
  },

  getDrugInfo: async (drugName: string, userId: string): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/api/drug-info', {
      drug_name: drugName,
      user_id: userId,
    });
    return response.data;
  },

  uploadPDF: async (file: File, userId: string): Promise<ChatResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    
    const response = await api.post<ChatResponse>('/api/analyze-pdf', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default api;
```

### Step 4: Chat Interface Component

Create `src/components/ChatInterface.tsx`:
```typescript
import React, { useState, useRef, useEffect } from 'react';
import { chatAPI, ChatResponse } from '../services/api';
import MessageBubble from './MessageBubble';
import EmergencyAlert from './EmergencyAlert';
import PDFUpload from './PDFUpload';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Array<{role: 'user' | 'ai', content: string, agent?: string, isEmergency?: boolean}>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [userId] = useState(`user_${Date.now()}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response: ChatResponse = await chatAPI.sendMessage(userMessage, userId);
      
      setMessages(prev => [...prev, {
        role: 'ai',
        content: response.response,
        agent: response.agent_used,
        isEmergency: response.is_emergency,
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        role: 'ai',
        content: 'Sorry, I encountered an error. Please try again.',
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handlePDFUpload = async (file: File) => {
    setLoading(true);
    try {
      const response: ChatResponse = await chatAPI.uploadPDF(file, userId);
      setMessages(prev => [...prev, {
        role: 'user',
        content: `Uploaded PDF: ${file.name}`,
      }, {
        role: 'ai',
        content: response.response,
        agent: response.agent_used,
      }]);
    } catch (error) {
      console.error('Error uploading PDF:', error);
      setMessages(prev => [...prev, {
        role: 'ai',
        content: 'Error analyzing PDF. Please try again.',
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 shadow-md">
        <h1 className="text-2xl font-bold">MedicalAI Assistant</h1>
        <p className="text-sm text-blue-100">Your intelligent healthcare companion</p>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p className="text-lg">👋 Welcome to MedicalAI!</p>
            <p className="text-sm mt-2">Ask me about symptoms, medications, appointments, or upload a medical PDF.</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx}>
            {msg.isEmergency && <EmergencyAlert />}
            <MessageBubble
              role={msg.role}
              content={msg.content}
              agent={msg.agent}
            />
          </div>
        ))}
        
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg p-4 shadow-md">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <PDFUpload onUpload={handlePDFUpload} />
        <form onSubmit={handleSend} className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
```

### Step 5: Message Bubble Component

Create `src/components/MessageBubble.tsx`:
```typescript
import React from 'react';
import AgentIndicator from './AgentIndicator';

interface MessageBubbleProps {
  role: 'user' | 'ai';
  content: string;
  agent?: string;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ role, content, agent }) => {
  if (role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-blue-600 text-white rounded-lg p-4 max-w-3xl shadow-md">
          <p className="whitespace-pre-wrap">{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="bg-white rounded-lg p-4 max-w-3xl shadow-md">
        {agent && <AgentIndicator agent={agent} />}
        <p className="text-gray-800 whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  );
};

export default MessageBubble;
```

### Step 6: Agent Indicator Component

Create `src/components/AgentIndicator.tsx`:
```typescript
import React from 'react';

interface AgentIndicatorProps {
  agent: string;
}

const agentIcons: Record<string, string> = {
  'symptom_analyzer': '🔍',
  'drug_info': '💊',
  'booking': '📅',
  'tips': '💡',
  'pdf_analyzer': '📄',
  'reminder': '⏰',
};

const agentNames: Record<string, string> = {
  'symptom_analyzer': 'Symptom Analyzer',
  'drug_info': 'Drug Information',
  'booking': 'Appointment Booking',
  'tips': 'Health Tips',
  'pdf_analyzer': 'PDF Analyzer',
  'reminder': 'Medication Reminder',
};

const AgentIndicator: React.FC<AgentIndicatorProps> = ({ agent }) => {
  const icon = agentIcons[agent] || '🤖';
  const name = agentNames[agent] || agent;

  return (
    <div className="flex items-center space-x-2 mb-2 text-sm text-gray-600">
      <span>{icon}</span>
      <span className="font-semibold">{name}</span>
    </div>
  );
};

export default AgentIndicator;
```

### Step 7: Emergency Alert Component

Create `src/components/EmergencyAlert.tsx`:
```typescript
import React from 'react';

const EmergencyAlert: React.FC = () => {
  return (
    <div className="bg-red-600 text-white p-4 rounded-lg mb-4 shadow-lg animate-pulse">
      <div className="flex items-center space-x-2">
        <span className="text-2xl">🚨</span>
        <div>
          <p className="font-bold">EMERGENCY DETECTED</p>
          <p className="text-sm">Please call 911 or go to the emergency room immediately!</p>
        </div>
      </div>
    </div>
  );
};

export default EmergencyAlert;
```

### Step 8: PDF Upload Component

Create `src/components/PDFUpload.tsx`:
```typescript
import React, { useRef } from 'react';

interface PDFUploadProps {
  onUpload: (file: File) => void;
}

const PDFUpload: React.FC<PDFUploadProps> = ({ onUpload }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      onUpload(file);
    } else {
      alert('Please upload a PDF file');
    }
  };

  return (
    <div className="mb-2">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".pdf"
        className="hidden"
        id="pdf-upload"
      />
      <label
        htmlFor="pdf-upload"
        className="inline-block bg-gray-200 text-gray-700 px-4 py-2 rounded-lg cursor-pointer hover:bg-gray-300 text-sm"
      >
        📄 Upload Medical PDF
      </label>
    </div>
  );
};

export default PDFUpload;
```

### Step 9: Update App.tsx

```typescript
import React from 'react';
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  return (
    <div className="App">
      <ChatInterface />
    </div>
  );
}

export default App;
```

### Step 10: Environment Variables

Create `.env` file in frontend root:
```
REACT_APP_API_URL=http://localhost:8000
```

For production:
```
REACT_APP_API_URL=https://your-backend-api.com
```

### Step 11: Run Frontend

```bash
npm start
```

Visit: http://localhost:3000

---

## 🎨 Styling Options

### Option 1: Tailwind CSS (Recommended - Fast)
```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Option 2: Material-UI
```bash
npm install @mui/material @emotion/react @emotion/styled
```

### Option 3: Plain CSS
Just use the classes in the examples above with your own CSS.

---

## ✅ Checklist

- [ ] React app created
- [ ] API client set up
- [ ] Chat interface working
- [ ] PDF upload working
- [ ] Emergency alerts showing
- [ ] Agent indicators showing
- [ ] Responsive design
- [ ] Connected to backend API
- [ ] Error handling
- [ ] Loading states

---

## 🐛 Common Issues

**CORS Error:**
- Make sure backend has CORS enabled
- Check API URL in `.env`

**API Not Responding:**
- Check backend is running
- Check API URL is correct
- Check network tab in browser dev tools

**PDF Upload Fails:**
- Check file size limits
- Check backend accepts multipart/form-data
- Check file is actually a PDF

---

Good luck! 🚀


