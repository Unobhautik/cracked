# AgentMed Frontend

A modern, interactive web interface for the AgentMed healthcare AI assistant.

## Features

### 🎨 Modern UI/UX
- Beautiful gradient design with smooth animations
- Responsive layout that works on all devices
- Real-time WebSocket communication
- Interactive agent selection
- File upload for PDF analysis

### 🏥 Healthcare Features
- **Symptom Analysis** - Get medically-grounded guidance
- **Drug Information** - Comprehensive medication info and interactions
- **Appointment Management** - Book, cancel, and reschedule appointments
- **PDF Analysis** - Upload and analyze medical documents
- **Medication Reminders** - Set up medication schedules
- **Health Tips** - Personalized wellness advice

### 🔒 Safety Features
- Emergency detection and alerts
- Risk level indicators
- Safety disclaimers
- HIPAA-compliant design

### ⚙️ User Features
- Chat history persistence
- Settings customization
- Agent selection
- Quick action buttons
- Voice input support (UI ready)
- File upload for PDFs

## Getting Started

### Prerequisites
- Python 3.8+
- FastAPI and dependencies installed
- AgentMed backend running

### Running the Frontend

1. **Start the backend server:**
   ```bash
   cd frontend
   python app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8000`

### Development

The frontend consists of:
- `index.html` - Main HTML structure
- `static/style.css` - Modern CSS styling
- `static/script.js` - Interactive JavaScript
- `app.py` - FastAPI backend server

## Usage

### Basic Chat
1. Type your message in the input field
2. Press Enter or click the send button
3. The AI assistant will respond using the appropriate agent

### Quick Actions
Click any quick action button in the sidebar to:
- Analyze symptoms
- Get drug information
- Book appointments
- Get health tips
- Set medication reminders
- Analyze PDF documents

### Upload PDF
1. Click the PDF upload button (📄) in the input toolbar
2. Select a PDF file
3. Wait for upload confirmation
4. Ask questions about the PDF

### Agent Selection
1. Click on an agent in the sidebar
2. Or use the agent selection modal
3. The selected agent will be used for your queries

### Settings
1. Click the settings icon (⚙️) in the header
2. Toggle options:
   - Sound notifications
   - Auto-scroll
   - Show timestamps

## API Endpoints

- `GET /` - Serve frontend
- `POST /api/chat` - Send chat message
- `WS /ws` - WebSocket for real-time chat
- `GET /api/agents` - Get available agents
- `POST /api/upload-pdf` - Upload PDF file
- `GET /api/health` - Health check

## File Structure

```
frontend/
├── app.py                 # FastAPI backend server
├── index.html             # Main HTML
├── static/
│   ├── style.css         # Styles
│   └── script.js         # JavaScript
├── uploads/              # PDF uploads (auto-created)
└── README.md             # This file
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Notes

- Chat history is stored in browser localStorage
- PDF uploads are stored in `frontend/uploads/`
- WebSocket automatically reconnects on disconnect
- All medical information is for guidance only - not a substitute for professional care

## Troubleshooting

**WebSocket not connecting:**
- Check that the backend is running
- Check browser console for errors
- Verify port 8000 is not blocked

**PDF upload fails:**
- Ensure file is a valid PDF
- Check file size (should be reasonable)
- Verify uploads directory exists and is writable

**Agents not loading:**
- Check backend is initialized
- Verify `/api/agents` endpoint is accessible
- Check browser console for errors

## License

Part of the AgentMed healthcare AI system.
