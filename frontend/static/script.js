// AgentMed Frontend JavaScript - Enhanced Version
class AgentMedApp {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.chatHistory = [];
        this.selectedAgent = null;
        this.settings = {
            soundEnabled: true,
            autoScroll: true,
            showTimestamps: true
        };
        this.uploadedPdfPath = null;
        this.init();
    }

    init() {
        this.loadSettings();
        this.setupEventListeners();
        this.loadAgents();
        this.loadChatHistory();
        this.connectWebSocket();
        this.setupFileUpload();
    }

    setupEventListeners() {
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');
        const actionButtons = document.querySelectorAll('.action-btn');
        const clearChatBtn = document.getElementById('clearChatBtn');
        const settingsBtn = document.getElementById('settingsBtn');
        const sidebarToggle = document.getElementById('sidebarToggle');
        const uploadPdfBtn = document.getElementById('uploadPdfBtn');
        const emergencyClose = document.getElementById('emergencyClose');
        const closeAgentModal = document.getElementById('closeAgentModal');
        const closeSettingsModal = document.getElementById('closeSettingsModal');

        // Send button
        sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key to send
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        messageInput.addEventListener('input', () => {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
        });

        // Quick action buttons
        actionButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                this.handleQuickAction(action);
            });
        });

        // Clear chat
        clearChatBtn.addEventListener('click', () => this.clearChat());

        // Settings
        settingsBtn.addEventListener('click', () => this.showSettingsModal());

        // Sidebar toggle
        sidebarToggle.addEventListener('click', () => this.toggleSidebar());

        // File upload
        uploadPdfBtn.addEventListener('click', () => {
            document.getElementById('pdfFileInput').click();
        });

        // Emergency banner close
        if (emergencyClose) {
            emergencyClose.addEventListener('click', () => {
                document.getElementById('emergencyBanner').style.display = 'none';
            });
        }

        // Modal closes
        if (closeAgentModal) {
            closeAgentModal.addEventListener('click', () => {
                document.getElementById('agentModal').style.display = 'none';
            });
        }

        if (closeSettingsModal) {
            closeSettingsModal.addEventListener('click', () => {
                this.hideSettingsModal();
            });
        }

        // Close modals on overlay click
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    overlay.style.display = 'none';
                }
            });
        });

        // Settings checkboxes
        const soundCheckbox = document.getElementById('soundEnabled');
        const autoScrollCheckbox = document.getElementById('autoScroll');
        const timestampsCheckbox = document.getElementById('showTimestamps');

        if (soundCheckbox) {
            soundCheckbox.addEventListener('change', (e) => {
                this.settings.soundEnabled = e.target.checked;
                this.saveSettings();
            });
        }

        if (autoScrollCheckbox) {
            autoScrollCheckbox.addEventListener('change', (e) => {
                this.settings.autoScroll = e.target.checked;
                this.saveSettings();
            });
        }

        if (timestampsCheckbox) {
            timestampsCheckbox.addEventListener('change', (e) => {
                this.settings.showTimestamps = e.target.checked;
                this.saveSettings();
                this.updateMessageTimestamps();
            });
        }
    }

    setupFileUpload() {
        const fileInput = document.getElementById('pdfFileInput');
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleFileUpload(file);
            }
        });
    }

    async handleFileUpload(file) {
        if (!file.type.includes('pdf')) {
            this.showNotification('Please upload a PDF file', 'error');
            return;
        }

        // Show upload progress
        const uploadProgress = document.createElement('div');
        uploadProgress.className = 'file-upload-progress';
        uploadProgress.innerHTML = `<i class="fas fa-spinner"></i> Uploading ${file.name}...`;
        const inputArea = document.querySelector('.input-area');
        inputArea.insertBefore(uploadProgress, inputArea.firstChild);

        try {
            // Upload to server
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/upload-pdf', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.success) {
                this.uploadedPdfPath = data.file_path;
                
                // Add message about PDF upload
                this.addMessage(`I've uploaded ${file.name}. You can ask me questions about it!`, 'assistant');
                
                // Auto-send a message about the PDF
                setTimeout(() => {
                    const messageInput = document.getElementById('messageInput');
                    messageInput.value = `Please analyze this PDF file: ${file.name}`;
                    this.sendMessage();
                }, 1000);

                this.showNotification('PDF uploaded successfully!', 'success');
            } else {
                throw new Error(data.message || 'Upload failed');
            }
        } catch (error) {
            this.showNotification('Failed to upload PDF', 'error');
            console.error('Upload error:', error);
        } finally {
            uploadProgress.remove();
        }
    }

    async loadAgents() {
        try {
            const response = await fetch('/api/agents');
            const data = await response.json();
            this.renderAgents(data.agents);
        } catch (error) {
            console.error('Error loading agents:', error);
            this.renderAgentsError();
        }
    }

    renderAgents(agents) {
        const agentsList = document.getElementById('agentsList');
        agentsList.innerHTML = agents.map(agent => `
            <div class="agent-item ${this.selectedAgent === agent.id ? 'active' : ''}" 
                 data-agent-id="${agent.id}"
                 onclick="agentMedApp.selectAgent('${agent.id}')">
                ${agent.name}
            </div>
        `).join('');

        // Also populate modal
        const modalBody = document.getElementById('agentModalBody');
        if (modalBody) {
            modalBody.innerHTML = agents.map(agent => `
                <div class="agent-card ${this.selectedAgent === agent.id ? 'active' : ''}" 
                     data-agent-id="${agent.id}"
                     onclick="agentMedApp.selectAgent('${agent.id}')">
                    <div class="agent-card-header">
                        <div class="agent-card-icon">${this.getAgentIcon(agent.id)}</div>
                        <div class="agent-card-title">${agent.name}</div>
                    </div>
                    <div class="agent-card-description">${agent.description}</div>
                </div>
            `).join('');
        }
    }

    getAgentIcon(agentId) {
        const icons = {
            'booking': '📅',
            'cancellation': '❌',
            'reschedule': '🔄',
            'pdf_analyzer': '📄',
            'reminder': '⏰',
            'tips': '💡',
            'symptom_analyzer': '🔍',
            'drug_info': '💊'
        };
        return icons[agentId] || '🤖';
    }

    renderAgentsError() {
        const agentsList = document.getElementById('agentsList');
        agentsList.innerHTML = '<div class="agent-loading"><span>Failed to load agents</span></div>';
    }

    async selectAgent(agentId) {
        this.selectedAgent = agentId;
        const agents = await this.getAgentsData();
        this.renderAgents(agents);
        const modal = document.getElementById('agentModal');
        if (modal) modal.style.display = 'none';
        this.showNotification(`Selected ${this.getAgentName(agentId)}`, 'success');
    }

    async getAgentsData() {
        try {
            const response = await fetch('/api/agents');
            return (await response.json()).agents;
        } catch (error) {
            return [];
        }
    }

    getAgentName(agentId) {
        const names = {
            'booking': 'Booking Agent',
            'cancellation': 'Cancellation Agent',
            'reschedule': 'Reschedule Agent',
            'pdf_analyzer': 'PDF Analyzer',
            'reminder': 'Reminder Agent',
            'tips': 'Health Tips',
            'symptom_analyzer': 'Symptom Analyzer',
            'drug_info': 'Drug Information'
        };
        return names[agentId] || 'Agent';
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateStatus('Connected', true);
                this.showNotification('Connected to AgentMed', 'success');
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('Connection Error', false);
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateStatus('Disconnected', false);
                // Reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.updateStatus('Connection Failed', false);
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message) return;

        // Add user message to chat
        this.addMessage(message, 'user');
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Show typing indicator
        this.showTypingIndicator();

        // Prepare message with context
        let enhancedMessage = message;
        if (this.uploadedPdfPath && message.toLowerCase().includes('pdf')) {
            enhancedMessage = `Please analyze the PDF at ${this.uploadedPdfPath}. ${message}`;
        }

        // Send via WebSocket if connected, otherwise use REST API
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ message: enhancedMessage }));
        } else {
            // Fallback to REST API
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: enhancedMessage })
                });
                const data = await response.json();
                this.handleResponse(data);
            } catch (error) {
                this.hideTypingIndicator();
                this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
                console.error('Error:', error);
            }
        }
    }

    handleWebSocketMessage(data) {
        this.hideTypingIndicator();

        if (data.type === 'emergency') {
            this.showEmergencyBanner(data.message);
            if (this.settings.soundEnabled) {
                this.playEmergencySound();
            }
        } else if (data.type === 'response') {
            this.addMessage(data.message, 'assistant', data.risk_level);
            if (data.emergency) {
                this.showEmergencyBanner('This appears to be a medical emergency. Please seek immediate medical attention.');
            }
        } else if (data.type === 'error') {
            this.addMessage(data.message, 'assistant');
        }
    }

    handleResponse(data) {
        this.hideTypingIndicator();
        this.addMessage(data.response, 'assistant', data.risk_level);
        
        if (data.emergency) {
            this.showEmergencyBanner('This appears to be a medical emergency. Please seek immediate medical attention.');
        }
    }

    addMessage(text, sender, riskLevel = null) {
        const chatMessages = document.getElementById('chatMessages');
        
        // Remove welcome message if exists
        const welcomeMsg = chatMessages.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = sender === 'user' ? '👤' : '🏥';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        let riskBadge = '';
        if (riskLevel && riskLevel !== 'low' && sender === 'assistant') {
            const riskClass = `risk-${riskLevel}`;
            riskBadge = `<div class="risk-badge ${riskClass}">
                <i class="fas fa-exclamation-triangle"></i>
                ${riskLevel.toUpperCase()} RISK
            </div>`;
        }

        const timestampHtml = this.settings.showTimestamps 
            ? `<div class="message-time">
                <i class="fas fa-clock"></i>
                ${time}
            </div>` 
            : '';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(text)}</div>
                ${timestampHtml}
                ${riskBadge}
            </div>
        `;

        chatMessages.appendChild(messageDiv);
        
        if (this.settings.autoScroll) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Animate message
        messageDiv.style.opacity = '0';
        setTimeout(() => {
            messageDiv.style.transition = 'opacity 0.3s';
            messageDiv.style.opacity = '1';
        }, 10);

        // Save to history
        this.chatHistory.push({ text, sender, time, riskLevel });
        this.saveChatHistory();
    }

    formatMessage(text) {
        // Convert markdown-like formatting to HTML
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        text = text.replace(/\n/g, '<br>');
        // Convert URLs to links
        text = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
        return text;
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">🏥</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        if (this.settings.autoScroll) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.opacity = '0';
            setTimeout(() => typingIndicator.remove(), 300);
        }
    }

    showEmergencyBanner(message) {
        const banner = document.getElementById('emergencyBanner');
        const text = document.getElementById('emergencyText');
        if (banner && text) {
            text.textContent = message;
            banner.style.display = 'block';
            
            // Auto-hide after 15 seconds
            setTimeout(() => {
                banner.style.display = 'none';
            }, 15000);
        }
    }

    handleQuickAction(action) {
        const prompts = {
            symptom: "I need help analyzing my symptoms. ",
            drug: "I need information about a medication. ",
            book: "I need to book a medical appointment. ",
            tips: "I need health and wellness tips. ",
            reminder: "I need to set up a medication reminder. ",
            pdf: "I want to analyze a medical PDF document. "
        };

        const messageInput = document.getElementById('messageInput');
        const prompt = prompts[action] || '';
        messageInput.value = prompt;
        messageInput.focus();
        
        // Highlight the action button
        const actionBtn = document.querySelector(`[data-action="${action}"]`);
        if (actionBtn) {
            actionBtn.style.transform = 'scale(0.95)';
            setTimeout(() => {
                actionBtn.style.transform = '';
            }, 200);
        }
    }

    updateStatus(text, isConnected) {
        const statusText = document.querySelector('.status-text');
        const statusDot = document.querySelector('.status-dot');
        if (statusText) statusText.textContent = text;
        if (statusDot) {
            statusDot.style.background = isConnected ? 'var(--secondary-color)' : 'var(--danger-color)';
        }
    }

    saveChatHistory() {
        try {
            localStorage.setItem('agentmed_chat_history', JSON.stringify(this.chatHistory));
        } catch (error) {
            console.error('Error saving chat history:', error);
        }
    }

    loadChatHistory() {
        try {
            const saved = localStorage.getItem('agentmed_chat_history');
            if (saved) {
                this.chatHistory = JSON.parse(saved);
                // Optionally restore last few messages
                if (this.chatHistory.length > 0) {
                    const recent = this.chatHistory.slice(-5);
                    recent.forEach(msg => {
                        this.addMessage(msg.text, msg.sender, msg.riskLevel);
                    });
                }
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            this.chatHistory = [];
            localStorage.removeItem('agentmed_chat_history');
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">
                        <i class="fas fa-stethoscope"></i>
                    </div>
                    <h2>Chat Cleared</h2>
                    <p>Start a new conversation!</p>
                </div>
            `;
            this.showNotification('Chat cleared', 'success');
        }
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('collapsed');
        const icon = document.querySelector('#sidebarToggle i');
        if (icon) {
            icon.classList.toggle('fa-chevron-left');
            icon.classList.toggle('fa-chevron-right');
        }
    }

    showSettingsModal() {
        const modal = document.getElementById('settingsModal');
        if (modal) {
            modal.style.display = 'flex';
        }
    }

    hideSettingsModal() {
        const modal = document.getElementById('settingsModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('agentmed_settings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            }
            // Apply settings to checkboxes
            const soundCheckbox = document.getElementById('soundEnabled');
            const autoScrollCheckbox = document.getElementById('autoScroll');
            const timestampsCheckbox = document.getElementById('showTimestamps');
            
            if (soundCheckbox) soundCheckbox.checked = this.settings.soundEnabled;
            if (autoScrollCheckbox) autoScrollCheckbox.checked = this.settings.autoScroll;
            if (timestampsCheckbox) timestampsCheckbox.checked = this.settings.showTimestamps;
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    saveSettings() {
        try {
            localStorage.setItem('agentmed_settings', JSON.stringify(this.settings));
        } catch (error) {
            console.error('Error saving settings:', error);
        }
    }

    updateMessageTimestamps() {
        const messages = document.querySelectorAll('.message-time');
        messages.forEach(msg => {
            msg.style.display = this.settings.showTimestamps ? 'flex' : 'none';
        });
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => notification.classList.add('show'), 10);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    playEmergencySound() {
        // Create a simple beep sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.agentMedApp = new AgentMedApp();
});

// Add notification styles dynamically
const style = document.createElement('style');
style.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        z-index: 2000;
        opacity: 0;
        transform: translateX(400px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .notification.show {
        opacity: 1;
        transform: translateX(0);
    }
    
    .notification-success {
        border-left: 4px solid #10b981;
    }
    
    .notification-error {
        border-left: 4px solid #ef4444;
    }
    
    .notification-info {
        border-left: 4px solid #3b82f6;
    }
    
    .notification i {
        font-size: 1.25rem;
    }
    
    .notification-success i {
        color: #10b981;
    }
    
    .notification-error i {
        color: #ef4444;
    }
    
    .notification-info i {
        color: #3b82f6;
    }
`;
document.head.appendChild(style);
