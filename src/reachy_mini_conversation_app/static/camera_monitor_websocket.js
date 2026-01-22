// WebSocket Camera Monitor JavaScript - Gradio-style Dashboard
let ws = null;
let reconnectInterval = null;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let actualFps = 0;
let messageCount = 0;
let startTime = Date.now();
let systemLogs = [];

// Connect to WebSocket
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/camera`;
  
  addLog('info', 'Connecting to WebSocket...');
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'blob';
  
  ws.onopen = () => {
    addLog('info', 'WebSocket connected successfully');
    document.getElementById('camera-status').textContent = 'Live';
    document.getElementById('camera-status').className = 'status-badge live';
    document.getElementById('connection-status').textContent = 'Connected';
    if (reconnectInterval) {
      clearInterval(reconnectInterval);
      reconnectInterval = null;
    }
  };
  
  ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
      // Camera frame
      const url = URL.createObjectURL(event.data);
      const img = document.getElementById('camera-feed');
      if (img.src && img.src.startsWith('blob:')) {
        URL.revokeObjectURL(img.src);
      }
      img.src = url;
      
      // Calculate FPS
      frameCount++;
      const now = Date.now();
      if (now - lastFpsUpdate >= 1000) {
        actualFps = frameCount;
        document.getElementById('actual-fps').textContent = actualFps;
        document.getElementById('fps-display').textContent = `${actualFps} FPS`;
        frameCount = 0;
        lastFpsUpdate = now;
      }
    } else {
      // JSON message (conversation history)
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'conversation') {
          updateConversationDisplay(data.messages);
        }
      } catch (e) {
        addLog('error', `Failed to parse message: ${e.message}`);
      }
    }
  };
  
  ws.onerror = (error) => {
    addLog('error', 'WebSocket connection error');
    document.getElementById('camera-status').textContent = 'Error';
    document.getElementById('connection-status').textContent = 'Error';
  };
  
  ws.onclose = () => {
    addLog('warning', 'WebSocket disconnected');
    document.getElementById('camera-status').textContent = 'Disconnected';
    document.getElementById('connection-status').textContent = 'Disconnected';
    
    // Auto-reconnect after 2 seconds
    if (!reconnectInterval) {
      reconnectInterval = setInterval(() => {
        addLog('info', 'Attempting to reconnect...');
        connectWebSocket();
      }, 2000);
    }
  };
}

// Update conversation display
function updateConversationDisplay(messages) {
  const historyDiv = document.getElementById('conversation-history');
  
  if (messages && messages.length > 0) {
    let html = '';
    let validMessageCount = 0;
    
    messages.forEach(msg => {
      const role = msg.role || 'unknown';
      let content = msg.content || '';
      
      // Skip system messages entirely
      if (role === 'system') return;
      
      // Only show user and assistant messages
      if (role !== 'user' && role !== 'assistant') return;
      
      // Clean up assistant messages
      if (role === 'assistant') {
        // Remove tool call markers like [TOOL:play_emotion:{"emotion":"attentive1"}]
        content = content.replace(/\[TOOL:[^\]]+\]/g, '').trim();
        
        // Remove quotes around entire response
        content = content.replace(/^["']|["']$/g, '').trim();
        
        // Skip if empty after cleaning
        if (!content) return;
      }
      
      validMessageCount++;
      
      html += `<div class="message ${role}">
                <div class="message-role">${role}</div>
                <div class="message-content">${content}</div>
               </div>`;
    });
    
    messageCount = validMessageCount;
    document.getElementById('message-count').textContent = messageCount;
    
    if (html) {
      historyDiv.innerHTML = html;
      historyDiv.scrollTop = historyDiv.scrollHeight;
    } else {
      historyDiv.innerHTML = '<div class="empty-state"><div class="empty-state-icon">💬</div><p>No messages yet</p></div>';
    }
  } else {
    historyDiv.innerHTML = '<div class="empty-state"><div class="empty-state-icon">💬</div><p>No messages yet</p></div>';
    messageCount = 0;
    document.getElementById('message-count').textContent = '0';
  }
}

// Add system log entry
function addLog(level, message) {
  const timestamp = new Date().toLocaleTimeString();
  const logEntry = { level, message, timestamp };
  systemLogs.push(logEntry);
  
  // Keep only last 50 logs
  if (systemLogs.length > 50) {
    systemLogs.shift();
  }
  
  updateSystemLogs();
}

// Update system logs display
function updateSystemLogs() {
  const logsDiv = document.getElementById('system-logs');
  
  if (systemLogs.length > 0) {
    let html = '';
    // Show logs in reverse order (newest first)
    for (let i = systemLogs.length - 1; i >= 0; i--) {
      const log = systemLogs[i];
      html += `<div class="log-entry ${log.level}">
                <span class="log-timestamp">[${log.timestamp}]</span>
                ${log.message}
               </div>`;
    }
    logsDiv.innerHTML = html;
  } else {
    logsDiv.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📋</div><p>No logs yet</p></div>';
  }
}

// Request conversation update
function requestConversationUpdate() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'get_conversation' }));
  }
}

// Clear conversation history
async function clearHistory() {
  try {
    addLog('info', 'Clearing conversation history...');
    const response = await fetch('/clear_history', { method: 'POST' });
    if (response.ok) {
      addLog('info', 'Conversation history cleared');
      requestConversationUpdate();
    } else {
      addLog('error', 'Failed to clear conversation history');
    }
  } catch (error) {
    addLog('error', `Clear history failed: ${error.message}`);
  }
}

// Clear system logs
function clearLogs() {
  systemLogs = [];
  updateSystemLogs();
  addLog('info', 'System logs cleared');
}

// Manual refresh
function manualRefresh() {
  addLog('info', 'Manually refreshing conversation...');
  requestConversationUpdate();
}

// Update uptime display
function updateUptime() {
  const elapsed = Date.now() - startTime;
  const minutes = Math.floor(elapsed / 60000);
  const seconds = Math.floor((elapsed % 60000) / 1000);
  document.getElementById('uptime').textContent = 
    `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// Event listeners
document.getElementById('refresh-btn').addEventListener('click', manualRefresh);
document.getElementById('clear-history-btn').addEventListener('click', clearHistory);
document.getElementById('clear-logs-btn').addEventListener('click', clearLogs);

// Initial connection
connectWebSocket();

// Request conversation updates every 2 seconds
setInterval(requestConversationUpdate, 2000);

// Update uptime every second
setInterval(updateUptime, 1000);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (ws) {
    ws.close();
  }
  if (reconnectInterval) {
    clearInterval(reconnectInterval);
  }
});