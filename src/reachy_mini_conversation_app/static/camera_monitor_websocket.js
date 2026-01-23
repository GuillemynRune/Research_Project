// Enhanced Dashboard JavaScript - Timers, Medicine, System Logs
let ws = null;
let reconnectInterval = null;
let frameCount = 0;
let lastFpsUpdate = Date.now();
let actualFps = 0;
let messageCount = 0;
let startTime = Date.now();
let systemLogs = [];
let activeTasks = {};
let latestMedicine = null;

// Connect to WebSocket
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/camera`;
  
  addLog('INFO', 'Connecting to WebSocket...');
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'blob';
  
  ws.onopen = () => {
    addLog('INFO', 'WebSocket connected successfully');
    document.getElementById('camera-status').textContent = 'Live';
    document.getElementById('camera-status').className = 'status-badge live';
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
      // JSON message
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (e) {
        addLog('ERROR', `Failed to parse message: ${e.message}`);
      }
    }
  };
  
  ws.onerror = (error) => {
    addLog('ERROR', 'WebSocket connection error');
    document.getElementById('camera-status').textContent = 'Error';
  };
  
  ws.onclose = () => {
    addLog('WARNING', 'WebSocket disconnected');
    document.getElementById('camera-status').textContent = 'Disconnected';
    
    // Auto-reconnect after 2 seconds
    if (!reconnectInterval) {
      reconnectInterval = setInterval(() => {
        addLog('INFO', 'Attempting to reconnect...');
        connectWebSocket();
      }, 2000);
    }
  };
}

// Handle different WebSocket message types
function handleWebSocketMessage(data) {
  if (data.type === 'conversation') {
    updateConversationDisplay(data.messages);
  } else if (data.type === 'tasks') {
    updateTasksDisplay(data.tasks);
  } else if (data.type === 'medicine') {
    updateMedicineDisplay(data.medicine);
  } else if (data.type === 'log') {
    addLog(data.level, data.message);
  }
}

// Update tasks display with live countdown
function updateTasksDisplay(tasks) {
  const container = document.getElementById('tasks-container');
  
  if (!tasks || tasks.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">⏰</div>
        <p>No active timers or reminders</p>
      </div>
    `;
    document.getElementById('active-tasks').textContent = '0';
    activeTasks = {};
    return;
  }
  
  // Update active tasks count
  document.getElementById('active-tasks').textContent = tasks.length;
  
  // Update activeTasks object
  const newActiveTasks = {};
  tasks.forEach(task => {
    newActiveTasks[task.task_id] = task;
  });
  activeTasks = newActiveTasks;
  
  // Render tasks
  let html = '';
  tasks.forEach(task => {
    const timeRemaining = Math.max(0, task.time_remaining_seconds);
    const progress = calculateProgress(task);
    
    html += `
      <div class="task-card ${task.type}">
        <div class="task-header">
          <span class="task-type">${task.type === 'timer' ? '⏱️ Timer' : '🔔 Reminder'}</span>
        </div>
        <div class="task-countdown" id="countdown-${task.task_id}">
          ${formatTimeRemaining(timeRemaining)}
        </div>
        ${task.message ? `<div class="task-message">${task.message}</div>` : ''}
        <div class="task-progress">
          <div class="task-progress-bar" style="width: ${progress}%"></div>
        </div>
      </div>
    `;
  });
  
  container.innerHTML = html;
}

// Calculate progress percentage for task
function calculateProgress(task) {
  // Progress is based on how much time has elapsed
  const totalDuration = task.total_duration_seconds || 100;
  const timeRemaining = task.time_remaining_seconds || 0;
  const elapsed = totalDuration - timeRemaining;
  return Math.min(100, Math.max(0, (elapsed / totalDuration) * 100));
}

// Format time remaining as MM:SS or HH:MM:SS
function formatTimeRemaining(seconds) {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hrs > 0) {
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  } else {
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
}

// Update countdown timers every second
function updateCountdowns() {
  Object.keys(activeTasks).forEach(taskId => {
    const task = activeTasks[taskId];
    const element = document.getElementById(`countdown-${taskId}`);
    
    if (element && task.time_remaining_seconds > 0) {
      task.time_remaining_seconds -= 1;
      element.textContent = formatTimeRemaining(task.time_remaining_seconds);
      
      // If timer reached zero, request update from server
      if (task.time_remaining_seconds <= 0) {
        requestTasksUpdate();
      }
    }
  });
}

// Update medicine display
function updateMedicineDisplay(medicine) {
  const container = document.getElementById('medicine-container');
  
  if (!medicine) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">💊</div>
        <p>No medicine identified yet</p>
      </div>
    `;
    return;
  }
  
  latestMedicine = medicine;
  
  // Parse medicine info
  const fields = parseMedicineInfo(medicine.medicine_info || medicine);
  
  let html = `<div class="medicine-card">
    <div class="medicine-title">💊 Identified Medicine</div>
  `;
  
  // Display each field
  Object.keys(fields).forEach(key => {
    if (fields[key]) {
      html += `
        <div class="medicine-field">
          <div class="medicine-label">${key}:</div>
          <div class="medicine-value">${fields[key]}</div>
        </div>
      `;
    }
  });
  
  html += `</div>`;
  container.innerHTML = html;
}

// Parse medicine information from text
function parseMedicineInfo(text) {
  const fields = {
    'Name': '',
    'Dosage': '',
    'Timing': '',
    'Instructions': '',
    'Other': ''
  };
  
  if (typeof text !== 'string') {
    return fields;
  }
  
  // Try to extract structured information
  const lines = text.split('\n');
  let currentField = 'Other';
  
  lines.forEach(line => {
    line = line.trim();
    if (!line) return;
    
    // Check for field markers
    if (line.match(/name|medicine|medication/i)) {
      currentField = 'Name';
      const match = line.match(/:\s*(.+)/);
      if (match) fields['Name'] = match[1].trim();
    } else if (line.match(/dosage|dose|strength/i)) {
      currentField = 'Dosage';
      const match = line.match(/:\s*(.+)/);
      if (match) fields['Dosage'] = match[1].trim();
    } else if (line.match(/timing|when|frequency/i)) {
      currentField = 'Timing';
      const match = line.match(/:\s*(.+)/);
      if (match) fields['Timing'] = match[1].trim();
    } else if (line.match(/instructions|directions|how/i)) {
      currentField = 'Instructions';
      const match = line.match(/:\s*(.+)/);
      if (match) fields['Instructions'] = match[1].trim();
    } else if (line.includes(':')) {
      // Custom field
      const [key, value] = line.split(':');
      fields[key.trim()] = value.trim();
    } else {
      // Append to current field
      if (fields[currentField]) {
        fields[currentField] += ' ' + line;
      } else {
        fields[currentField] = line;
      }
    }
  });
  
  // Clean up empty fields
  Object.keys(fields).forEach(key => {
    if (!fields[key] || fields[key] === 'None') {
      delete fields[key];
    }
  });
  
  return fields;
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
      
      // Skip system messages
      if (role === 'system') return;
      
      // Only show user and assistant messages
      if (role !== 'user' && role !== 'assistant') return;
      
      // Clean up assistant messages
      if (role === 'assistant') {
        // Remove tool call markers
        content = content.replace(/\[TOOL:[^\]]+\]/g, '').trim();
        content = content.replace(/^["']|["']$/g, '').trim();
        
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
  
  // Keep only last 100 logs
  if (systemLogs.length > 100) {
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
    logsDiv.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📋</div><p>Waiting for logs...</p></div>';
  }
}

// Request updates from server
function requestConversationUpdate() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'get_conversation' }));
  }
}

function requestTasksUpdate() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'get_tasks' }));
  }
}

function requestMedicineUpdate() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'get_medicine' }));
  }
}

// Clear conversation history
async function clearHistory() {
  try {
    addLog('INFO', 'Clearing conversation history...');
    const response = await fetch('/clear_history', { method: 'POST' });
    if (response.ok) {
      addLog('INFO', 'Conversation history cleared');
      requestConversationUpdate();
    } else {
      addLog('ERROR', 'Failed to clear conversation history');
    }
  } catch (error) {
    addLog('ERROR', `Clear history failed: ${error.message}`);
  }
}

// Clear system logs
function clearLogs() {
  systemLogs = [];
  updateSystemLogs();
  addLog('INFO', 'System logs cleared');
}

// Manual refresh
function manualRefresh() {
  addLog('INFO', 'Manually refreshing...');
  requestConversationUpdate();
  requestTasksUpdate();
  requestMedicineUpdate();
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

// Request updates every 2 seconds
setInterval(requestConversationUpdate, 2000);
setInterval(requestTasksUpdate, 2000);
setInterval(requestMedicineUpdate, 2000);

// Update countdowns every second
setInterval(updateCountdowns, 1000);

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