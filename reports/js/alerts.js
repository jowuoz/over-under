// reports/js/alerts.js

// Alerts Page JavaScript
console.log('⚠️ Alerts JavaScript loaded');

// Global variables
let allAlerts = [];
let filteredAlerts = [];
let currentPage = 1;
const alertsPerPage = 20;
let sortColumn = 'probability';
let sortDirection = 'desc';

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Alerts page initialized');
    
    // Load alerts data
    loadAlertsData();
    
    // Initialize filters and search
    initializeFilters();
    
    // Initialize export functionality
    initializeExport();
    
    // Set up auto-refresh (every 30 seconds)
    setInterval(loadAlertsData, 30000);
    
    // Initialize pagination
    initializePagination();
    
    // Set current year in footer
    document.getElementById('currentYear').textContent = new Date().getFullYear();
    
    // Initialize date display
    updateDateDisplay();
});

// Load alerts data
function loadAlertsData() {
    console.log('Loading alerts data...');
    
    // Show loading state
    showLoadingState();
    
    // Try to load from multiple sources
    Promise.any([
        loadFromLatestJson(),
        loadFromAlertsJson(),
        loadFromPredictionsJson()
    ])
    .then(data => {
        console.log('Alerts data loaded successfully:', data.length, 'alerts');
        
        // Process and store alerts
        allAlerts = processAlertsData(data);
        
        // Update UI with loaded data
        updateAlertsDisplay();
        
        // Update summary statistics
        updateSummaryStats();
        
        // Update last updated time
        updateLastUpdated();
        
        // Hide loading state
        hideLoadingState();
    })
    .catch(error => {
        console.error('All data sources failed:', error);
        
        // Use sample data as fallback
        useSampleData();
        
        // Hide loading state
        hideLoadingState();
        
        showNotification('Using sample data for demonstration', 'info');
    });
}

// Load data from latest.json
function loadFromLatestJson() {
    return fetch('data/latest.json')
        .then(response => {
            if (!response.ok) throw new Error('latest.json not found');
            return response.json();
        })
        .then(data => data.alerts || []);
}

// Load data from alerts-specific file
function loadFromAlertsJson() {
    return fetch('data/alerts.json')
        .then(response => {
            if (!response.ok) throw new Error('alerts.json not found');
            return response.json();
        })
        .then(data => Array.isArray(data) ? data : []);
}

// Load data from predictions file
function loadFromPredictionsJson() {
    return fetch('data/predictions.json')
        .then(response => {
            if (!response.ok) throw new Error('predictions.json not found');
            return response.json();
        })
        .then(data => {
            // Convert predictions to alerts format
            const predictions = data.predictions || data || [];
            return predictions.filter(p => 
                (p.over_25_prob || 0) >= 0.7 || // High probability
                (p.confidence || 0) >= 0.8 ||   // High confidence
                p.status === 'live'              // Live games
            );
        });
}

// Process alerts data into consistent format
function processAlertsData(data) {
    if (!Array.isArray(data)) {
        return [];
    }
    
    return data.map((alert, index) => ({
        id: alert.id || `alert_${index}_${Date.now()}`,
        home_team: alert.home_team || 'Home Team',
        away_team: alert.away_team || 'Away Team',
        league: alert.league || 'Unknown League',
        home_score: alert.home_score || 0,
        away_score: alert.away_score || 0,
        minute: alert.minute || 'FT',
        over_25_prob: alert.over_25_prob || alert.probability || 0,
        confidence: alert.confidence || 0.7,
        status: alert.status || 'completed',
        timestamp: alert.timestamp || new Date().toISOString(),
        probability_percentage: ((alert.over_25_prob || alert.probability || 0) * 100).toFixed(1),
        confidence_percentage: ((alert.confidence || 0.7) * 100).toFixed(1),
        alert_type: determineAlertType(alert)
    }));
}

// Determine alert type based on probability and confidence
function determineAlertType(alert) {
    const probability = alert.over_25_prob || alert.probability || 0;
    const confidence = alert.confidence || 0;
    
    if (probability >= 0.8 && confidence >= 0.8) {
        return 'high';
    } else if (probability >= 0.7 || confidence >= 0.7) {
        return 'medium';
    } else {
        return 'low';
    }
}

// Update alerts display
function updateAlertsDisplay() {
    console.log('Updating alerts display');
    
    // Apply current filters
    applyFilters();
    
    // Sort alerts
    sortAlerts();
    
    // Update table
    updateAlertsTable();
    
    // Update pagination
    updatePaginationControls();
    
    // Update summary counts
    updateFilteredCounts();
}

// Initialize filters and search
function initializeFilters() {
    const searchInput = document.getElementById('alertSearch');
    const confidenceFilter = document.getElementById('alertFilter');
    const leagueFilter = document.getElementById('leagueFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', debounce(() => {
            applyFilters();
            updateAlertsDisplay();
        }, 300));
    }
    
    if (confidenceFilter) {
        confidenceFilter.addEventListener('change', () => {
            applyFilters();
            updateAlertsDisplay();
        });
    }
    
    if (leagueFilter) {
        leagueFilter.addEventListener('change', () => {
            applyFilters();
            updateAlertsDisplay();
        });
    }
    
    // Add league options dynamically
    populateLeagueFilter();
}

// Apply current filters
function applyFilters() {
    const searchTerm = document.getElementById('alertSearch')?.value.toLowerCase() || '';
    const confidenceFilter = document.getElementById('alertFilter')?.value || 'all';
    const leagueFilter = document.getElementById('leagueFilter')?.value || 'all';
    
    filteredAlerts = allAlerts.filter(alert => {
        // Search filter
        const searchMatch = searchTerm === '' || 
            alert.home_team.toLowerCase().includes(searchTerm) ||
            alert.away_team.toLowerCase().includes(searchTerm) ||
            alert.league.toLowerCase().includes(searchTerm);
        
        // Confidence filter
        let confidenceMatch = true;
        if (confidenceFilter !== 'all') {
            if (confidenceFilter === 'high') {
                confidenceMatch = alert.alert_type === 'high';
            } else if (confidenceFilter === 'medium') {
                confidenceMatch = alert.alert_type === 'medium';
            } else if (confidenceFilter === 'low') {
                confidenceMatch = alert.alert_type === 'low';
            }
        }
        
        // League filter
        let leagueMatch = true;
        if (leagueFilter !== 'all') {
            leagueMatch = alert.league.toLowerCase().includes(leagueFilter.toLowerCase());
        }
        
        return searchMatch && confidenceMatch && leagueMatch;
    });
}

// Sort alerts based on current sort column and direction
function sortAlerts() {
    filteredAlerts.sort((a, b) => {
        let aValue, bValue;
        
        switch(sortColumn) {
            case 'probability':
                aValue = a.over_25_prob;
                bValue = b.over_25_prob;
                break;
            case 'confidence':
                aValue = a.confidence;
                bValue = b.confidence;
                break;
            case 'minute':
                aValue = parseMinute(a.minute);
                bValue = parseMinute(b.minute);
                break;
            case 'time':
                aValue = new Date(a.timestamp).getTime();
                bValue = new Date(b.timestamp).getTime();
                break;
            default:
                aValue = a.over_25_prob;
                bValue = b.over_25_prob;
        }
        
        if (sortDirection === 'asc') {
            return aValue > bValue ? 1 : -1;
        } else {
            return aValue < bValue ? 1 : -1;
        }
    });
}

// Parse minute string to number
function parseMinute(minute) {
    if (minute === 'FT' || minute === 'HT') return 999;
    const num = parseInt(minute);
    return isNaN(num) ? 0 : num;
}

// Update alerts table
function updateAlertsTable() {
    const tbody = document.querySelector('#alertsTable tbody');
    if (!tbody) return;
    
    // Calculate pagination
    const startIndex = (currentPage - 1) * alertsPerPage;
    const endIndex = startIndex + alertsPerPage;
    const pageAlerts = filteredAlerts.slice(startIndex, endIndex);
    
    if (pageAlerts.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center py-8">
                    <div class="empty-table-state">
                        <i class="fas fa-inbox fa-2x"></i>
                        <p class="mt-4 text-gray-500">No alerts match your filters</p>
                        <button onclick="clearFilters()" class="btn-clear-filters mt-2">
                            Clear Filters
                        </button>
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    // Build table rows
    let html = '';
    pageAlerts.forEach(alert => {
        html += `
            <tr class="alert-row" data-alert-id="${alert.id}" data-alert-type="${alert.alert_type}">
                <td class="py-4">
                    <div class="flex items-center">
                        <div class="match-teams">
                            <div class="font-semibold">${alert.home_team}</div>
                            <div class="text-gray-500 text-sm">vs</div>
                            <div class="font-semibold">${alert.away_team}</div>
                        </div>
                    </div>
                </td>
                <td class="py-4">
                    <span class="league-badge">${alert.league}</span>
                </td>
                <td class="py-4">
                    <div class="score-display">
                        <span class="score-number">${alert.home_score}</span>
                        <span class="score-separator">-</span>
                        <span class="score-number">${alert.away_score}</span>
                    </div>
                </td>
                <td class="py-4">
                    <span class="minute-badge ${alert.minute === 'FT' ? 'finished' : 'live'}">
                        ${alert.minute}
                    </span>
                </td>
                <td class="py-4">
                    <div class="probability-display">
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${alert.probability_percentage}%"></div>
                        </div>
                        <span class="probability-text">${alert.probability_percentage}%</span>
                    </div>
                </td>
                <td class="py-4">
                    <span class="confidence-badge confidence-${alert.alert_type}">
                        ${alert.confidence_percentage}%
                    </span>
                </td>
                <td class="py-4">
                    <span class="status-badge status-${alert.status}">
                        ${getStatusText(alert.status)}
                    </span>
                </td>
                <td class="py-4">
                    <span class="timestamp">${formatTimestamp(alert.timestamp)}</span>
                </td>
                <td class="py-4">
                    <div class="action-buttons">
                        <button class="btn-action view" onclick="viewAlertDetails('${alert.id}')" 
                                title="View Details">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn-action track" onclick="trackAlert('${alert.id}')" 
                                title="Track Alert">
                            <i class="fas fa-bell"></i>
                        </button>
                        <button class="btn-action share" onclick="shareAlertRow('${alert.id}')" 
                                title="Share Alert">
                            <i class="fas fa-share-alt"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
    
    // Add click handlers for sorting
    addSortHandlers();
}

// Get status text
function getStatusText(status) {
    const statusMap = {
        'live': 'LIVE',
        'completed': 'COMPLETED',
        'scheduled': 'UPCOMING',
        'postponed': 'POSTPONED',
        'cancelled': 'CANCELLED'
    };
    return statusMap[status] || status.toUpperCase();
}

// Format timestamp for display
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Add sort handlers to table headers
function addSortHandlers() {
    const headers = document.querySelectorAll('#alertsTable thead th[data-sort]');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const column = header.getAttribute('data-sort');
            
            if (sortColumn === column) {
                // Toggle direction
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                // New column, default to descending
                sortColumn = column;
                sortDirection = 'desc';
            }
            
            // Update sort indicators
            updateSortIndicators();
            
            // Re-sort and update display
            updateAlertsDisplay();
        });
    });
}

// Update sort indicators
function updateSortIndicators() {
    const headers = document.querySelectorAll('#alertsTable thead th[data-sort]');
    headers.forEach(header => {
        const column = header.getAttribute('data-sort');
        header.classList.remove('sort-asc', 'sort-desc');
        
        if (column === sortColumn) {
            header.classList.add(`sort-${sortDirection}`);
        }
    });
}

// Initialize pagination
function initializePagination() {
    const paginationContainer = document.getElementById('alertsPagination');
    if (!paginationContainer) return;
    
    paginationContainer.addEventListener('click', (e) => {
        if (e.target.classList.contains('page-btn')) {
            const page = parseInt(e.target.getAttribute('data-page'));
            if (!isNaN(page)) {
                goToPage(page);
            }
        }
    });
}

// Update pagination controls
function updatePaginationControls() {
    const paginationContainer = document.getElementById('alertsPagination');
    if (!paginationContainer) return;
    
    const totalPages = Math.ceil(filteredAlerts.length / alertsPerPage);
    
    if (totalPages <= 1) {
        paginationContainer.innerHTML = '';
        return;
    }
    
    let html = '';
    
    // Previous button
    html += `
        <button class="page-btn ${currentPage === 1 ? 'disabled' : ''}" 
                ${currentPage === 1 ? 'disabled' : ''}
                data-page="${currentPage - 1}">
            <i class="fas fa-chevron-left"></i>
        </button>
    `;
    
    // Page numbers
    const maxPages = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxPages / 2));
    let endPage = Math.min(totalPages, startPage + maxPages - 1);
    
    if (endPage - startPage + 1 < maxPages) {
        startPage = Math.max(1, endPage - maxPages + 1);
    }
    
    for (let i = startPage; i <= endPage; i++) {
        html += `
            <button class="page-btn ${i === currentPage ? 'active' : ''}" 
                    data-page="${i}">
                ${i}
            </button>
        `;
    }
    
    // Next button
    html += `
        <button class="page-btn ${currentPage === totalPages ? 'disabled' : ''}" 
                ${currentPage === totalPages ? 'disabled' : ''}
                data-page="${currentPage + 1}">
            <i class="fas fa-chevron-right"></i>
        </button>
    `;
    
    // Page info
    html += `
        <div class="page-info">
            Showing ${Math.min(filteredAlerts.length, (currentPage - 1) * alertsPerPage + 1)}-${Math.min(filteredAlerts.length, currentPage * alertsPerPage)} 
            of ${filteredAlerts.length} alerts
        </div>
    `;
    
    paginationContainer.innerHTML = html;
}

// Go to specific page
function goToPage(page) {
    const totalPages = Math.ceil(filteredAlerts.length / alertsPerPage);
    
    if (page < 1 || page > totalPages || page === currentPage) {
        return;
    }
    
    currentPage = page;
    updateAlertsDisplay();
    
    // Scroll to top of table
    const table = document.querySelector('#alertsTable');
    if (table) {
        table.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// Update summary statistics
function updateSummaryStats() {
    const highCount = allAlerts.filter(a => a.alert_type === 'high').length;
    const mediumCount = allAlerts.filter(a => a.alert_type === 'medium').length;
    const lowCount = allAlerts.filter(a => a.alert_type === 'low').length;
    const totalCount = allAlerts.length;
    
    // Update summary cards
    document.getElementById('highAlertsCount').textContent = highCount;
    document.getElementById('mediumAlertsCount').textContent = mediumCount;
    document.getElementById('lowAlertsCount').textContent = lowCount;
    document.getElementById('totalAlertsCount').textContent = totalCount;
    
    // Update quick stats
    document.getElementById('totalAlerts').textContent = totalCount;
    document.getElementById('activeAlerts').textContent = allAlerts.filter(a => a.status === 'live').length;
    
    // Calculate average probability
    if (allAlerts.length > 0) {
        const avgProb = allAlerts.reduce((sum, a) => sum + parseFloat(a.probability_percentage), 0) / allAlerts.length;
        document.getElementById('avgProbability').textContent = avgProb.toFixed(1) + '%';
    } else {
        document.getElementById('avgProbability').textContent = '0%';
    }
}

// Update filtered counts
function updateFilteredCounts() {
    const filteredCount = document.getElementById('filteredCount');
    if (filteredCount) {
        filteredCount.textContent = `(${filteredAlerts.length} filtered)`;
    }
}

// Initialize export functionality
function initializeExport() {
    const exportBtn = document.querySelector('.btn-export');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportAlerts);
    }
}

// Export alerts to CSV
function exportAlerts() {
    console.log('Exporting alerts to CSV');
    
    if (filteredAlerts.length === 0) {
        showNotification('No alerts to export', 'warning');
        return;
    }
    
    // Prepare CSV data
    const headers = [
        'Home Team',
        'Away Team',
        'League',
        'Score',
        'Minute',
        'Over 2.5 Probability',
        'Confidence',
        'Status',
        'Time',
        'Alert Type'
    ];
    
    const rows = filteredAlerts.map(alert => [
        alert.home_team,
        alert.away_team,
        alert.league,
        `${alert.home_score}-${alert.away_score}`,
        alert.minute,
        `${alert.probability_percentage}%`,
        `${alert.confidence_percentage}%`,
        getStatusText(alert.status),
        formatTimestamp(alert.timestamp),
        alert.alert_type.toUpperCase()
    ]);
    
    const csvContent = [
        headers.join(','),
        ...rows.map(row => row.map(field => `"${field}"`).join(','))
    ].join('\n');
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().split('T')[0];
    
    link.href = url;
    link.download = `alerts_export_${timestamp}.csv`;
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showNotification(`Exported ${filteredAlerts.length} alerts to CSV`, 'success');
}

// View alert details
function viewAlertDetails(alertId) {
    console.log('Viewing alert details:', alertId);
    
    const alert = allAlerts.find(a => a.id === alertId);
    if (!alert) {
        showNotification('Alert not found', 'warning');
        return;
    }
    
    // Create modal with alert details
    showAlertDetailsModal(alert);
}

// Show alert details modal
function showAlertDetailsModal(alert) {
    const modalHtml = `
        <div class="modal-overlay" id="alertDetailsModal">
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-info-circle"></i> Alert Details</h3>
                    <button class="modal-close" onclick="closeAlertModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="alert-details-header">
                        <h4>${alert.home_team} vs ${alert.away_team}</h4>
                        <div class="alert-meta">
                            <span class="league-badge large">${alert.league}</span>
                            <span class="status-badge large status-${alert.status}">
                                ${getStatusText(alert.status)}
                            </span>
                        </div>
                    </div>
                    
                    <div class="details-grid">
                        <div class="detail-card">
                            <div class="detail-icon score">
                                <i class="fas fa-futbol"></i>
                            </div>
                            <div class="detail-content">
                                <h5>Score</h5>
                                <p class="detail-value">${alert.home_score} - ${alert.away_score}</p>
                                <p class="detail-label">Minute: ${alert.minute}</p>
                            </div>
                        </div>
                        
                        <div class="detail-card">
                            <div class="detail-icon probability">
                                <i class="fas fa-chart-line"></i>
                            </div>
                            <div class="detail-content">
                                <h5>Over 2.5 Probability</h5>
                                <p class="detail-value">${alert.probability_percentage}%</p>
                                <div class="probability-meter">
                                    <div class="meter-bar">
                                        <div class="meter-fill" style="width: ${alert.probability_percentage}%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="detail-card">
                            <div class="detail-icon confidence">
                                <i class="fas fa-bullseye"></i>
                            </div>
                            <div class="detail-content">
                                <h5>Confidence</h5>
                                <p class="detail-value">${alert.confidence_percentage}%</p>
                                <span class="confidence-badge large confidence-${alert.alert_type}">
                                    ${alert.alert_type.toUpperCase()} CONFIDENCE
                                </span>
                            </div>
                        </div>
                        
                        <div class="detail-card">
                            <div class="detail-icon time">
                                <i class="fas fa-clock"></i>
                            </div>
                            <div class="detail-content">
                                <h5>Time</h5>
                                <p class="detail-value">${formatTimestamp(alert.timestamp)}</p>
                                <p class="detail-label">${new Date(alert.timestamp).toLocaleString()}</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="prediction-analysis">
                        <h5><i class="fas fa-chart-bar"></i> Prediction Analysis</h5>
                        <div class="analysis-content">
                            <p>Based on current statistics and historical data, there's a 
                            <strong>${alert.probability_percentage}% chance</strong> that this match will have over 2.5 goals.</p>
                            
                            <div class="analysis-factors">
                                <div class="factor">
                                    <span class="factor-label">Team Form:</span>
                                    <span class="factor-value">Above Average</span>
                                </div>
                                <div class="factor">
                                    <span class="factor-label">Attack Strength:</span>
                                    <span class="factor-value">High</span>
                                </div>
                                <div class="factor">
                                    <span class="factor-label">Defense Weakness:</span>
                                    <span class="factor-value">Moderate</span>
                                </div>
                                <div class="factor">
                                    <span class="factor-label">Historical Data:</span>
                                    <span class="factor-value">Supportive</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn-secondary" onclick="closeAlertModal()">
                        Close
                    </button>
                    <button class="btn-primary" onclick="shareAlert('${alert.home_team}', '${alert.away_team}', '${alert.probability_percentage}')">
                        <i class="fas fa-share-alt"></i> Share Alert
                    </button>
                    <button class="btn-warning" onclick="trackAlert('${alert.id}')">
                        <i class="fas fa-bell"></i> Track This Alert
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to page
    const modalContainer = document.createElement('div');
    modalContainer.innerHTML = modalHtml;
    document.body.appendChild(modalContainer.firstElementChild);
    
    // Add modal styles if not already present
    addModalStyles();
    
    // Prevent scrolling on body
    document.body.style.overflow = 'hidden';
}

// Close alert modal
function closeAlertModal() {
    const modal = document.getElementById('alertDetailsModal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = '';
    }
}

// Share alert from table row
function shareAlertRow(alertId) {
    const alert = allAlerts.find(a => a.id === alertId);
    if (!alert) {
        showNotification('Alert not found', 'warning');
        return;
    }
    
    shareAlert(alert.home_team, alert.away_team, alert.probability_percentage);
}

// Share alert function
function shareAlert(homeTeam, awayTeam, probability) {
    console.log('Sharing alert:', homeTeam, awayTeam, probability);
    
    const text = `⚽ Over/Under Alert: ${homeTeam} vs ${awayTeam}\n📊 Over 2.5 Probability: ${probability}%\n\nCheck out the Over/Under Predictor for more insights!`;
    const url = window.location.href;
    
    if (navigator.share) {
        navigator.share({
            title: 'Over/Under Prediction Alert',
            text: text,
            url: url
        })
        .then(() => {
            console.log('Alert shared successfully');
            showNotification('Alert shared!', 'success');
        })
        .catch(error => {
            console.error('Error sharing:', error);
            copyToClipboard(`${text}\n\n${url}`);
        });
    } else {
        copyToClipboard(`${text}\n\n${url}`);
    }
}

// Track alert
function trackAlert(alertId) {
    console.log('Tracking alert:', alertId);
    
    // In a real implementation, this would send tracking data to backend
    // For now, just show a notification and update UI
    
    const alert = allAlerts.find(a => a.id === alertId);
    if (alert) {
        // Update tracking status in localStorage
        const trackedAlerts = JSON.parse(localStorage.getItem('trackedAlerts') || '[]');
        if (!trackedAlerts.includes(alertId)) {
            trackedAlerts.push(alertId);
            localStorage.setItem('trackedAlerts', JSON.stringify(trackedAlerts));
        }
        
        // Show notification
        showNotification(`Now tracking ${alert.home_team} vs ${alert.away_team}`, 'success');
        
        // Update button in table if visible
        const row = document.querySelector(`tr[data-alert-id="${alertId}"]`);
        if (row) {
            const trackBtn = row.querySelector('.btn-action.track');
            if (trackBtn) {
                trackBtn.innerHTML = '<i class="fas fa-check"></i>';
                trackBtn.classList.add('tracked');
                trackBtn.title = 'Tracking';
                
                // Revert after 2 seconds
                setTimeout(() => {
                    trackBtn.innerHTML = '<i class="fas fa-bell"></i>';
                    trackBtn.classList.remove('tracked');
                    trackBtn.title = 'Track Alert';
                }, 2000);
            }
        }
    }
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text)
        .then(() => {
            console.log('Text copied to clipboard');
            showNotification('Alert copied to clipboard!', 'success');
        })
        .catch(error => {
            console.error('Clipboard error:', error);
            showNotification('Could not copy to clipboard', 'warning');
            
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                showNotification('Alert copied to clipboard!', 'success');
            } catch (err) {
                showNotification('Failed to copy to clipboard', 'error');
            }
            document.body.removeChild(textarea);
        });
}

// Populate league filter with unique leagues
function populateLeagueFilter() {
    const leagueFilter = document.getElementById('leagueFilter');
    if (!leagueFilter) return;
    
    // Get unique leagues
    const leagues = [...new Set(allAlerts.map(alert => alert.league))].sort();
    
    // Clear existing options except "All Leagues"
    while (leagueFilter.options.length > 1) {
        leagueFilter.remove(1);
    }
    
    // Add league options
    leagues.forEach(league => {
        const option = document.createElement('option');
        option.value = league.toLowerCase().replace(/\s+/g, '_');
        option.textContent = league;
        leagueFilter.appendChild(option);
    });
}

// Clear all filters
function clearFilters() {
    document.getElementById('alertSearch').value = '';
    document.getElementById('alertFilter').value = 'all';
    document.getElementById('leagueFilter').value = 'all';
    
    applyFilters();
    updateAlertsDisplay();
    
    showNotification('Filters cleared', 'info');
}

// Show loading state
function showLoadingState() {
    const tbody = document.querySelector('#alertsTable tbody');
    if (tbody) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center py-12">
                    <div class="loading-state">
                        <i class="fas fa-spinner fa-spin fa-2x"></i>
                        <p class="mt-4 text-gray-500">Loading alerts...</p>
                    </div>
                </td>
            </tr>
        `;
    }
    
    // Disable filters and buttons
    document.getElementById('alertSearch').disabled = true;
    document.getElementById('alertFilter').disabled = true;
    document.getElementById('leagueFilter').disabled = true;
    document.querySelector('.btn-export').disabled = true;
}

// Hide loading state
function hideLoadingState() {
    // Enable filters and buttons
    document.getElementById('alertSearch').disabled = false;
    document.getElementById('alertFilter').disabled = false;
    document.getElementById('leagueFilter').disabled = false;
    document.querySelector('.btn-export').disabled = false;
}

// Update last updated time
function updateLastUpdated() {
    const lastUpdatedElement = document.getElementById('lastUpdated');
    if (lastUpdatedElement) {
        const now = new Date();
        lastUpdatedElement.textContent = now.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// Update date display
function updateDateDisplay() {
    const today = new Date();
    const dateElement = document.getElementById('todayDate');
    if (dateElement) {
        dateElement.textContent = today.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }
}

// Use sample data (fallback)
function useSampleData() {
    console.log('Using sample alerts data');
    
    const sampleAlerts = [
        {
            id: '1',
            home_team: 'Manchester City',
            away_team: 'Liverpool',
            league: 'Premier League',
            home_score: 2,
            away_score: 1,
            minute: 75,
            over_25_prob: 0.85,
            confidence: 0.92,
            status: 'live',
            timestamp: new Date(Date.now() - 15 * 60000).toISOString()
        },
        {
            id: '2',
            home_team: 'Real Madrid',
            away_team: 'Barcelona',
            league: 'La Liga',
            home_score: 1,
            away_score: 1,
            minute: 60,
            over_25_prob: 0.72,
            confidence: 0.78,
            status: 'live',
            timestamp: new Date(Date.now() - 30 * 60000).toISOString()
        },
        {
            id: '3',
            home_team: 'Bayern Munich',
            away_team: 'Borussia Dortmund',
            league: 'Bundesliga',
            home_score: 3,
            away_score: 0,
            minute: 'FT',
            over_25_prob: 0.95,
            confidence: 0.98,
            status: 'completed',
            timestamp: new Date(Date.now() - 120 * 60000).toISOString()
        },
        {
            id: '4',
            home_team: 'AC Milan',
            away_team: 'Inter Milan',
            league: 'Serie A',
            home_score: 0,
            away_score: 0,
            minute: 45,
            over_25_prob: 0.45,
            confidence: 0.65,
            status: 'live',
            timestamp: new Date(Date.now() - 45 * 60000).toISOString()
        }
    ];
    
    allAlerts = processAlertsData(sampleAlerts);
    updateAlertsDisplay();
    updateSummaryStats();
    populateLeagueFilter();
}

// Show notification
function showNotification(message, type = 'info') {
    console.log(`Notification [${type}]:`, message);
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add styles for notification if not already present
    addNotificationStyles();
    
    // Get or create notification container
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Get notification icon
function getNotificationIcon(type) {
    switch(type) {
        case 'success': return 'check-circle';
        case 'warning': return 'exclamation-triangle';
        case 'error': return 'exclamation-circle';
        default: return 'info-circle';
    }
}

// Add notification styles
function addNotificationStyles() {
    if (document.getElementById('notification-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'notification-styles';
    style.textContent = `
        .notification-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        .notification {
            background: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
            animation: slideIn 0.3s ease-out;
            border-left: 4px solid #3b82f6;
        }
        
        .notification-success {
            border-left-color: #10b981;
        }
        
        .notification-info {
            border-left-color: #3b82f6;
        }
        
        .notification-warning {
            border-left-color: #f59e0b;
        }
        
        .notification-error {
            border-left-color: #ef4444;
        }
        
        .notification button {
            background: none;
            border: none;
            cursor: pointer;
            color: #64748b;
            margin-left: auto;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);
}

// Add modal styles
function addModalStyles() {
    if (document.getElementById('modal-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'modal-styles';
    style.textContent = `
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            animation: fadeIn 0.3s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .modal-content {
            background: white;
            border-radius: 1rem;
            width: 90%;
            max-width: 600px;
            max-height: 90vh;
            overflow-y: auto;
            animation: slideUp 0.3s ease-out;
        }
        
        @keyframes slideUp {
            from { 
                transform: translateY(50px);
                opacity: 0;
            }
            to { 
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem;
            border-bottom: 2px solid #f1f5f9;
        }
        
        .modal-header h3 {
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .modal-close {
            background: none;
            border: none;
            font-size: 1.25rem;
            color: #64748b;
            cursor: pointer;
            padding: 0.5rem;
        }
        
        .modal-close:hover {
            color: #1e293b;
        }
        
        .modal-body {
            padding: 1.5rem;
        }
        
        .alert-details-header {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .alert-details-header h4 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .alert-meta {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .detail-card {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .detail-icon {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: white;
        }
        
        .detail-icon.score { background: #3b82f6; }
        .detail-icon.probability { background: #10b981; }
        .detail-icon.confidence { background: #f59e0b; }
        .detail-icon.time { background: #8b5cf6; }
        
        .detail-content h5 {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        
        .detail-value {
            font-size: 1.25rem;
            font-weight: bold;
            color: #1e293b;
        }
        
        .detail-label {
            font-size: 0.85rem;
            color: #94a3b8;
        }
        
        .probability-meter {
            margin-top: 0.5rem;
        }
        
        .meter-bar {
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            overflow: hidden;
        }
        
        .meter-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        }
        
        .prediction-analysis {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 2px solid #f1f5f9;
        }
        
        .prediction-analysis h5 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .analysis-content {
            background: #f0f9ff;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        .analysis-factors {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
            margin-top: 1rem;
        }
        
        .factor {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
        }
        
        .factor-label {
            color: #475569;
        }
        
        .factor-value {
            font-weight: 600;
            color: #1e293b;
        }
        
        .modal-footer {
            display: flex;
            gap: 1rem;
            padding: 1.5rem;
            border-top: 2px solid #f1f5f9;
            flex-wrap: wrap;
        }
        
        .btn-primary, .btn-secondary, .btn-warning {
            flex: 1;
            min-width: 120px;
            padding: 0.75rem;
            border-radius: 0.5rem;
            border: none;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background: #3b82f6;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2563eb;
        }
        
        .btn-secondary {
            background: #f1f5f9;
            color: #475569;
        }
        
        .btn-secondary:hover {
            background: #e2e8f0;
        }
        
        .btn-warning {
            background: #f59e0b;
            color: white;
        }
        
        .btn-warning:hover {
            background: #d97706;
        }
    `;
    document.head.appendChild(style);
}

// Debounce function for search
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Make functions available globally
window.exportAlerts = exportAlerts;
window.viewAlertDetails = viewAlertDetails;
window.trackAlert = trackAlert;
window.shareAlert = shareAlert;
window.shareAlertRow = shareAlertRow;
window.clearFilters = clearFilters;
window.closeAlertModal = closeAlertModal;
window.showNotification = showNotification;
