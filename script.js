/**
 * script.js - Main JavaScript for Over/Under Predictor Dashboard
 * Version: 2.0.0
 * Last Updated: 2024-01-06
 */

// ============================================
// GLOBAL CONFIGURATION
// ============================================

const CONFIG = {
    refreshInterval: 300, // 5 minutes in seconds
    apiBaseUrl: '', // Leave empty for same origin
    chartColors: {
        primary: '#2563eb',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#3b82f6',
        over25: '#10b981',
        under25: '#ef4444'
    },
    leagueLogos: {
        'Premier League': 'https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg',
        'La Liga': 'https://upload.wikimedia.org/wikipedia/en/0/0e/LaLiga_logo_2023.svg',
        'Bundesliga': 'https://upload.wikimedia.org/wikipedia/en/d/df/Bundesliga_logo_%282017%29.svg',
        'Serie A': 'https://upload.wikimedia.org/wikipedia/en/e/e9/Serie_A_logo_%282023%29.svg',
        'Ligue 1': 'https://upload.wikimedia.org/wikipedia/en/2/29/Ligue_1_Uber_Eats_logo.svg',
        'Champions League': 'https://upload.wikimedia.org/wikipedia/en/b/bf/UEFA_Champions_League_logo_2.svg',
        'Europa League': 'https://upload.wikimedia.org/wikipedia/en/0/05/UEFA_Europa_League_logo_2.svg',
        'FA Cup': 'https://upload.wikimedia.org/wikipedia/en/f/f2/The_Football_Association_Logo.svg'
    }
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toLocaleString('en-US');
}

/**
 * Format percentage
 */
function formatPercentage(num, decimals = 1) {
    return `${num.toFixed(decimals)}%`;
}

/**
 * Format time
 */
function formatTime(date) {
    return new Date(date).toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

/**
 * Debounce function to limit rapid calls
 */
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

/**
 * Get URL parameter
 */
function getUrlParam(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

/**
 * Set URL parameter
 */
function setUrlParam(name, value) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.pushState({}, '', url);
}

/**
 * Remove URL parameter
 */
function removeUrlParam(name) {
    const url = new URL(window.location);
    url.searchParams.delete(name);
    window.history.pushState({}, '', url);
}

/**
 * Show notification/toast
 */
function showNotification(message, type = 'info', duration = 5000) {
    const container = document.getElementById('notifications') || createNotificationContainer();
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    container.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, duration);
    
    return notification;
}

/**
 * Get notification icon based on type
 */
function getNotificationIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Create notification container if it doesn't exist
 */
function createNotificationContainer() {
    const container = document.createElement('div');
    container.id = 'notifications';
    container.className = 'notifications-container';
    document.body.appendChild(container);
    return container;
}

// ============================================
// AUTO-REFRESH SYSTEM
// ============================================

let refreshTimer = null;
let countdownTimer = null;

/**
 * Start auto-refresh countdown
 */
function startCountdown() {
    const countdownElement = document.getElementById('countdown');
    if (!countdownElement) return;
    
    let seconds = CONFIG.refreshInterval;
    
    // Clear any existing timer
    if (countdownTimer) clearInterval(countdownTimer);
    
    countdownTimer = setInterval(() => {
        seconds--;
        countdownElement.textContent = seconds;
        
        // Update progress bar if exists
        const progressBar = document.getElementById('countdown-progress');
        if (progressBar) {
            const percentage = (seconds / CONFIG.refreshInterval) * 100;
            progressBar.style.width = `${percentage}%`;
        }
        
        if (seconds <= 0) {
            clearInterval(countdownTimer);
            refreshPage();
        }
    }, 1000);
}

/**
 * Refresh the page
 */
function refreshPage() {
    showNotification('Refreshing data...', 'info', 2000);
    
    // Add loading state
    document.body.classList.add('loading');
    
    // Reload page after a short delay
    setTimeout(() => {
        window.location.reload();
    }, 1000);
}

/**
 * Pause auto-refresh
 */
function pauseRefresh() {
    if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
        showNotification('Auto-refresh paused', 'warning');
        return true;
    }
    return false;
}

/**
 * Resume auto-refresh
 */
function resumeRefresh() {
    if (!countdownTimer) {
        startCountdown();
        showNotification('Auto-refresh resumed', 'success');
        return true;
    }
    return false;
}

/**
 * Toggle auto-refresh
 */
function toggleAutoRefresh() {
    const btn = document.getElementById('toggle-refresh');
    if (!btn) return;
    
    if (countdownTimer) {
        pauseRefresh();
        btn.innerHTML = '<i class="fas fa-play"></i> Resume Refresh';
        btn.classList.remove('btn-success');
        btn.classList.add('btn-warning');
    } else {
        resumeRefresh();
        btn.innerHTML = '<i class="fas fa-pause"></i> Pause Refresh';
        btn.classList.remove('btn-warning');
        btn.classList.add('btn-success');
    }
}

// ============================================
// CHART FUNCTIONS
// ============================================

let charts = {};

/**
 * Load and display confidence chart
 */
function loadConfidenceChart() {
    const ctx = document.getElementById('confidenceChart');
    if (!ctx) return;
    
    fetch('data/confidence_chart.json')
        .then(response => {
            if (!response.ok) throw new Error('Chart data not found');
            return response.json();
        })
        .then(data => {
            if (charts.confidence) charts.confidence.destroy();
            
            charts.confidence = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.labels,
                    datasets: [{
                        data: data.data,
                        backgroundColor: data.colors,
                        borderWidth: 2,
                        borderColor: '#ffffff',
                        hoverOffset: 15
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '65%',
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                usePointStyle: true,
                                font: {
                                    size: 12
                                }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.raw || 0;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = Math.round((value / total) * 100);
                                    return `${label}: ${value} predictions (${percentage}%)`;
                                }
                            }
                        }
                    },
                    animation: {
                        animateScale: true,
                        animateRotate: true
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error loading confidence chart:', error);
            showNotification('Could not load confidence chart data', 'error');
        });
}

/**
 * Load and display league distribution chart
 */
function loadLeagueChart() {
    const ctx = document.getElementById('leagueChart');
    if (!ctx) return;
    
    fetch('data/league_chart.json')
        .then(response => {
            if (!response.ok) throw new Error('League chart data not found');
            return response.json();
        })
        .then(data => {
            if (charts.league) charts.league.destroy();
            
            charts.league = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Number of Games',
                        data: data.data,
                        backgroundColor: data.colors,
                        borderWidth: 1,
                        borderColor: 'rgba(0, 0, 0, 0.1)',
                        borderRadius: 6,
                        borderSkipped: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1,
                                callback: function(value) {
                                    if (value % 1 === 0) return value;
                                }
                            },
                            grid: {
                                drawBorder: false
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 0
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: ${context.raw}`;
                                }
                            }
                        }
                    },
                    animation: {
                        duration: 2000,
                        easing: 'easeOutQuart'
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error loading league chart:', error);
            showNotification('Could not load league distribution data', 'error');
        });
}

/**
 * Load and display historical trends chart
 */
function loadHistoricalChart() {
    const ctx = document.getElementById('historicalChart');
    if (!ctx) return;
    
    fetch('data/historical_summary.json')
        .then(response => {
            if (!response.ok) {
                console.log('Historical data not available yet');
                return null;
            }
            return response.json();
        })
        .then(data => {
            if (!data || data.length === 0) {
                // Show placeholder message
                const container = ctx.parentElement;
                container.innerHTML = `
                    <div class="chart-placeholder">
                        <i class="fas fa-chart-line"></i>
                        <p>Historical data will appear after 24 hours of operation</p>
                    </div>
                `;
                return;
            }
            
            if (charts.historical) charts.historical.destroy();
            
            // Process historical data
            const labels = data.map(item => {
                const date = new Date(item.timestamp);
                return date.toLocaleDateString([], { 
                    month: 'short', 
                    day: 'numeric',
                    hour: '2-digit'
                });
            });
            
            const confidenceData = data.map(item => item.avg_confidence);
            const gamesData = data.map(item => item.total_games);
            
            charts.historical = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Average Confidence',
                            data: confidenceData,
                            borderColor: CONFIG.chartColors.primary,
                            backgroundColor: CONFIG.chartColors.primary + '20',
                            borderWidth: 3,
                            tension: 0.4,
                            yAxisID: 'y',
                            fill: true
                        },
                        {
                            label: 'Total Games',
                            data: gamesData,
                            borderColor: CONFIG.chartColors.success,
                            backgroundColor: CONFIG.chartColors.success + '20',
                            borderWidth: 3,
                            tension: 0.4,
                            yAxisID: 'y1',
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Confidence %'
                            },
                            min: 0,
                            max: 100
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Games'
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error loading historical chart:', error);
        });
}

/**
 * Destroy all charts (for page cleanup)
 */
function destroyAllCharts() {
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    charts = {};
}

// ============================================
// ALERTS SYSTEM
// ============================================

/**
 * Track alert click/engagement
 */
function trackAlert(alertId) {
    console.log('Tracking alert engagement:', alertId);
    
    // Send analytics data
    const analyticsData = {
        alert_id: alertId,
        action: 'view',
        timestamp: new Date().toISOString(),
        page: window.location.pathname
    };
    
    // In a real system, you would send this to your analytics backend
    localStorage.setItem(`alert_track_${alertId}`, JSON.stringify(analyticsData));
    
    showNotification('Alert tracked for analysis', 'info', 3000);
    
    // Highlight the tracked alert
    const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
    if (alertElement) {
        alertElement.classList.add('alert-tracked');
        setTimeout(() => {
            alertElement.classList.remove('alert-tracked');
        }, 2000);
    }
}

/**
 * Share alert via Web Share API or clipboard
 */
function shareAlert(homeTeam, awayTeam, probability, alertId = null) {
    const shareData = {
        title: '⚽ Over/Under Alert',
        text: `${homeTeam} vs ${awayTeam} - Over 2.5 probability: ${probability}%`,
        url: window.location.href
    };
    
    if (navigator.share) {
        navigator.share(shareData)
            .then(() => {
                showNotification('Alert shared successfully!', 'success');
                if (alertId) logShare(alertId);
            })
            .catch(error => {
                console.error('Error sharing:', error);
                fallbackShare(shareData.text);
            });
    } else {
        fallbackShare(shareData.text);
        if (alertId) logShare(alertId);
    }
}

/**
 * Fallback share method using clipboard
 */
function fallbackShare(text) {
    navigator.clipboard.writeText(text)
        .then(() => {
            showNotification('Alert copied to clipboard!', 'success');
        })
        .catch(error => {
            console.error('Error copying to clipboard:', error);
            showNotification('Could not share alert', 'error');
        });
}

/**
 * Log share action for analytics
 */
function logShare(alertId) {
    const shareLog = {
        alert_id: alertId,
        timestamp: new Date().toISOString(),
        platform: 'share'
    };
    console.log('Share logged:', shareLog);
    // Send to analytics backend in production
}

/**
 * View game details
 */
function viewGameDetails(gameId) {
    console.log('Viewing game details:', gameId);
    
    // Create modal with game details
    const modal = document.createElement('div');
    modal.className = 'modal game-details-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>Game Details</h3>
                <button class="modal-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div class="loading-spinner">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Loading game details...</p>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // In a real system, you would fetch game details from an API
    setTimeout(() => {
        const modalBody = modal.querySelector('.modal-body');
        modalBody.innerHTML = `
            <div class="game-details">
                <p><i class="fas fa-info-circle"></i> Detailed game analysis would appear here.</p>
                <p><i class="fas fa-database"></i> Game ID: ${gameId}</p>
                <p><i class="fas fa-clock"></i> This feature is coming soon!</p>
            </div>
        `;
    }, 1000);
}

// ============================================
// TABLE & DATA MANAGEMENT
// ============================================

let currentPage = 1;
const itemsPerPage = 20;

/**
 * Initialize pagination for tables
 */
function initPagination(tableId = 'alertsTable') {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const rows = table.querySelectorAll('tbody tr');
    const pageCount = Math.ceil(rows.length / itemsPerPage);
    
    if (pageCount <= 1) return;
    
    const paginationContainer = document.getElementById(`${tableId}Pagination`) || 
                               createPaginationContainer(tableId);
    
    paginationContainer.innerHTML = '';
    
    // Previous button
    const prevButton = document.createElement('button');
    prevButton.innerHTML = '<i class="fas fa-chevron-left"></i>';
    prevButton.className = 'page-btn page-prev';
    prevButton.disabled = true;
    prevButton.onclick = () => changePage(currentPage - 1, tableId);
    paginationContainer.appendChild(prevButton);
    
    // Page buttons
    for (let i = 1; i <= pageCount; i++) {
        const button = document.createElement('button');
        button.textContent = i;
        button.className = 'page-btn';
        if (i === 1) button.classList.add('active');
        
        button.onclick = () => changePage(i, tableId);
        paginationContainer.appendChild(button);
    }
    
    // Next button
    const nextButton = document.createElement('button');
    nextButton.innerHTML = '<i class="fas fa-chevron-right"></i>';
    nextButton.className = 'page-btn page-next';
    nextButton.disabled = pageCount === 1;
    nextButton.onclick = () => changePage(currentPage + 1, tableId);
    paginationContainer.appendChild(nextButton);
    
    // Show first page
    showPage(1, tableId);
}

/**
 * Create pagination container if it doesn't exist
 */
function createPaginationContainer(tableId) {
    const container = document.createElement('div');
    container.id = `${tableId}Pagination`;
    container.className = 'pagination';
    
    const table = document.getElementById(tableId);
    if (table) {
        table.parentNode.insertBefore(container, table.nextSibling);
    }
    
    return container;
}

/**
 * Show specific page of table
 */
function showPage(pageNum, tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const rows = table.querySelectorAll('tbody tr');
    const start = (pageNum - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    
    rows.forEach((row, index) => {
        row.style.display = (index >= start && index < end) ? '' : 'none';
    });
}

/**
 * Change to specific page
 */
function changePage(pageNum, tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const rows = table.querySelectorAll('tbody tr');
    const pageCount = Math.ceil(rows.length / itemsPerPage);
    
    if (pageNum < 1 || pageNum > pageCount) return;
    
    currentPage = pageNum;
    showPage(pageNum, tableId);
    
    // Update active button
    const pagination = document.getElementById(`${tableId}Pagination`);
    if (pagination) {
        const buttons = pagination.querySelectorAll('.page-btn');
        buttons.forEach((btn, index) => {
            btn.classList.remove('active');
            if (index === pageNum) { // +1 for prev button
                btn.classList.add('active');
            }
        });
        
        // Update prev/next button states
        const prevBtn = pagination.querySelector('.page-prev');
        const nextBtn = pagination.querySelector('.page-next');
        if (prevBtn) prevBtn.disabled = pageNum === 1;
        if (nextBtn) nextBtn.disabled = pageNum === pageCount;
    }
}

/**
 * Filter table rows based on search input
 */
function filterTable(tableId, searchId, filterId) {
    const table = document.getElementById(tableId);
    const searchInput = document.getElementById(searchId);
    const filterSelect = document.getElementById(filterId);
    
    if (!table || !searchInput) return;
    
    const searchTerm = searchInput.value.toLowerCase();
    const filterValue = filterSelect ? filterSelect.value : 'all';
    const rows = table.querySelectorAll('tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        const shouldShow = text.includes(searchTerm) && passesFilter(row, filterValue);
        row.style.display = shouldShow ? '' : 'none';
    });
    
    // Reinitialize pagination
    initPagination(tableId);
}

/**
 * Check if row passes filter criteria
 */
function passesFilter(row, filterValue) {
    if (filterValue === 'all') return true;
    
    // Check for specific filter criteria
    if (filterValue === 'high') {
        return row.querySelector('.badge-success') !== null;
    } else if (filterValue === 'medium') {
        return row.querySelector('.badge-warning') !== null;
    } else if (filterValue === 'low') {
        return row.querySelector('.badge-secondary') !== null;
    }
    
    return true;
}

/**
 * Export table data to CSV
 */
function exportTableToCSV(tableId, filename = 'export.csv') {
    const table = document.getElementById(tableId);
    if (!table) {
        showNotification('Table not found for export', 'error');
        return;
    }
    
    const rows = table.querySelectorAll('tr');
    const csvData = [];
    
    rows.forEach(row => {
        const rowData = [];
        const cells = row.querySelectorAll('th, td');
        
        cells.forEach(cell => {
            // Remove action buttons and icons from export
            const clone = cell.cloneNode(true);
            const buttons = clone.querySelectorAll('button');
            buttons.forEach(btn => btn.remove());
            
            const icons = clone.querySelectorAll('i');
            icons.forEach(icon => icon.remove());
            
            rowData.push(`"${clone.textContent.trim().replace(/"/g, '""')}"`);
        });
        
        csvData.push(rowData.join(','));
    });
    
    const csvContent = csvData.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
    showNotification('Data exported successfully!', 'success');
}

// ============================================
// THEME & UI MANAGEMENT
// ============================================

/**
 * Toggle dark/light theme
 */
function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    const newTheme = isDark ? 'light' : 'dark';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update theme icon
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.innerHTML = newTheme === 'dark' ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    }
    
    showNotification(`Switched to ${newTheme} theme`, 'info', 2000);
}

/**
 * Initialize theme from localStorage or system preference
 */
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    const theme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
    
    // Set initial icon
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.innerHTML = theme === 'dark' ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    }
}

/**
 * Initialize responsive sidebar (if exists)
 */
function initSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');
    
    if (!sidebar || !toggleBtn) return;
    
    toggleBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
    });
    
    // Restore collapsed state
    const collapsed = localStorage.getItem('sidebar-collapsed') === 'true';
    if (collapsed) {
        sidebar.classList.add('collapsed');
    }
}

// ============================================
// DATA LOADING & VALIDATION
// ============================================

/**
 * Validate data integrity
 */
async function validateData() {
    try {
        const response = await fetch('data/latest.json');
        if (!response.ok) throw new Error('Data file not found');
        
        const data = await response.json();
        
        // Check if data is stale (older than 10 minutes)
        const lastUpdated = new Date(data.last_updated);
        const now = new Date();
        const minutesDiff = (now - lastUpdated) / (1000 * 60);
        
        if (minutesDiff > 10) {
            showNotification(`Data is ${Math.floor(minutesDiff)} minutes old. System may need attention.`, 'warning', 10000);
            return false;
        }
        
        // Check for critical data
        if (data.total_predictions === 0 && data.total_alerts === 0) {
            showNotification('No data available. System may be idle.', 'info', 5000);
            return false;
        }
        
        return true;
    } catch (error) {
        console.error('Data validation failed:', error);
        showNotification('Could not validate data integrity', 'error');
        return false;
    }
}

/**
 * Load latest data stats
 */
async function loadLatestStats() {
    try {
        const response = await fetch('data/latest.json');
        const data = await response.json();
        
        // Update stats in real-time if elements exist
        const elements = {
            'live-games-count': data.live_games,
            'total-predictions': data.total_predictions,
            'total-alerts': data.total_alerts,
            'avg-confidence': `${Math.round(data.avg_confidence)}%`
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                const current = parseInt(element.textContent) || 0;
                if (current !== value) {
                    // Animate value change
                    animateValue(element, current, value, 1000);
                }
            }
        });
        
        return data;
    } catch (error) {
        console.error('Error loading latest stats:', error);
        return null;
    }
}

/**
 * Animate value changes
 */
function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = Math.floor(start + (end - start) * progress);
        element.textContent = currentValue;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize all dashboard functionality
 */
function initDashboard() {
    console.log('Initializing Over/Under Predictor Dashboard v2.0');
    
    // Initialize theme
    initTheme();
    
    // Initialize sidebar if exists
    initSidebar();
    
    // Start auto-refresh countdown
    startCountdown();
    
    // Load charts
    loadConfidenceChart();
    loadLeagueChart();
    loadHistoricalChart();
    
    // Initialize pagination for alerts table
    initPagination('alertsTable');
    
    // Set up search and filter for alerts
    const searchInput = document.getElementById('alertSearch');
    const filterSelect = document.getElementById('alertFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', debounce(() => {
            filterTable('alertsTable', 'alertSearch', 'alertFilter');
        }, 300));
    }
    
    if (filterSelect) {
        filterSelect.addEventListener('change', () => {
            filterTable('alertsTable', 'alertSearch', 'alertFilter');
        });
    }
    
    // Set up export button
    const exportBtn = document.getElementById('exportAlerts');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            exportTableToCSV('alertsTable', `alerts_${new Date().toISOString().split('T')[0]}.csv`);
        });
    }
    
    // Set up theme toggle button
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }
    
    // Set up refresh toggle button
    const refreshBtn = document.getElementById('toggle-refresh');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', toggleAutoRefresh);
    }
    
    // Validate data integrity
    setTimeout(() => {
        validateData();
    }, 2000);
    
    // Periodically update stats
    setInterval(() => {
        loadLatestStats();
    }, 30000); // Every 30 seconds
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl+R to refresh
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            refreshPage();
        }
        
        // Esc to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal');
            modals.forEach(modal => modal.remove());
        }
        
        // Space to pause/resume auto-refresh
        if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
            toggleAutoRefresh();
        }
    });
    
    // Show welcome notification
    setTimeout(() => {
        showNotification('Dashboard loaded successfully! Auto-refresh is enabled.', 'success', 5000);
    }, 1000);
    
    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        destroyAllCharts();
        if (countdownTimer) clearInterval(countdownTimer);
    });
}

/**
 * Initialize alerts page specifically
 */
function initAlertsPage() {
    console.log('Initializing Alerts Page');
    
    initPagination('alertsTable');
    
    const searchInput = document.getElementById('alertSearch');
    const filterSelect = document.getElementById('alertFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', debounce(() => {
            filterTable('alertsTable', 'alertSearch', 'alertFilter');
        }, 300));
    }
    
    if (filterSelect) {
        filterSelect.addEventListener('change', () => {
            filterTable('alertsTable', 'alertSearch', 'alertFilter');
        });
    }
    
    // Initialize export functionality
    const exportBtn = document.getElementById('exportAlerts');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            exportTableToCSV('alertsTable', `alerts_${new Date().toISOString().split('T')[0]}.csv`);
        });
    }
}

/**
 * Initialize statistics page
 */
function initStatisticsPage() {
    console.log('Initializing Statistics Page');
    // Additional stats page initialization
    loadHistoricalChart();
}

// ============================================
// AUTO-DETECT PAGE TYPE AND INITIALIZE
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    const body = document.body;
    
    // Detect page type from body class or URL
    if (body.classList.contains('dashboard-page') || window.location.pathname.endsWith('index.html') || window.location.pathname === '/') {
        initDashboard();
    } else if (body.classList.contains('alerts-page') || window.location.pathname.includes('alerts.html')) {
        initAlertsPage();
    } else if (body.classList.contains('stats-page') || window.location.pathname.includes('statistics.html')) {
        initStatisticsPage();
    } else {
        // Generic initialization for other pages
        initTheme();
        startCountdown();
    }
    
    // Add CSS for notifications
    addNotificationStyles();
});

/**
 * Add notification styles dynamically
 */
function addNotificationStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .notifications-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            max-width: 350px;
        }
        
        .notification {
            background: white;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease;
        }
        
        .notification-success {
            border-left-color: #10b981;
        }
        
        .notification-error {
            border-left-color: #ef4444;
        }
        
        .notification-warning {
            border-left-color: #f59e0b;
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .notification-close {
            background: none;
            border: none;
            cursor: pointer;
            opacity: 0.6;
            transition: opacity 0.2s;
        }
        
        .notification-close:hover {
            opacity: 1;
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
        
        [data-theme="dark"] .notification {
            background: #1e293b;
            color: #f1f5f9;
        }
    `;
    document.head.appendChild(style);
}

// ============================================
// PUBLIC API
// ============================================

// Export functions for use in HTML onclick attributes
window.trackAlert = trackAlert;
window.shareAlert = shareAlert;
window.viewGameDetails = viewGameDetails;
window.exportTableToCSV = exportTableToCSV;
window.toggleTheme = toggleTheme;
window.toggleAutoRefresh = toggleAutoRefresh;
window.refreshPage = refreshPage;

console.log('✅ Over/Under Predictor JavaScript loaded successfully');
