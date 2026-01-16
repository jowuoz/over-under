// statistics.js - Statistics Page JavaScript for Over/Under Predictor
console.log('📊 Statistics page loaded');

// DOM Elements
const performanceMetrics = {
    responseTimeChart: null,
    errorRateChart: null,
    uptimeChart: null
};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing statistics page...');
    
    // Load performance data
    loadPerformanceData();
    
    // Initialize charts
    initializeCharts();
    
    // Set up periodic updates
    startAutoRefresh();
    
    // Set up event listeners
    setupEventListeners();
    
    console.log('Statistics page initialized successfully');
});

// Load performance data from the server/API
function loadPerformanceData() {
    console.log('Loading performance data...');
    
    // Try to load from local data file first
    fetch('data/performance.json')
        .then(response => {
            if (!response.ok) throw new Error('Performance data not found');
            return response.json();
        })
        .then(data => {
            console.log('Performance data loaded:', data);
            updateMetricsDisplay(data);
            updateChartsWithData(data);
        })
        .catch(error => {
            console.warn('Could not load performance data:', error);
            // Load sample data for demonstration
            loadSamplePerformanceData();
        });
}

// Update the metrics display with loaded data
function updateMetricsDisplay(data) {
    console.log('Updating metrics display with data:', data);
    
    // Update accuracy rate
    const accuracyElement = document.querySelector('.stat-large-number');
    if (accuracyElement && data.accuracy_rate) {
        accuracyElement.textContent = `${data.accuracy_rate}%`;
    }
    
    // Update response time
    const responseTimeElement = document.querySelectorAll('.stat-large-number')[2];
    if (responseTimeElement && data.avg_response_time) {
        responseTimeElement.textContent = `${data.avg_response_time}ms`;
    }
    
    // Update detailed statistics
    const detailStats = document.querySelectorAll('.detail-number');
    if (detailStats.length >= 4) {
        if (data.leagues_covered) detailStats[0].textContent = data.leagues_covered;
        if (data.avg_games_per_run) detailStats[1].textContent = `${data.avg_games_per_run}`;
        if (data.over_25_success) detailStats[2].textContent = `${data.over_25_success}%`;
        if (data.uptime) detailStats[3].textContent = `${data.uptime}%`;
    }
    
    // Update source counts
    const sourceCounts = document.querySelectorAll('.source-count');
    if (sourceCounts.length >= 4) {
        if (data.api_sources) sourceCounts[0].textContent = data.api_sources;
        if (data.web_scrapers) sourceCounts[1].textContent = data.web_scrapers;
        if (data.cache_hit_rate) sourceCounts[2].textContent = `${data.cache_hit_rate}%`;
        if (data.data_reliability) sourceCounts[3].textContent = `${data.data_reliability}%`;
    }
}

// Initialize all charts on the page
function initializeCharts() {
    console.log('Initializing charts...');
    
    // Create response time chart
    const responseTimeCtx = document.getElementById('responseTimeChart');
    if (responseTimeCtx) {
        performanceMetrics.responseTimeChart = new Chart(responseTimeCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: getChartOptions('Response Time Over Time', 'ms')
        });
    }
    
    // Create error rate chart
    const errorRateCtx = document.getElementById('errorRateChart');
    if (errorRateCtx) {
        performanceMetrics.errorRateChart = new Chart(errorRateCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Error Rate (%)',
                    data: [],
                    backgroundColor: '#ef4444',
                    borderColor: '#dc2626',
                    borderWidth: 1
                }]
            },
            options: getChartOptions('Error Rate by Hour', '%')
        });
    }
    
    // Create uptime chart
    const uptimeCtx = document.getElementById('uptimeChart');
    if (uptimeCtx) {
        performanceMetrics.uptimeChart = new Chart(uptimeCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Uptime', 'Downtime'],
                datasets: [{
                    data: [99.8, 0.2],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Update charts with loaded data
function updateChartsWithData(data) {
    console.log('Updating charts with data:', data);
    
    if (data.response_time_history && performanceMetrics.responseTimeChart) {
        const labels = data.response_time_history.map((_, index) => `${index * 5} min ago`);
        performanceMetrics.responseTimeChart.data.labels = labels;
        performanceMetrics.responseTimeChart.data.datasets[0].data = data.response_time_history;
        performanceMetrics.responseTimeChart.update();
    }
    
    if (data.error_rate_history && performanceMetrics.errorRateChart) {
        const labels = data.error_rate_history.map((_, index) => `Hour ${index + 1}`);
        performanceMetrics.errorRateChart.data.labels = labels;
        performanceMetrics.errorRateChart.data.datasets[0].data = data.error_rate_history;
        performanceMetrics.errorRateChart.update();
    }
}

// Load sample data for demonstration
function loadSamplePerformanceData() {
    console.log('Loading sample performance data...');
    
    const sampleData = {
        accuracy_rate: 78.5,
        avg_response_time: 245,
        error_rate: 2.3,
        leagues_covered: 15,
        avg_games_per_run: 42.7,
        over_25_success: 82,
        uptime: 99.8,
        api_sources: 3,
        web_scrapers: 2,
        cache_hit_rate: 65,
        data_reliability: 94,
        response_time_history: generateSampleData(12, 200, 300),
        error_rate_history: generateSampleData(24, 1, 5),
        last_updated: new Date().toISOString()
    };
    
    updateMetricsDisplay(sampleData);
    updateChartsWithData(sampleData);
    
    // Save sample data for future reference
    saveSampleData(sampleData);
}

// Generate sample data for charts
function generateSampleData(count, min, max) {
    const data = [];
    for (let i = 0; i < count; i++) {
        // Add some randomness and trend
        const baseValue = min + (max - min) * (i / count);
        const randomFactor = 0.8 + Math.random() * 0.4;
        data.push(Math.round(baseValue * randomFactor));
    }
    return data;
}

// Save sample data to localStorage for consistency
function saveSampleData(data) {
    try {
        localStorage.setItem('overUnderSampleData', JSON.stringify(data));
        console.log('Sample data saved to localStorage');
    } catch (error) {
        console.warn('Could not save sample data to localStorage:', error);
    }
}

// Get chart options with consistent styling
function getChartOptions(title, unit) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: {
                    font: {
                        size: 12
                    }
                }
            },
            title: {
                display: true,
                text: title,
                font: {
                    size: 14,
                    weight: 'bold'
                },
                padding: {
                    bottom: 20
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return `${context.dataset.label}: ${context.raw}${unit}`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                },
                ticks: {
                    font: {
                        size: 11
                    }
                }
            },
            x: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.05)'
                },
                ticks: {
                    font: {
                        size: 11
                    },
                    maxRotation: 45
                }
            }
        }
    };
}

// Set up event listeners
function setupEventListeners() {
    console.log('Setting up event listeners...');
    
    // Refresh button
    const refreshBtn = document.querySelector('.btn-refresh');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            console.log('Manual refresh requested');
            loadPerformanceData();
            showNotification('Performance data refreshed', 'success');
        });
    }
    
    // Export button
    const exportBtn = document.querySelector('.btn-export');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportStatistics);
    }
    
    // Time range selector
    const timeRangeSelect = document.getElementById('timeRange');
    if (timeRangeSelect) {
        timeRangeSelect.addEventListener('change', function() {
            console.log('Time range changed to:', this.value);
            updateChartsForTimeRange(this.value);
        });
    }
    
    // Chart type toggles
    const chartTypeToggles = document.querySelectorAll('.chart-type-toggle');
    chartTypeToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const chartType = this.dataset.chartType;
            console.log('Switching to chart type:', chartType);
            switchChartType(chartType);
        });
    });
}

// Export statistics as CSV
function exportStatistics() {
    console.log('Exporting statistics...');
    
    // In a real implementation, you would fetch actual data
    const data = {
        timestamp: new Date().toISOString(),
        accuracy_rate: 78.5,
        avg_response_time: 245,
        error_rate: 2.3,
        leagues_covered: 15,
        avg_games_per_run: 42.7,
        over_25_success: 82,
        uptime: 99.8,
        api_sources: 3,
        web_scrapers: 2,
        cache_hit_rate: 65,
        data_reliability: 94
    };
    
    // Convert to CSV
    const csvContent = Object.entries(data)
        .map(([key, value]) => `${key},${value}`)
        .join('\n');
    
    // Add headers
    const headers = 'Metric,Value\n';
    const fullCSV = headers + csvContent;
    
    // Create download link
    const blob = new Blob([fullCSV], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `over-under-statistics-${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showNotification('Statistics exported successfully', 'success');
}

// Update charts based on selected time range
function updateChartsForTimeRange(range) {
    console.log('Updating charts for time range:', range);
    
    // Generate appropriate data based on time range
    let dataPoints;
    switch(range) {
        case '24h':
            dataPoints = 24;
            break;
        case '7d':
            dataPoints = 7;
            break;
        case '30d':
            dataPoints = 30;
            break;
        default:
            dataPoints = 12;
    }
    
    // Update response time chart
    if (performanceMetrics.responseTimeChart) {
        const labels = Array.from({length: dataPoints}, (_, i) => {
            if (range === '24h') return `${i}:00`;
            if (range === '7d') return `Day ${i + 1}`;
            return `Week ${i + 1}`;
        });
        
        performanceMetrics.responseTimeChart.data.labels = labels;
        performanceMetrics.responseTimeChart.data.datasets[0].data = 
            generateSampleData(dataPoints, 200, 300);
        performanceMetrics.responseTimeChart.update();
    }
    
    // Update error rate chart
    if (performanceMetrics.errorRateChart) {
        const labels = Array.from({length: dataPoints}, (_, i) => {
            if (range === '24h') return `${i}:00`;
            if (range === '7d') return `Day ${i + 1}`;
            return `Week ${i + 1}`;
        });
        
        performanceMetrics.errorRateChart.data.labels = labels;
        performanceMetrics.errorRateChart.data.datasets[0].data = 
            generateSampleData(dataPoints, 1, 5);
        performanceMetrics.errorRateChart.update();
    }
}

// Switch between different chart types
function switchChartType(chartType) {
    console.log('Switching chart type to:', chartType);
    
    if (!performanceMetrics.responseTimeChart) return;
    
    // Update chart type
    performanceMetrics.responseTimeChart.config.type = chartType;
    
    // Adjust styling based on chart type
    if (chartType === 'line') {
        performanceMetrics.responseTimeChart.data.datasets[0].fill = true;
        performanceMetrics.responseTimeChart.data.datasets[0].tension = 0.4;
    } else if (chartType === 'bar') {
        performanceMetrics.responseTimeChart.data.datasets[0].fill = false;
        performanceMetrics.responseTimeChart.data.datasets[0].tension = 0;
    }
    
    performanceMetrics.responseTimeChart.update();
    
    // Update active toggle button
    document.querySelectorAll('.chart-type-toggle').forEach(btn => {
        if (btn.dataset.chartType === chartType) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// Start auto-refresh timer
function startAutoRefresh() {
    console.log('Starting auto-refresh (every 5 minutes)...');
    
    setInterval(() => {
        console.log('Auto-refreshing performance data...');
        loadPerformanceData();
    }, 5 * 60 * 1000); // 5 minutes
}

// Show notification to user
function showNotification(message, type = 'info') {
    console.log(`Notification (${type}):`, message);
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close">&times;</button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show with animation
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
    
    // Close button functionality
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    });
}

// Utility function to format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Add CSS for notifications
function addNotificationStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 1000;
            transform: translateX(120%);
            transition: transform 0.3s ease;
            max-width: 350px;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification-success {
            border-left: 4px solid #10b981;
        }
        
        .notification-info {
            border-left: 4px solid #3b82f6;
        }
        
        .notification-warning {
            border-left: 4px solid #f59e0b;
        }
        
        .notification i {
            font-size: 1.2em;
        }
        
        .notification-success i {
            color: #10b981;
        }
        
        .notification-info i {
            color: #3b82f6;
        }
        
        .notification-warning i {
            color: #f59e0b;
        }
        
        .notification span {
            flex: 1;
            font-size: 0.9em;
        }
        
        .notification-close {
            background: none;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
            color: #64748b;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
        }
        
        .notification-close:hover {
            background: #f1f5f9;
            color: #475569;
        }
    `;
    document.head.appendChild(style);
}

// Add notification styles when page loads
addNotificationStyles();

// Error handling for missing chart elements
window.addEventListener('error', function(e) {
    if (e.message.includes('Chart') || e.message.includes('canvas')) {
        console.warn('Chart.js related error:', e.message);
        // Fallback to showing data in tables
        showDataInTables();
    }
});

// Fallback: show data in tables if charts fail
function showDataInTables() {
    console.log('Falling back to table display for statistics');
    
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
        const canvas = container.querySelector('canvas');
        if (canvas) {
            canvas.style.display = 'none';
            
            // Create table as fallback
            const table = document.createElement('div');
            table.className = 'data-table-fallback';
            table.innerHTML = `
                <h4>Data Table (Chart Fallback)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Array.from({length: 10}, (_, i) => `
                            <tr>
                                <td>${i * 5} min ago</td>
                                <td>${Math.round(200 + Math.random() * 100)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            container.appendChild(table);
        }
    });
}

// Public API (if other scripts need to interact with this)
window.StatisticsPage = {
    refresh: loadPerformanceData,
    export: exportStatistics,
    updateTimeRange: updateChartsForTimeRange,
    showNotification: showNotification
};

console.log('📊 Statistics.js loaded successfully - Ready to analyze!');
