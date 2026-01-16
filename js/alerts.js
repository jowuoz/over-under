
// Alerts page JavaScript
function exportAlerts() {
    const table = document.getElementById('alertsTable');
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    
    const csvContent = [
        ['Home Team', 'Away Team', 'League', 'Score', 'Minute', 'Over 2.5 Probability', 'Confidence', 'Status', 'Time'],
        ...rows.map(row => {
            const cells = row.querySelectorAll('td');
            return [
                cells[0].textContent.split(' vs ')[0].trim(),
                cells[0].textContent.split(' vs ')[1].trim(),
                cells[1].textContent,
                cells[2].textContent,
                cells[3].textContent,
                cells[4].querySelector('span').textContent,
                cells[5].textContent,
                cells[6].textContent,
                cells[7].textContent
            ];
        })
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alerts_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function viewAlertDetails(alertId) {
    console.log('Viewing alert details:', alertId);
    // Implement alert details modal
}

// Search and filter functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('alertSearch');
    const filterSelect = document.getElementById('alertFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', filterAlerts);
    }
    
    if (filterSelect) {
        filterSelect.addEventListener('change', filterAlerts);
    }
    
    // Initialize pagination
    initPagination();
});

function filterAlerts() {
    const searchTerm = document.getElementById('alertSearch').value.toLowerCase();
    const filterValue = document.getElementById('alertFilter').value;
    const rows = document.querySelectorAll('#alertsTable tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        const confidence = row.querySelector('td:nth-child(6)').textContent;
        
        let matchesSearch = searchTerm === '' || text.includes(searchTerm);
        let matchesFilter = filterValue === 'all' || 
                           (filterValue === 'high' && parseFloat(confidence) >= 80) ||
                           (filterValue === 'medium' && parseFloat(confidence) >= 60 && parseFloat(confidence) < 80) ||
                           (filterValue === 'low' && parseFloat(confidence) < 60);
        
        row.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
    });
}

function initPagination() {
    const rows = document.querySelectorAll('#alertsTable tbody tr');
    const rowsPerPage = 20;
    const pageCount = Math.ceil(rows.length / rowsPerPage);
    
    if (pageCount <= 1) return;
    
    const pagination = document.getElementById('alertsPagination');
    pagination.innerHTML = '';
    
    for (let i = 1; i <= pageCount; i++) {
        const button = document.createElement('button');
        button.textContent = i;
        button.className = 'page-btn';
        if (i === 1) button.classList.add('active');
        
        button.addEventListener('click', () => {
            showPage(i);
            document.querySelectorAll('.page-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
        
        pagination.appendChild(button);
    }
    
    showPage(1);
}

function showPage(pageNum) {
    const rows = document.querySelectorAll('#alertsTable tbody tr');
    const rowsPerPage = 20;
    const start = (pageNum - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    
    rows.forEach((row, index) => {
        row.style.display = (index >= start && index < end) ? '' : 'none';
    });
}
