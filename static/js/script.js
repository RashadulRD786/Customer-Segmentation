// script.js

// --- DOM Element References ---
const excelFileInput = document.getElementById('excelFileInput');
const dropZone = document.getElementById('drop-zone');
const fileStatus = document.getElementById('file-status');
const nClustersInput = document.getElementById('nClusters');
const nClustersSlider = document.getElementById('nClustersSlider');
const analyzeButton = document.getElementById('analyzeButton');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsArea = document.getElementById('results-area');
const treemapPlot = document.getElementById('treemap-plot');
const clusterSummaryTableBody = document.querySelector('#cluster-summary-table tbody');
const exportButton = document.getElementById('exportButton');
const notificationArea = document.getElementById('notification-area');

let selectedFile = null; // To store the file selected by user

// --- Notification/Message Handling ---
function showMessage(type, text) {
    const notification = document.createElement('div');
    notification.className = `p-4 rounded-lg shadow-lg flex items-center space-x-3 transition-all duration-300 transform translate-x-full opacity-0`;

    // Icon based on type
    let iconSvg = '';
    if (type === 'success') {
        notification.classList.add('bg-green-500', 'text-white');
        iconSvg = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
    } else if (type === 'error') {
        notification.classList.add('bg-red-500', 'text-white');
        iconSvg = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2A9 9 0 1112 1 9 9 0 0121 12z"></path></svg>`;
    } else if (type === 'info') {
        notification.classList.add('bg-blue-500', 'text-white');
        iconSvg = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
    }

    notification.innerHTML = `${iconSvg}<p class="font-medium">${text}</p>`;
    notificationArea.appendChild(notification);

    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full', 'opacity-0');
        notification.classList.add('translate-x-0', 'opacity-100');
    }, 10); // Small delay to trigger transition

    // Animate out and remove
    setTimeout(() => {
        notification.classList.remove('translate-x-0', 'opacity-100');
        notification.classList.add('translate-x-full', 'opacity-0');
        notification.addEventListener('transitionend', () => notification.remove(), { once: true });
    }, 5000); // Notification visible for 5 seconds
}

// --- File Input Handling ---
excelFileInput.addEventListener('change', (event) => {
    selectedFile = event.target.files[0];
    if (selectedFile) {
        fileStatus.textContent = `File selected: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(2)} KB)`;
        analyzeButton.disabled = false; // Enable analyze button
        analyzeButton.classList.remove('bg-gray-400', 'cursor-not-allowed');
        analyzeButton.classList.add('bg-green-600', 'hover:bg-green-700', 'focus:ring-green-300');
        showMessage('info', `File selected: ${selectedFile.name}`);
        // Hide previous results if a new file is uploaded
        resultsArea.classList.add('hidden');
    } else {
        fileStatus.textContent = 'No file selected.';
        analyzeButton.disabled = true; // Disable analyze button
        analyzeButton.classList.remove('bg-green-600', 'hover:bg-green-700', 'focus:ring-green-300');
        analyzeButton.classList.add('bg-gray-400', 'cursor-not-allowed');
    }
});

// --- Drag and Drop Handling ---
dropZone.addEventListener('click', () => excelFileInput.click());
dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    excelFileInput.files = event.dataTransfer.files; // Assign dropped files to input
    excelFileInput.dispatchEvent(new Event('change')); // Trigger change event manually
});

// --- n_clusters Input Sync ---
nClustersInput.addEventListener('input', () => {
    const value = parseInt(nClustersInput.value);
    if (!isNaN(value) && value >= 2 && value <= 10) {
        nClustersSlider.value = value;
    }
});

nClustersSlider.addEventListener('input', () => {
    nClustersInput.value = nClustersSlider.value;
});

// --- Main Analysis Logic ---
analyzeButton.addEventListener('click', async () => {
    if (!selectedFile) {
        showMessage('error', 'Please upload an Excel file first.');
        return;
    }

    const nClusters = parseInt(nClustersInput.value);
    if (isNaN(nClusters) || nClusters < 2 || nClusters > 10) {
        showMessage('error', 'Number of clusters must be between 2 and 10.');
        return;
    }

    setLoadingState(true);
    showMessage('info', 'Processing data and running segmentation. This may take a moment...');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('nClusters', nClusters);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (result.success) {
            showMessage('success', result.message);
            renderClusterTable(result.clusterSummary);
            renderTreemap(result.treemapJson);
            resultsArea.classList.remove('hidden'); // Show results section
        } else {
            showMessage('error', result.message);
            resultsArea.classList.add('hidden'); // Hide results on error
        }
    } catch (error) {
        console.error('Fetch error during analysis:', error);
        showMessage('error', 'Network or server error during analysis. Please try again.');
        resultsArea.classList.add('hidden');
    } finally {
        setLoadingState(false);
    }
});

// --- Render Cluster Summary Table ---
function renderClusterTable(clusterSummary) {
    clusterSummaryTableBody.innerHTML = ''; // Clear previous rows
    clusterSummary.forEach(cluster => {
        const row = clusterSummaryTableBody.insertRow();
        row.className = 'hover:bg-gray-100 transition'; // Add hover effect
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${cluster.cluster}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">${cluster.Auto_Cluster_Name}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">${cluster.Customer_Count}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">${cluster.Customer_Percentage.toFixed(2)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">${cluster.Recency_Mean.toFixed(0)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">${cluster.Frequency_Mean.toFixed(1)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">${cluster.Monetary_Mean.toFixed(2)}</td>
        `;
    });
}

// --- Render Plotly Treemap ---
function renderTreemap(treemapJson) {
    if (!treemapJson) {
        treemapPlot.innerHTML = '<p class="text-center text-red-500">Could not generate treemap. Data might be insufficient.</p>';
        return;
    }
    const treemapData = JSON.parse(treemapJson);
    Plotly.newPlot(treemapPlot, treemapData.data, treemapData.layout, { responsive: true });
}

// --- Export to Excel Logic ---
exportButton.addEventListener('click', async () => {
    showMessage('info', 'Generating Excel report... this may take a moment.');
    try {
        // We don't send data with GET for export, it uses the session ID stored on backend
        const response = await fetch('/export_clusters', {
            method: 'GET',
        });

        if (response.ok) {
            // Get the filename from Content-Disposition header
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = 'customer_segmentation_report.xlsx'; // Default filename
            if (contentDisposition && contentDisposition.includes('filename=')) {
                filename = contentDisposition.split('filename=')[1].trim().replace(/"/g, '');
            }

            // Create a Blob from the response to trigger file download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            showMessage('success', 'Excel report downloaded successfully!');
        } else {
            const errorResult = await response.json();
            showMessage('error', `Export failed: ${errorResult.message || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Fetch error during export:', error);
        showMessage('error', 'Network or server error during export. Please try again.');
    }
});

// --- Loading State Management ---
function setLoadingState(isLoading) {
    if (isLoading) {
        analyzeButton.disabled = true;
        analyzeButton.classList.add('bg-gray-400', 'cursor-not-allowed');
        analyzeButton.classList.remove('bg-green-600', 'hover:bg-green-700', 'focus:ring-green-300');
        loadingIndicator.classList.remove('hidden');
        resultsArea.classList.add('hidden'); // Hide results when processing starts
    } else {
        analyzeButton.disabled = false;
        // Only re-enable green if a file is selected
        if (selectedFile) {
            analyzeButton.classList.remove('bg-gray-400', 'cursor-not-allowed');
            analyzeButton.classList.add('bg-green-600', 'hover:bg-green-700', 'focus:ring-green-300');
        }
        loadingIndicator.classList.add('hidden');
    }
}

// Initial state: Disable analyze button until file is selected
analyzeButton.disabled = true;
analyzeButton.classList.add('bg-gray-400', 'cursor-not-allowed');
analyzeButton.classList.remove('bg-green-600', 'hover:bg-green-700', 'focus:ring-green-300'); // Ensure initial state is grayed out