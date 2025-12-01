/**
 * Credit Spreads Screener - Frontend Application
 * Polished UI with mobile-first responsive design
 */

// ============================================
// UTILITY FUNCTIONS
// ============================================

// Escape HTML to prevent XSS attacks
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// Format number safely with specified decimals
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return Number(num).toFixed(decimals);
}

// Format currency value
function formatCurrency(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return '$' + Number(num).toFixed(decimals);
}

// Format percentage value
function formatPercent(num, decimals = 0) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return Number(num * 100).toFixed(decimals) + '%';
}

// Check if viewport is mobile
function isMobile() {
    return window.innerWidth <= 640;
}

// ============================================
// STATE MANAGEMENT
// ============================================

let state = {
    spreads: [],
    pcsSpreads: [],
    ccsSpreads: [],
    loading: false,
    error: null,
    sortColumn: 'total_score',
    sortDirection: 'desc',
};

// ============================================
// DOM ELEMENTS
// ============================================

const elements = {
    // Form inputs
    tickerSelect: document.getElementById('tickerSelect'),
    spreadTypeSelect: document.getElementById('spreadTypeSelect'),
    minDte: document.getElementById('minDte'),
    maxDte: document.getElementById('maxDte'),
    minDelta: document.getElementById('minDelta'),
    maxDelta: document.getElementById('maxDelta'),
    maxWidth: document.getElementById('maxWidth'),
    minRoc: document.getElementById('minRoc'),
    runScreenerBtn: document.getElementById('runScreenerBtn'),

    // Summary cards
    symbolDisplay: document.getElementById('symbolDisplay'),
    underlyingPrice: document.getElementById('underlyingPrice'),
    ivpDisplay: document.getElementById('ivpDisplay'),
    totalSpreads: document.getElementById('totalSpreads'),
    pcsCount: document.getElementById('pcsCount'),
    ccsCount: document.getElementById('ccsCount'),
    infoCards: document.getElementById('infoCards'),

    // Results containers
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    pcsTableContainer: document.getElementById('pcsTableContainer'),
    ccsTableContainer: document.getElementById('ccsTableContainer'),
    pcsBody: document.getElementById('pcsBody'),
    ccsBody: document.getElementById('ccsBody'),
    pcsMobileCards: document.getElementById('pcsMobileCards'),
    ccsMobileCards: document.getElementById('ccsMobileCards'),
    noResultsState: document.getElementById('noResultsState'),
    errorDisplay: document.getElementById('errorDisplay'),
    legendSection: document.getElementById('legendSection'),
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Setup all event listeners
function setupEventListeners() {
    // Run screener button click
    elements.runScreenerBtn.addEventListener('click', runScreener);

    // Keyboard shortcut: Enter to run (except in number inputs)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.target.matches('input[type="number"]')) {
            runScreener();
        }
    });

    // Sortable column headers
    document.querySelectorAll('.sortable').forEach(header => {
        header.addEventListener('click', () => {
            const sortKey = header.dataset.sort;
            const tableContainer = header.closest('.results-table-container');
            const tableType = tableContainer.id === 'pcsTableContainer' ? 'pcs' : 'ccs';
            handleSort(sortKey, tableType);
        });
    });

    // Re-render on window resize for mobile/desktop switch
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (state.spreads.length > 0) {
                renderResults();
            }
        }, 250);
    });
}

// ============================================
// API INTERACTION
// ============================================

async function runScreener() {
    // Gather form values
    const symbol = elements.tickerSelect.value;
    const spreadType = elements.spreadTypeSelect.value;
    const minDte = parseInt(elements.minDte.value);
    const maxDte = parseInt(elements.maxDte.value);
    const minDelta = parseFloat(elements.minDelta.value);
    const maxDelta = parseFloat(elements.maxDelta.value);
    const maxWidth = parseFloat(elements.maxWidth.value);
    const minRoc = parseFloat(elements.minRoc.value) / 100;

    // Client-side validation
    if (minDte >= maxDte) {
        showError('Min DTE must be less than Max DTE');
        return;
    }

    if (minDelta >= maxDelta) {
        showError('Min Delta must be less than Max Delta');
        return;
    }

    setLoading(true);
    hideError();

    try {
        const response = await fetch('/api/credit-spreads', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                spread_type: spreadType,
                min_dte: minDte,
                max_dte: maxDte,
                min_delta: minDelta,
                max_delta: maxDelta,
                max_width: maxWidth,
                min_roc: minRoc,
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to fetch credit spreads');
        }

        const data = await response.json();
        updateUI(data);

    } catch (err) {
        console.error('Error fetching credit spreads:', err);
        showError(err.message);
    } finally {
        setLoading(false);
    }
}

// ============================================
// UI UPDATE FUNCTIONS
// ============================================

function updateUI(data) {
    // Update state
    state.spreads = data.spreads;
    state.pcsSpreads = data.spreads.filter(s => s.spread_type === 'PCS');
    state.ccsSpreads = data.spreads.filter(s => s.spread_type === 'CCS');

    // Update summary cards
    elements.symbolDisplay.textContent = data.symbol;
    elements.underlyingPrice.textContent = formatCurrency(data.underlying_price);
    elements.ivpDisplay.textContent = data.ivp.toFixed(0) + '%';
    elements.totalSpreads.textContent = data.spreads.length;
    elements.pcsCount.textContent = data.total_pcs;
    elements.ccsCount.textContent = data.total_ccs;

    // Show summary cards
    elements.infoCards.style.display = 'grid';

    // Hide empty state
    elements.emptyState.style.display = 'none';

    // Handle no results
    if (data.spreads.length === 0) {
        elements.pcsTableContainer.style.display = 'none';
        elements.ccsTableContainer.style.display = 'none';
        elements.noResultsState.style.display = 'block';
        elements.legendSection.style.display = 'none';
        return;
    }

    elements.noResultsState.style.display = 'none';
    elements.legendSection.style.display = 'block';

    // Sort spreads by score
    state.pcsSpreads.sort((a, b) => b.total_score - a.total_score);
    state.ccsSpreads.sort((a, b) => b.total_score - a.total_score);

    // Render results
    renderResults();
}

function renderResults() {
    const spreadType = elements.spreadTypeSelect.value;

    // PCS Section
    if ((spreadType === 'ALL' || spreadType === 'PCS') && state.pcsSpreads.length > 0) {
        elements.pcsTableContainer.style.display = 'block';
        renderTable(state.pcsSpreads, elements.pcsBody);
        renderMobileCards(state.pcsSpreads, elements.pcsMobileCards, 'PCS');
    } else {
        elements.pcsTableContainer.style.display = 'none';
    }

    // CCS Section
    if ((spreadType === 'ALL' || spreadType === 'CCS') && state.ccsSpreads.length > 0) {
        elements.ccsTableContainer.style.display = 'block';
        renderTable(state.ccsSpreads, elements.ccsBody);
        renderMobileCards(state.ccsSpreads, elements.ccsMobileCards, 'CCS');
    } else {
        elements.ccsTableContainer.style.display = 'none';
    }
}

// Render desktop table view
function renderTable(spreads, tbody) {
    tbody.innerHTML = spreads.map((s, idx) => `
        <tr data-spread="${idx}">
            <td class="col-exp">${escapeHtml(s.expiration)}</td>
            <td class="col-dte">${s.dte}</td>
            <td class="col-strike">${formatCurrency(s.short_strike, 0)}</td>
            <td class="col-strike">${formatCurrency(s.long_strike, 0)}</td>
            <td class="col-money">${formatCurrency(s.credit, 2)}</td>
            <td class="col-pct ${s.roc >= 0.30 ? 'positive' : ''}">${formatPercent(s.roc, 1)}</td>
            <td class="col-delta hide-mobile">${formatNumber(s.short_delta, 2)}${s.delta_estimated ? '<span class="delta-estimated">*</span>' : ''}</td>
            <td class="col-pct ${s.prob_profit >= 0.75 ? 'positive' : ''}">${formatPercent(s.prob_profit, 0)}</td>
            <td class="col-money hide-mobile">${formatCurrency(s.break_even, 2)}</td>
            <td class="col-score"><span class="score-badge ${getScoreClass(s.total_score)}">${formatNumber(s.total_score, 2)}</span></td>
        </tr>
    `).join('');
}

// Render mobile card view for better readability on small screens
function renderMobileCards(spreads, container, type) {
    if (!container) return;

    container.innerHTML = spreads.map((s, idx) => `
        <div class="mobile-spread-card" data-spread="${idx}">
            <div class="mobile-card-header">
                <div>
                    <div class="mobile-card-strikes">${formatCurrency(s.short_strike, 0)} / ${formatCurrency(s.long_strike, 0)}</div>
                    <div class="mobile-card-exp">${escapeHtml(s.expiration)} (${s.dte} DTE)</div>
                </div>
                <div class="mobile-card-score">
                    <span class="score-badge ${getScoreClass(s.total_score)}">${formatNumber(s.total_score, 2)}</span>
                </div>
            </div>
            <div class="mobile-card-metrics">
                <div class="mobile-metric">
                    <div class="mobile-metric-label">Credit</div>
                    <div class="mobile-metric-value">${formatCurrency(s.credit, 2)}</div>
                </div>
                <div class="mobile-metric">
                    <div class="mobile-metric-label">ROC</div>
                    <div class="mobile-metric-value ${s.roc >= 0.30 ? 'positive' : ''}">${formatPercent(s.roc, 1)}</div>
                </div>
                <div class="mobile-metric">
                    <div class="mobile-metric-label">Win%</div>
                    <div class="mobile-metric-value ${s.prob_profit >= 0.75 ? 'positive' : ''}">${formatPercent(s.prob_profit, 0)}</div>
                </div>
                <div class="mobile-metric">
                    <div class="mobile-metric-label">Delta</div>
                    <div class="mobile-metric-value">${formatNumber(s.short_delta, 2)}${s.delta_estimated ? '*' : ''}</div>
                </div>
                <div class="mobile-metric">
                    <div class="mobile-metric-label">Break-Even</div>
                    <div class="mobile-metric-value">${formatCurrency(s.break_even, 2)}</div>
                </div>
                <div class="mobile-metric">
                    <div class="mobile-metric-label">Max Loss</div>
                    <div class="mobile-metric-value">${formatCurrency(s.max_loss, 2)}</div>
                </div>
            </div>
        </div>
    `).join('');
}

// Get CSS class for score badge styling
function getScoreClass(score) {
    if (score >= 0.7) return 'score-high';
    if (score >= 0.4) return 'score-medium';
    return 'score-low';
}

// ============================================
// SORTING FUNCTIONALITY
// ============================================

function handleSort(column, tableType) {
    // Toggle direction if same column, else default to descending
    if (state.sortColumn === column) {
        state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        state.sortColumn = column;
        state.sortDirection = 'desc';
    }

    const spreads = tableType === 'pcs' ? state.pcsSpreads : state.ccsSpreads;
    const tbody = tableType === 'pcs' ? elements.pcsBody : elements.ccsBody;
    const mobileContainer = tableType === 'pcs' ? elements.pcsMobileCards : elements.ccsMobileCards;

    // Perform sort
    spreads.sort((a, b) => {
        let aVal = a[column];
        let bVal = b[column];

        // String comparison for expiration dates
        if (column === 'expiration') {
            return state.sortDirection === 'asc'
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal);
        }

        // Numeric comparison for everything else
        return state.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    });

    // Re-render both views
    renderTable(spreads, tbody);
    renderMobileCards(spreads, mobileContainer, tableType.toUpperCase());

    // Update sort indicators in headers
    updateSortIndicators(tableType);
}

function updateSortIndicators(tableType) {
    const container = tableType === 'pcs' ? elements.pcsTableContainer : elements.ccsTableContainer;
    container.querySelectorAll('.sortable').forEach(header => {
        header.classList.remove('sort-asc', 'sort-desc');
        if (header.dataset.sort === state.sortColumn) {
            header.classList.add(state.sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
        }
    });
}

// ============================================
// LOADING & ERROR STATES
// ============================================

function setLoading(loading) {
    state.loading = loading;
    elements.runScreenerBtn.disabled = loading;

    // Update button text during loading
    if (loading) {
        elements.runScreenerBtn.innerHTML = '<span class="btn-icon">&#8635;</span> Scanning...';
        elements.loadingState.style.display = 'flex';
        elements.pcsTableContainer.style.display = 'none';
        elements.ccsTableContainer.style.display = 'none';
        elements.emptyState.style.display = 'none';
        elements.noResultsState.style.display = 'none';
    } else {
        elements.runScreenerBtn.innerHTML = '<span class="btn-icon">&#9654;</span> Run Screener';
        elements.loadingState.style.display = 'none';
    }
}

function showError(message) {
    const errorBanner = elements.errorDisplay;
    const errorMessage = errorBanner.querySelector('.error-message');
    if (errorMessage) {
        errorMessage.textContent = message;
    } else {
        errorBanner.textContent = message;
    }
    errorBanner.style.display = 'flex';
}

function hideError() {
    elements.errorDisplay.style.display = 'none';
}
