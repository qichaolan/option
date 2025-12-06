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
    // Simulator state
    selectedSpread: null,
    simulatorData: null,
    underlyingPrice: 0,
    payoffTableExpanded: false,  // Whether P/L table is expanded
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

    // Simulator elements
    spreadSimulator: document.getElementById('spreadSimulator'),
    closeSimulator: document.getElementById('closeSimulator'),
    simSymbol: document.getElementById('simSymbol'),
    simType: document.getElementById('simType'),
    simExpiration: document.getElementById('simExpiration'),
    simShortStrike: document.getElementById('simShortStrike'),
    simLongStrike: document.getElementById('simLongStrike'),
    simNetCredit: document.getElementById('simNetCredit'),
    simMaxGain: document.getElementById('simMaxGain'),
    simMaxLoss: document.getElementById('simMaxLoss'),
    simBreakeven: document.getElementById('simBreakeven'),
    simBreakevenPct: document.getElementById('simBreakevenPct'),
    simTableBody: document.getElementById('simTableBody'),
    simLoadingState: document.getElementById('simLoadingState'),
    simProfitRange: document.getElementById('simProfitRange'),
    simLegsText: document.getElementById('simLegsText'),
    togglePayoffTable: document.getElementById('togglePayoffTable'),
};

// ============================================
// AI EXPLAINER
// ============================================

let aiExplainerController = null;

/**
 * Get metadata for AI Explainer based on current simulation state
 */
function getAiExplainerMetadata() {
    if (!state.selectedSpread || !state.simulatorData) {
        return null;
    }

    const spread = state.selectedSpread;
    const simData = state.simulatorData;

    return {
        symbol: spread.symbol,
        spread_type: spread.spread_type,
        expiration: spread.expiration,
        short_strike: spread.short_strike,
        long_strike: spread.long_strike,
        net_credit: spread.credit,
        underlying_price: state.underlyingPrice,
        dte: spread.dte,
        short_delta: spread.short_delta,
        prob_profit: spread.prob_profit,
        max_gain: simData.summary.max_gain,
        max_loss: simData.summary.max_loss,
        breakeven_price: simData.summary.breakeven_price,
        breakeven_pct: simData.summary.breakeven_pct,
        roc: spread.roc,
        iv: spread.iv,
        ivp: spread.ivp,
    };
}

/**
 * Initialize AI Explainer controller
 */
function initAiExplainer() {
    if (typeof AiExplainerController === 'undefined') {
        console.warn('AI Explainer not available');
        return;
    }

    aiExplainerController = new AiExplainerController({
        pageId: 'credit_spread_screener',
        contextType: 'spread_simulator',
        getMetadata: getAiExplainerMetadata,
        buttonContainerId: 'aiExplainerBtnContainer',
        panelContainerId: 'aiExplainerPanelContainer',
    });
}

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    initAiExplainer();
    runScreener(); // Auto-scan on page load
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

    // Close simulator button
    if (elements.closeSimulator) {
        elements.closeSimulator.addEventListener('click', closeSimulator);
    }

    // Toggle P/L table expand/collapse
    if (elements.togglePayoffTable) {
        elements.togglePayoffTable.addEventListener('click', togglePayoffTableExpand);
    }

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
    state.underlyingPrice = data.underlying_price;

    // Close simulator when new data is loaded
    closeSimulator();

    // Update summary cards
    elements.symbolDisplay.textContent = data.symbol;
    elements.underlyingPrice.textContent = formatCurrency(data.underlying_price);
    elements.ivpDisplay.textContent = data.ivp.toFixed(0) + '%';
    elements.totalSpreads.textContent = data.spreads.length;
    // pcsCount and ccsCount are optional (removed in compact layout)
    if (elements.pcsCount) elements.pcsCount.textContent = data.total_pcs;
    if (elements.ccsCount) elements.ccsCount.textContent = data.total_ccs;

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

        // Update no results message with helpful info
        const noResultsTitle = elements.noResultsState.querySelector('.empty-state-title');
        const noResultsText = elements.noResultsState.querySelector('.empty-state-text');
        const spreadType = elements.spreadTypeSelect.value;

        if (spreadType === 'PCS' && data.total_ccs > 0) {
            if (noResultsTitle) noResultsTitle.textContent = 'No Put Credit Spreads Found';
            if (noResultsText) noResultsText.innerHTML = `No bullish put spreads meet the current criteria.<br>However, <strong>${data.total_ccs} bearish call spreads</strong> are available. Try switching to "Bearish (Call Credit Spread)".`;
        } else if (spreadType === 'CCS' && data.total_pcs > 0) {
            if (noResultsTitle) noResultsTitle.textContent = 'No Call Credit Spreads Found';
            if (noResultsText) noResultsText.innerHTML = `No bearish call spreads meet the current criteria.<br>However, <strong>${data.total_pcs} bullish put spreads</strong> are available. Try switching to "Bullish (Put Credit Spread)".`;
        } else {
            if (noResultsTitle) noResultsTitle.textContent = 'No Spreads Found';
            if (noResultsText) noResultsText.textContent = 'No spreads match your current criteria. Try adjusting the filters.';
        }
        return;
    }

    elements.noResultsState.style.display = 'none';
    elements.legendSection.style.display = 'block';

    // Sort spreads by score
    state.pcsSpreads.sort((a, b) => b.total_score - a.total_score);
    state.ccsSpreads.sort((a, b) => b.total_score - a.total_score);

    // Render results
    renderResults();

    // Auto-select the top spread for simulation
    autoSelectTopSpread();
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

// Render desktop table view (limited to top 20 by score)
function renderTable(spreads, tbody) {
    const top20 = spreads.slice(0, 20);
    tbody.innerHTML = top20.map((s, idx) => `
        <tr data-spread-idx="${idx}" data-spread-type="${s.spread_type}" class="clickable-row" title="Click to simulate P/L">
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

    // Add click handlers for row selection
    tbody.querySelectorAll('tr[data-spread-idx]').forEach(row => {
        row.addEventListener('click', () => {
            const idx = parseInt(row.dataset.spreadIdx);
            const spreadType = row.dataset.spreadType;
            const spread = spreadType === 'PCS' ? state.pcsSpreads[idx] : state.ccsSpreads[idx];
            selectSpreadForSimulation(spread, row);
        });
    });
}

// Render mobile card view for better readability on small screens (limited to top 20 by score)
function renderMobileCards(spreads, container, type) {
    if (!container) return;

    const top20 = spreads.slice(0, 20);
    container.innerHTML = top20.map((s, idx) => `
        <div class="mobile-spread-card clickable-card" data-spread-idx="${idx}" data-spread-type="${s.spread_type}">
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
            <div class="mobile-card-action">
                <span class="tap-to-simulate">Tap to simulate P/L →</span>
            </div>
        </div>
    `).join('');

    // Add click handlers for card selection
    container.querySelectorAll('.mobile-spread-card[data-spread-idx]').forEach(card => {
        card.addEventListener('click', () => {
            const idx = parseInt(card.dataset.spreadIdx);
            const spreadType = card.dataset.spreadType;
            const spread = spreadType === 'PCS' ? state.pcsSpreads[idx] : state.ccsSpreads[idx];
            selectSpreadForSimulation(spread, card);
        });
    });
}

// Get CSS class for score badge styling
function getScoreClass(score) {
    if (score >= 0.7) return 'score-high';
    if (score >= 0.4) return 'score-medium';
    return 'score-low';
}

// Auto-select the top spread for simulation
function autoSelectTopSpread() {
    const spreadType = elements.spreadTypeSelect.value;
    let topSpread = null;
    let element = null;

    // Determine which spread to select based on filter and availability
    if ((spreadType === 'ALL' || spreadType === 'PCS') && state.pcsSpreads.length > 0) {
        topSpread = state.pcsSpreads[0];
        // Get the first row/card for PCS
        if (isMobile()) {
            element = elements.pcsMobileCards.querySelector('.mobile-spread-card[data-spread-idx="0"]');
        } else {
            element = elements.pcsBody.querySelector('tr[data-spread-idx="0"]');
        }
    } else if ((spreadType === 'ALL' || spreadType === 'CCS') && state.ccsSpreads.length > 0) {
        topSpread = state.ccsSpreads[0];
        // Get the first row/card for CCS
        if (isMobile()) {
            element = elements.ccsMobileCards.querySelector('.mobile-spread-card[data-spread-idx="0"]');
        } else {
            element = elements.ccsBody.querySelector('tr[data-spread-idx="0"]');
        }
    }

    if (topSpread && element) {
        selectSpreadForSimulation(topSpread, element);
    }
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
        elements.runScreenerBtn.innerHTML = '<span class="btn-icon">&#9654;</span> Find Spreads';
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

// ============================================
// CREDIT SPREAD SIMULATOR
// ============================================

// Select a spread for simulation
function selectSpreadForSimulation(spread, element) {
    // Remove selection from all rows/cards
    document.querySelectorAll('.selected-for-sim').forEach(el => {
        el.classList.remove('selected-for-sim');
    });

    // Add selection to clicked element
    element.classList.add('selected-for-sim');

    // Store selected spread
    state.selectedSpread = spread;

    // Run simulation
    runSpreadSimulation(spread);
}

// Close the simulator
function closeSimulator() {
    if (elements.spreadSimulator) {
        elements.spreadSimulator.style.display = 'none';
    }
    state.selectedSpread = null;
    state.simulatorData = null;
    state.payoffTableExpanded = false;  // Reset to collapsed when closing

    // Remove selection highlighting
    document.querySelectorAll('.selected-for-sim').forEach(el => {
        el.classList.remove('selected-for-sim');
    });

    // Clear AI Explainer
    if (aiExplainerController) {
        aiExplainerController.clearExplanation();
    }
}

// Run simulation for a spread
async function runSpreadSimulation(spread) {
    // Clear AI Explainer panel when selecting a new spread
    if (aiExplainerController) {
        aiExplainerController.clearExplanation();
    }

    // Reset table to collapsed state when selecting a new spread
    state.payoffTableExpanded = false;

    // Show simulator with loading state
    elements.spreadSimulator.style.display = 'block';
    elements.simLoadingState.style.display = 'flex';

    // Update spread info display
    elements.simSymbol.textContent = spread.symbol;
    elements.simType.textContent = spread.spread_type === 'PCS' ? 'Put Credit Spread (Bullish)' : 'Call Credit Spread (Bearish)';
    elements.simExpiration.textContent = spread.expiration;
    elements.simShortStrike.textContent = formatCurrency(spread.short_strike, 0);
    elements.simLongStrike.textContent = formatCurrency(spread.long_strike, 0);
    elements.simNetCredit.textContent = formatCurrency(spread.credit, 2);

    try {
        const response = await fetch('/api/credit-spreads/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: spread.symbol,
                spread_type: spread.spread_type,
                expiration: spread.expiration,
                short_strike: spread.short_strike,
                long_strike: spread.long_strike,
                net_credit: spread.credit,
                underlying_price_now: state.underlyingPrice,
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to simulate spread');
        }

        const data = await response.json();
        state.simulatorData = data;

        // Update action summary - compact two-line format
        const breakevenFormatted = formatCurrency(data.summary.breakeven_price, 2);
        const shortStrikeFormatted = formatCurrency(spread.short_strike, 0);
        const longStrikeFormatted = formatCurrency(spread.long_strike, 0);
        const optionType = spread.spread_type === 'PCS' ? 'puts' : 'calls';

        if (elements.simProfitRange) {
            if (spread.spread_type === 'PCS') {
                elements.simProfitRange.textContent = `Above ${breakevenFormatted}`;
            } else {
                elements.simProfitRange.textContent = `Below ${breakevenFormatted}`;
            }
        }
        if (elements.simLegsText) {
            elements.simLegsText.textContent = `Sell ${shortStrikeFormatted} / Buy ${longStrikeFormatted} ${optionType}`;
        }

        // Update summary
        elements.simMaxGain.textContent = '+' + formatCurrency(data.summary.max_gain, 0);
        elements.simMaxLoss.textContent = '-' + formatCurrency(data.summary.max_loss, 0);
        elements.simBreakeven.textContent = formatCurrency(data.summary.breakeven_price, 2);
        elements.simBreakevenPct.textContent = (data.summary.breakeven_pct >= 0 ? '+' : '') + formatNumber(data.summary.breakeven_pct, 1) + '% from current';

        // Render table
        renderSimulatorTable(data.points);

        // Update AI Explainer button state
        if (aiExplainerController) {
            aiExplainerController.render();
        }

        // Scroll to simulator
        elements.spreadSimulator.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        console.error('Error simulating spread:', err);
        showError('Failed to simulate spread: ' + err.message);
        closeSimulator();
    } finally {
        elements.simLoadingState.style.display = 'none';
    }
}

// Render the P/L table
function renderSimulatorTable(points) {
    if (!elements.simTableBody) return;

    if (!points || points.length === 0) {
        elements.simTableBody.innerHTML =
            '<tr><td colspan="3" class="text-center">No data</td></tr>';
        if (elements.togglePayoffTable) {
            elements.togglePayoffTable.style.display = 'none';
        }
        return;
    }

    // Key moves to show in collapsed state: -5%, 0%, +5%
    const keyMoves = [-5, 0, 5];

    // Filter points for collapsed view
    let displayPoints = points;
    if (!state.payoffTableExpanded) {
        displayPoints = points.filter(p => keyMoves.includes(p.pct_move));
        // Fallback: if no key points found, show first 3 points
        if (displayPoints.length === 0) {
            displayPoints = points.slice(0, 3);
        }
    }

    elements.simTableBody.innerHTML = displayPoints.map(point => {
        const plClass = point.pl_per_spread >= 0 ? 'positive' : 'negative';
        const plPrefix = point.pl_per_spread >= 0 ? '+' : '';
        const pctPrefix = point.pct_move >= 0 ? '+' : '';

        return `
            <tr>
                <td class="col-pct-move">${pctPrefix}${point.pct_move}%</td>
                <td class="col-price">${formatCurrency(point.underlying_price, 2)}</td>
                <td class="col-pl ${plClass}">${plPrefix}${formatCurrency(point.pl_per_spread, 0)}</td>
            </tr>
        `;
    }).join('');

    // Show/hide toggle button and update its text
    if (elements.togglePayoffTable) {
        if (points.length > displayPoints.length || state.payoffTableExpanded) {
            elements.togglePayoffTable.style.display = 'block';
            const icon = elements.togglePayoffTable.querySelector('.toggle-icon');
            const text = elements.togglePayoffTable.querySelector('.toggle-text');
            if (state.payoffTableExpanded) {
                if (icon) icon.innerHTML = '&#9650;';  // Up arrow
                if (text) text.textContent = 'Show less';
            } else {
                if (icon) icon.innerHTML = '&#9660;';  // Down arrow
                if (text) text.textContent = 'Show full P/L table';
            }
        } else {
            elements.togglePayoffTable.style.display = 'none';
        }
    }
}

/**
 * Toggle P/L table between collapsed (3 key rows) and expanded (all rows) state.
 */
function togglePayoffTableExpand() {
    state.payoffTableExpanded = !state.payoffTableExpanded;

    // Re-render the table with current simulator data
    if (state.simulatorData && Array.isArray(state.simulatorData.points)) {
        renderSimulatorTable(state.simulatorData.points);
    }
}
