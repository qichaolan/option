/**
 * LEAPS Ranker - Frontend Application
 */

// State
let state = {
    tickers: [],
    contracts: [],
    selectedContract: null,
    loading: false,
    error: null,
};

// DOM Elements
const elements = {
    tickerSelect: document.getElementById('tickerSelect'),
    targetPctInput: document.getElementById('targetPct'),
    modeSelect: document.getElementById('modeSelect'),
    fetchBtn: document.getElementById('fetchBtn'),
    symbolDisplay: document.getElementById('symbolDisplay'),
    underlyingPrice: document.getElementById('underlyingPrice'),
    targetPrice: document.getElementById('targetPrice'),
    targetPctDisplay: document.getElementById('targetPctDisplay'),
    contractsBody: document.getElementById('contractsBody'),
    loadingState: document.getElementById('loadingState'),
    emptyState: document.getElementById('emptyState'),
    tableContainer: document.getElementById('tableContainer'),
    errorDisplay: document.getElementById('errorDisplay'),
    simStrike: document.getElementById('simStrike'),
    simPremium: document.getElementById('simPremium'),
    simUnderlying: document.getElementById('simUnderlying'),
    simCost: document.getElementById('simCost'),
    simBreakeven: document.getElementById('simBreakeven'),
    simTargetInput: document.getElementById('simTargetInput'),
    runSimBtn: document.getElementById('runSimBtn'),
    simResultsGrid: document.getElementById('simResultsGrid'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadTickers();
    setupEventListeners();
});

// Load supported tickers
async function loadTickers() {
    try {
        const response = await fetch('/api/tickers');
        if (!response.ok) throw new Error('Failed to load tickers');

        state.tickers = await response.json();

        // Populate ticker dropdown
        elements.tickerSelect.innerHTML = state.tickers.map(t =>
            `<option value="${t.symbol}" data-target="${t.default_target_pct}">
                ${t.symbol} - ${t.name}
            </option>`
        ).join('');

        // Set default target pct from first ticker
        if (state.tickers.length > 0) {
            elements.targetPctInput.value = (state.tickers[0].default_target_pct * 100).toFixed(0);
        }
    } catch (err) {
        console.error('Error loading tickers:', err);
        showError('Failed to load supported tickers');
    }
}

// Setup event listeners
function setupEventListeners() {
    // Ticker change - update default target
    elements.tickerSelect.addEventListener('change', () => {
        const selected = elements.tickerSelect.selectedOptions[0];
        if (selected) {
            const targetPct = parseFloat(selected.dataset.target);
            elements.targetPctInput.value = (targetPct * 100).toFixed(0);
        }
    });

    // Fetch button
    elements.fetchBtn.addEventListener('click', fetchLEAPS);

    // Run simulation button
    elements.runSimBtn.addEventListener('click', runSimulation);

    // Keyboard shortcut: Enter to fetch
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.target.matches('input, textarea')) {
            fetchLEAPS();
        }
    });
}

// Fetch LEAPS data
async function fetchLEAPS() {
    const symbol = elements.tickerSelect.value;
    const targetPct = parseFloat(elements.targetPctInput.value) / 100;
    const mode = elements.modeSelect.value;

    if (!symbol) {
        showError('Please select a ticker');
        return;
    }

    if (isNaN(targetPct) || targetPct <= 0 || targetPct > 2) {
        showError('Please enter a valid target percentage (1-200)');
        return;
    }

    setLoading(true);
    hideError();

    try {
        const response = await fetch('/api/leaps', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                target_pct: targetPct,
                mode,
                top_n: 20,
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to fetch LEAPS data');
        }

        const data = await response.json();
        updateUI(data);

    } catch (err) {
        console.error('Error fetching LEAPS:', err);
        showError(err.message);
    } finally {
        setLoading(false);
    }
}

// Update UI with data
function updateUI(data) {
    state.contracts = data.contracts;

    // Update info cards
    elements.symbolDisplay.textContent = data.symbol;
    elements.underlyingPrice.textContent = `$${data.underlying_price.toFixed(2)}`;
    elements.targetPrice.textContent = `$${data.target_price.toFixed(2)}`;
    elements.targetPctDisplay.textContent = `+${(data.target_pct * 100).toFixed(0)}%`;

    // Update table
    if (data.contracts.length === 0) {
        elements.emptyState.style.display = 'block';
        elements.tableContainer.style.display = 'none';
    } else {
        elements.emptyState.style.display = 'none';
        elements.tableContainer.style.display = 'block';
        renderTable(data.contracts);
    }

    // Select first contract for simulator
    if (data.contracts.length > 0) {
        selectContract(data.contracts[0], 0);
    }
}

// Render contracts table
function renderTable(contracts) {
    elements.contractsBody.innerHTML = contracts.map((c, idx) => `
        <tr data-contract="${idx}" class="${idx === 0 ? 'selected' : ''}">
            <td>${c.contract_symbol}</td>
            <td>${c.expiration}</td>
            <td>$${c.strike.toFixed(2)}</td>
            <td>$${c.premium.toFixed(2)}</td>
            <td>$${c.cost.toFixed(0)}</td>
            <td>$${c.payoff_target.toFixed(0)}</td>
            <td class="${c.roi_target >= 0 ? 'positive' : 'negative'}">${c.roi_target.toFixed(1)}%</td>
            <td>${c.ease_score.toFixed(2)}</td>
            <td>${c.roi_score.toFixed(2)}</td>
            <td><span class="score-badge ${getScoreClass(c.score)}">${c.score.toFixed(2)}</span></td>
            <td>${c.implied_volatility ? (c.implied_volatility * 100).toFixed(1) + '%' : '-'}</td>
            <td>${c.open_interest ? c.open_interest.toLocaleString() : '-'}</td>
        </tr>
    `).join('');

    // Add click handlers
    document.querySelectorAll('[data-contract]').forEach(row => {
        row.addEventListener('click', () => {
            const idx = parseInt(row.dataset.contract);
            selectContract(contracts[idx], idx);

            // Update selected class
            document.querySelectorAll('[data-contract]').forEach(r => r.classList.remove('selected'));
            row.classList.add('selected');
        });
    });
}

// Get score class for styling
function getScoreClass(score) {
    if (score >= 0.7) return 'score-high';
    if (score >= 0.4) return 'score-medium';
    return 'score-low';
}

// Select contract for simulator
function selectContract(contract, idx) {
    state.selectedContract = contract;

    // Update simulator inputs
    elements.simStrike.value = contract.strike.toFixed(2);
    elements.simPremium.value = contract.premium.toFixed(2);

    // Calculate underlying from target price
    const underlying = contract.target_price / (1 + parseFloat(elements.targetPctInput.value) / 100);
    elements.simUnderlying.value = underlying.toFixed(2);

    // Calculate cost and breakeven
    const cost = contract.premium * 100;
    const breakeven = contract.strike + contract.premium;

    elements.simCost.textContent = `$${cost.toFixed(0)}`;
    elements.simBreakeven.textContent = `$${breakeven.toFixed(2)}`;

    // Set default target prices for simulation
    const targets = [];
    const pctSteps = [-10, 0, 10, 25, 50, 75, 100];
    pctSteps.forEach(pct => {
        targets.push((underlying * (1 + pct / 100)).toFixed(2));
    });
    elements.simTargetInput.value = targets.join(', ');
}

// Run ROI simulation
async function runSimulation() {
    const strike = parseFloat(elements.simStrike.value);
    const premium = parseFloat(elements.simPremium.value);
    const underlying = parseFloat(elements.simUnderlying.value);
    const targetStr = elements.simTargetInput.value;

    if (isNaN(strike) || isNaN(premium) || isNaN(underlying)) {
        showError('Please enter valid strike, premium, and underlying price');
        return;
    }

    // Parse target prices
    const targets = targetStr.split(',')
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n) && n > 0);

    if (targets.length === 0) {
        showError('Please enter at least one target price');
        return;
    }

    try {
        const response = await fetch('/api/roi-simulator', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                strike,
                premium,
                underlying_price: underlying,
                target_prices: targets,
                contract_size: 100,
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Simulation failed');
        }

        const data = await response.json();
        renderSimResults(data);

    } catch (err) {
        console.error('Simulation error:', err);
        showError(err.message);
    }
}

// Render simulation results
function renderSimResults(data) {
    elements.simResultsGrid.innerHTML = data.results.map(r => `
        <div class="simulator-result">
            <div class="simulator-result-label">$${r.target_price.toFixed(2)} (${r.price_change_pct >= 0 ? '+' : ''}${r.price_change_pct.toFixed(1)}%)</div>
            <div class="simulator-result-value ${r.roi_pct >= 0 ? 'positive' : 'negative'}">
                ${r.roi_pct >= 0 ? '+' : ''}${r.roi_pct.toFixed(1)}%
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary);">
                P&L: ${r.profit >= 0 ? '+' : ''}$${r.profit.toFixed(0)}
            </div>
        </div>
    `).join('');
}

// Set loading state
function setLoading(loading) {
    state.loading = loading;
    elements.fetchBtn.disabled = loading;
    elements.loadingState.style.display = loading ? 'flex' : 'none';

    if (loading) {
        elements.tableContainer.style.display = 'none';
        elements.emptyState.style.display = 'none';
    }
}

// Show error message
function showError(message) {
    elements.errorDisplay.textContent = message;
    elements.errorDisplay.style.display = 'block';
}

// Hide error message
function hideError() {
    elements.errorDisplay.style.display = 'none';
}
