# OpenAI Integration for Stock Analysis

AI-powered analysis of stock technical indicators using OpenAI's GPT models.

## Overview

The OpenAI integration adds AI-powered insights to your stock analysis:
- Automatically analyzes technical indicators
- Provides trading recommendations
- Identifies trends and patterns
- Assesses risk levels
- Generates detailed reports

## Setup

### 1. Install OpenAI Library

```bash
pip install openai
```

### 2. Create OpenAI Config File

Copy the template and add your API key:

```bash
cp openai_config.yaml.template openai_config.yaml
```

Edit `openai_config.yaml`:
```yaml
openai_api_key: sk-your-actual-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Customize Prompts (Optional)

The `prompts.yaml` file contains different analysis prompts:
- `default_analysis` - Comprehensive analysis (default)
- `short_summary` - Brief 3-4 sentence summary
- `detailed_report` - Full technical report
- `risk_assessment` - Risk-focused analysis
- `entry_exit_signals` - Trading signals
- `trend_confirmation` - Trend validation
- `comparative_analysis` - Historical comparison

## Usage

### Command Line

**Basic AI analysis:**
```bash
python stock_analyzer.py AAPL --ai
```

**With custom prompt:**
```bash
python stock_analyzer.py AAPL --ai --ai-prompt short_summary
```

**Full example:**
```bash
python stock_analyzer.py AAPL \
  --days 365 \
  --auth finviz_auth.yaml \
  --output aapl_data.csv \
  --ai \
  --ai-config openai_config.yaml \
  --ai-prompt detailed_report
```

### Python API

```python
from stock_analyzer import StockAnalyzer

# Initialize
analyzer = StockAnalyzer('auth.yaml')

# Analyze with AI
results = analyzer.analyze_stock(
    stock_name='AAPL',
    days=365,
    use_ai=True,
    ai_prompt_type='default_analysis'
)

# Access AI analysis
if 'ai_analysis' in results:
    print(results['ai_analysis'])
    print(f"Saved to: {results['ai_analysis_file']}")
```

### Standalone OpenAI Analyzer

```python
from openai_analyzer import OpenAIAnalyzer

# Initialize
ai = OpenAIAnalyzer('openai_config.yaml', 'prompts.yaml')

# Analyze indicators
latest_data = {
    'date': '2024-10-24',
    'close': 150.25,
    'rsi_14': 65.32,
    'mfi_14': 58.45,
    'macd': 2.1234,
    'ma20': 148.50,
    'ma50': 145.20,
    'ma200': 140.80
}

analysis = ai.analyze_stock_indicators(
    ticker='AAPL',
    latest_data=latest_data,
    prompt_type='default_analysis'
)

# Save to file
ai.save_analysis(analysis, 'AAPL')
```

## Output

### Files Created

1. **{TICKER}_indicators.csv** - All stock data with calculated indicators
2. **{TICKER}_analysis.txt** - AI-generated analysis report

### AI Analysis Includes

- **Overall Market Sentiment** - Bullish, bearish, or neutral
- **Momentum Analysis** - RSI and MFI interpretation
- **Trend Analysis** - MACD and moving average signals
- **Key Observations** - Significant technical signals
- **Trading Recommendations** - Buy, sell, or hold with reasoning
- **Risk Assessment** - Risk factors and considerations
- **Support/Resistance** - Key price levels to watch

## Available Prompts

### default_analysis
Comprehensive analysis covering sentiment, momentum, trends, and recommendations.

**Use when:** You want a balanced, complete analysis.

### short_summary
Brief 3-4 sentence summary with simple recommendation.

**Use when:** You need a quick overview.

### detailed_report
Full technical report with executive summary, detailed sections, and specific trading signals.

**Use when:** You're making important trading decisions.

### risk_assessment
Focused on risk factors, stop-loss levels, and reward-to-risk ratio.

**Use when:** You want to understand the risks before entering.

### entry_exit_signals
Specific entry and exit points with confirmation signals.

**Use when:** You're looking for precise trading levels.

### trend_confirmation
Validates whether indicators confirm or contradict each other.

**Use when:** You want to verify signal reliability.

### comparative_analysis
Compares current readings to historical norms.

**Use when:** You want context for the current indicators.

## Example Output

### Command:
```bash
python stock_analyzer.py AAPL --ai
```

### Console Output:
```
Stock Analyzer Configuration:
  Ticker: AAPL
  Days: 365
  Auth file: auth.yaml
  Output file: AAPL_indicators.csv (auto-generated)
  AI Analysis: Enabled
  AI Config: openai_config.yaml
  AI Prompt Type: default_analysis

Analyzing AAPL...
Downloading AAPL data from Finviz...
Downloaded 250 records

Calculating technical indicators...
✓ RSI (14-period)
✓ MFI (14-period)
✓ MACD (12, 26, 9)
✓ Moving Averages [20, 50, 200]
✓ Analysis complete: 250 records saved to AAPL_indicators.csv

======================================================================
AI Analysis (OpenAI)
======================================================================
✓ OpenAI client initialized

Analyzing AAPL with OpenAI (gpt-4)...
✓ Analysis received from OpenAI
✓ Analysis saved to AAPL_analysis.txt

======================================================================
AI Analysis Preview:
======================================================================
Based on the technical indicators provided, here's my analysis:

1. OVERALL MARKET SENTIMENT: The stock shows a moderately bullish trend...
...

======================================================================

✓ Analysis completed successfully!
   Total records: 250
   Output file: AAPL_indicators.csv
   AI Analysis file: AAPL_analysis.txt
```

## Customizing Prompts

Edit `prompts.yaml` to add your own analysis prompts:

```yaml
my_custom_prompt: |
  Please analyze this stock with focus on:
  1. Short-term trading opportunities
  2. Volume analysis
  3. Price action patterns

  Provide specific entry points and time frames.
```

Use it:
```bash
python stock_analyzer.py AAPL --ai --ai-prompt my_custom_prompt
```

## Cost Considerations

**OpenAI API costs money!**

Typical costs per analysis:
- **GPT-4**: ~$0.03 - $0.10 per analysis
- **GPT-3.5-Turbo**: ~$0.001 - $0.005 per analysis

To use GPT-3.5-Turbo (cheaper):
```python
from openai_analyzer import OpenAIAnalyzer

ai = OpenAIAnalyzer(
    'openai_config.yaml',
    'prompts.yaml',
    model='gpt-3.5-turbo'  # Cheaper model
)
```

## Security

**IMPORTANT:** Never commit API keys!

- ✅ `openai_config.yaml` is in `.gitignore`
- ✅ `auth.yaml` is in `.gitignore`
- ✅ Use template files for reference
- ❌ Don't share your config files
- ❌ Don't commit files with API keys

## Troubleshooting

### "OpenAI library not installed"
```bash
pip install openai
```

### "Config file not found"
Create from template:
```bash
cp openai_config.yaml.template openai_config.yaml
# Edit and add your API key
```

### "Prompt type not found"
Check `prompts.yaml` for available prompt types. Default is `default_analysis`.

### "API key invalid"
Verify your API key at: https://platform.openai.com/api-keys

### "Rate limit exceeded"
Wait a moment and try again. Consider using GPT-3.5-Turbo for lower rate limits.

## Best Practices

1. **Start with short_summary** - Test with cheaper, faster responses
2. **Use detailed_report sparingly** - More expensive, use for important decisions
3. **Combine with your own analysis** - AI is a tool, not a replacement for research
4. **Track your costs** - Monitor API usage at OpenAI dashboard
5. **Review the analysis** - Don't blindly follow AI recommendations

## Limitations

- AI analysis is based only on technical indicators provided
- Does not include fundamental analysis, news, or market sentiment
- Past patterns don't guarantee future results
- Should be used as one input among many in trading decisions
- API costs can add up quickly

## Advanced Usage

### Batch Analysis

Analyze multiple stocks with AI:

```python
from stock_analyzer import StockAnalyzer

analyzer = StockAnalyzer('auth.yaml')
tickers = ['AAPL', 'GOOGL', 'MSFT']

for ticker in tickers:
    print(f"\n{'='*70}")
    print(f"Analyzing {ticker}")
    print('='*70)

    results = analyzer.analyze_stock(
        stock_name=ticker,
        days=365,
        use_ai=True,
        ai_prompt_type='short_summary'  # Use cheaper prompt for batch
    )

    if results and 'ai_analysis' in results:
        print(f"\n{results['ai_analysis']}")
```

### Compare Different Prompts

```python
prompts = ['short_summary', 'risk_assessment', 'entry_exit_signals']

for prompt_type in prompts:
    results = analyzer.analyze_stock(
        'AAPL',
        use_ai=True,
        ai_prompt_type=prompt_type
    )
    # Save with different names
```

## See Also

- `openai_analyzer.py` - Source code
- `prompts.yaml` - Prompt templates
- `openai_config.yaml.template` - Config template
- `INTEGRATION_README.md` - Stock analyzer integration guide
