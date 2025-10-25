#!/usr/bin/env python3
"""
OpenAI Analyzer Module

Connects to OpenAI API to analyze stock indicator data and provide insights.
Integrates with stock_analyzer to download data, calculate indicators, and provide AI analysis.

Features:
- Loads API key and settings from config file
- Configurable model, temperature, and max_tokens from config
- Uses a predefined prompt template file
- Complete workflow: download stock data → calculate indicators → AI analysis
- Returns AI-generated insights and recommendations

OpenAI Config file format (openai_config.yaml):
    openai_api_key: your_api_key_here
    model: gpt-4o-mini    # Optional, defaults to gpt-4
    temperature: 0.7      # Optional, defaults to 0.7

    # Use max_completion_tokens for newer models (gpt-4o, gpt-4o-mini)
    max_completion_tokens: 2000  # Optional, defaults to 1500

    # Or use max_tokens for older models (gpt-4, gpt-3.5-turbo)
    # max_tokens: 1500

Finviz Auth file format (finviz_auth.yaml):
    auth_token: your_finviz_elite_token_here

OpenAI Prompt file format (openai_prompts/stock_analysis.txt):
    Plain text file containing the analysis prompt template.
    The stock data will be appended to this prompt.

Usage:
    from openai_analyzer import OpenAIAnalyzer

    # Initialize with OpenAI config and prompt files
    analyzer = OpenAIAnalyzer(
        config_file='openai_config.yaml',
        prompt_file='openai_prompts/stock_analysis.txt'
    )

    # Complete analysis: download data, calculate indicators, and get AI insights
    results = analyzer.analyze_stock(
        ticker='AAPL',
        finviz_auth_file='finviz_auth.yaml',
        days=365
    )

    print(results['ai_analysis'])
    df = results['data']  # DataFrame with indicators

    # Or analyze pre-calculated indicators
    analysis = analyzer.analyze_stock_indicators(
        ticker='AAPL',
        latest_data={'close': 150.0, 'rsi_14': 65.5, ...}
    )
"""

import os
import yaml
from typing import Dict, Any, Optional

class OpenAIAnalyzer:
    """
    Connects to OpenAI API to analyze stock technical indicators.
    Integrates with StockAnalyzer for complete end-to-end analysis.

    Attributes:
        api_key (str): OpenAI API key
        prompts (dict): Dictionary of prompt templates
        model (str): OpenAI model to use (default: gpt-4)
        temperature (float): Sampling temperature for API calls (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 1500)

    Methods:
        analyze_stock(): Complete workflow - download data, calculate indicators, AI analysis
        analyze_stock_indicators(): Analyze pre-calculated indicator data with AI
    """

    def __init__(
        self,
        config_file: str,
        prompt_file: str,
        model: Optional[str] = None
    ):
        """
        Initialize OpenAI Analyzer.

        Args:
            config_file: Path to YAML config file with API key and settings (required)
            prompt_file: Path to prompt template file (required)
            model: OpenAI model to use (optional, reads from config if not provided)

        Raises:
            FileNotFoundError: If config file or prompt file not found
            KeyError: If required keys missing in config
        """
        self.config_file = config_file
        self.prompt_file = prompt_file

        # Load configuration
        config = self._load_config()
        self.api_key = config['openai_api_key']
        self.model = model if model is not None else config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)

        # Support both old and new parameter names
        # Newer models (gpt-4o, gpt-4o-mini, etc.) use max_completion_tokens
        # Older models use max_tokens
        self.max_tokens = config.get('max_tokens', config.get('max_completion_tokens', 1500))
        self.max_completion_tokens = config.get('max_completion_tokens', config.get('max_tokens', 1500))

        print(f"✓ Config loaded: model={self.model}, temperature={self.temperature}, max_tokens={self.max_completion_tokens}")

        # Load and validate prompt file
        self.prompt_template = self._load_prompt()

        # Initialize OpenAI client
        self._init_openai_client()

    def _load_config(self) -> dict:
        """
        Load configuration from config file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            KeyError: If 'openai_api_key' not in config
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Config file '{self.config_file}' not found. "
                f"Create it with:\n"
                f"  openai_api_key: your_api_key_here\n"
                f"  model: gpt-4\n"
                f"  temperature: 0.7\n"
                f"  max_tokens: 1500"
            )

        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        if 'openai_api_key' not in config:
            raise KeyError(
                f"'openai_api_key' not found in {self.config_file}. "
                f"Add it to the config file."
            )

        return config

    def _load_prompt(self) -> str:
        """
        Load the prompt template from the prompt file.

        Returns:
            Prompt text string

        Raises:
            FileNotFoundError: If prompt file not found
        """
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(
                f"Prompt file '{self.prompt_file}' not found. "
                f"Please ensure the prompt file exists."
            )

        with open(self.prompt_file, 'r') as f:
            prompt_text = f.read()

        print(f"✓ Prompt template loaded from: {self.prompt_file}")

        return prompt_text

    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print("✓ OpenAI client initialized")
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with:\n"
                "  pip install openai"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")

    def format_indicator_data(
        self,
        ticker: str,
        latest_data: Dict[str, Any],
        historical_summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format indicator data into a readable string for the AI.

        Args:
            ticker: Stock ticker symbol
            latest_data: Dictionary with latest indicator values
            historical_summary: Optional historical statistics

        Returns:
            Formatted string with indicator data
        """
        output = [f"Stock Analysis for {ticker}"]
        output.append("=" * 50)

        # Latest values
        output.append("\nLatest Indicator Values:")
        output.append("-" * 50)

        if 'date' in latest_data:
            output.append(f"Date: {latest_data['date']}")
        if 'close' in latest_data:
            output.append(f"Close Price: ${latest_data['close']:.2f}")
        if 'volume' in latest_data:
            output.append(f"Volume: {latest_data['volume']:,.0f}")

        output.append("\nMomentum Indicators:")
        if 'rsi_14' in latest_data and latest_data['rsi_14'] is not None:
            output.append(f"  RSI (14): {latest_data['rsi_14']:.2f}")
        if 'mfi_14' in latest_data and latest_data['mfi_14'] is not None:
            output.append(f"  MFI (14): {latest_data['mfi_14']:.2f}")

        output.append("\nTrend Indicators:")
        if 'macd' in latest_data and latest_data['macd'] is not None:
            output.append(f"  MACD: {latest_data['macd']:.4f}")
        if 'macd_signal' in latest_data and latest_data['macd_signal'] is not None:
            output.append(f"  Signal: {latest_data['macd_signal']:.4f}")
        if 'macd_histogram' in latest_data and latest_data['macd_histogram'] is not None:
            output.append(f"  Histogram: {latest_data['macd_histogram']:.4f}")

        output.append("\nMoving Averages:")
        if 'ma20' in latest_data and latest_data['ma20'] is not None:
            output.append(f"  MA20: ${latest_data['ma20']:.2f}")
        if 'ma50' in latest_data and latest_data['ma50'] is not None:
            output.append(f"  MA50: ${latest_data['ma50']:.2f}")
        if 'ma200' in latest_data and latest_data['ma200'] is not None:
            output.append(f"  MA200: ${latest_data['ma200']:.2f}")

        # Historical summary if provided
        if historical_summary:
            output.append("\nHistorical Summary:")
            output.append("-" * 50)
            for key, value in historical_summary.items():
                output.append(f"  {key}: {value}")

        return "\n".join(output)

    def analyze_stock_indicators(
        self,
        ticker: str,
        latest_data: Dict[str, Any],
        historical_summary: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Send stock indicator data to OpenAI for analysis.

        Args:
            ticker: Stock ticker symbol
            latest_data: Dictionary with latest indicator values
            historical_summary: Optional historical statistics

        Returns:
            AI-generated analysis text, or None if failed
        """
        print(f"\nAnalyzing {ticker} with OpenAI ({self.model})...")

        # Format the data
        data_text = self.format_indicator_data(ticker, latest_data, historical_summary)

        # Build the full prompt using the predefined template
        full_prompt = f"{self.prompt_template}\n\n{data_text}"

        try:
            # Call OpenAI API
            # Use max_completion_tokens for newer models (gpt-4o, gpt-4o-mini, etc.)
            # Use max_tokens for older models (gpt-4, gpt-3.5-turbo, etc.)
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional stock market analyst with expertise in technical analysis."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "temperature": self.temperature
            }

            # Determine which token parameter to use based on model
            if 'gpt-4o' in self.model.lower() or 'gpt-5' in self.model.lower():
                # Newer models use max_completion_tokens
                api_params['max_completion_tokens'] = self.max_completion_tokens
            else:
                # Older models use max_tokens
                api_params['max_tokens'] = self.max_tokens

            response = self.client.chat.completions.create(**api_params)

            # Extract the analysis
            analysis = response.choices[0].message.content

            print("✓ Analysis received from OpenAI")

            return analysis

        except Exception as e:
            print(f"✗ Error calling OpenAI API: {e}")
            return None

    def analyze_stock(
        self,
        ticker: str,
        finviz_auth_file: str,
        days: int = 365,
        rsi_period: int = 14,
        mfi_period: int = 14,
        ma_periods: Optional[list] = None,
        save_analysis: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Complete stock analysis: download data, calculate indicators, and analyze with AI.

        This method uses analyze_stock_to_dataframe to get stock data and indicators,
        then sends the results to OpenAI for analysis.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            finviz_auth_file: Path to Finviz Elite authentication file
            days: Number of days of historical data (default: 365)
            rsi_period: Period for RSI calculation (default: 14)
            mfi_period: Period for MFI calculation (default: 14)
            ma_periods: Periods for MA calculation (default: [20, 50, 200])
            save_analysis: Whether to save AI analysis to file (default: True)

        Returns:
            Dictionary containing:
                - 'ticker': Stock ticker symbol
                - 'data': Full DataFrame with indicators
                - 'latest': Latest indicator values
                - 'ai_analysis': AI-generated analysis text
                - 'ai_analysis_file': Path to saved analysis file (if save_analysis=True)
            Returns None if analysis fails

        Example:
            >>> analyzer = OpenAIAnalyzer('openai_config.yaml', 'prompts/stock_analysis.txt')
            >>> results = analyzer.analyze_stock('AAPL', 'finviz_auth.yaml', days=365)
            >>> print(results['ai_analysis'])
        """
        try:
            from stock_analyzer import StockAnalyzer

            print(f"\n{'='*70}")
            print(f"Complete Stock Analysis: {ticker}")
            print(f"{'='*70}")

            # Step 1: Get stock data and calculate indicators
            stock_analyzer = StockAnalyzer(finviz_auth_file)
            stock_results = stock_analyzer.analyze_stock_to_dataframe(
                ticker,
                days=days,
                rsi_period=rsi_period,
                mfi_period=mfi_period,
                ma_periods=ma_periods
            )

            if stock_results is None:
                print(f"✗ Failed to analyze stock data for {ticker}")
                return None

            # Step 2: Analyze with OpenAI
            print(f"\n{'='*70}")
            print("AI Analysis (OpenAI)")
            print(f"{'='*70}")

            ai_analysis = self.analyze_stock_indicators(
                ticker=ticker,
                latest_data=stock_results['latest']
            )

            if ai_analysis is None:
                print("✗ AI analysis failed")
                return None

            # Step 3: Prepare results
            results = {
                'ticker': ticker,
                'data': stock_results['data'],
                'latest': stock_results['latest'],
                'ai_analysis': ai_analysis
            }

            # Step 4: Optionally save analysis to file
            if save_analysis:
                analysis_file = self.save_analysis(ai_analysis, ticker)
                results['ai_analysis_file'] = analysis_file

            # Display preview
            print(f"\n{'='*70}")
            print("AI Analysis Preview:")
            print(f"{'='*70}")
            preview = ai_analysis[:500] + "..." if len(ai_analysis) > 500 else ai_analysis
            print(preview)
            print(f"\n{'='*70}")

            return results

        except ImportError as e:
            print(f"✗ Error importing stock_analyzer: {e}")
            print("  Make sure stock_analyzer.py is in the Python path")
            return None
        except Exception as e:
            print(f"✗ Error in stock analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_analysis(
        self,
        analysis: str,
        ticker: str,
        output_file: Optional[str] = None
    ) -> str:
        """
        Save AI analysis to a text file.

        Args:
            analysis: AI-generated analysis text
            ticker: Stock ticker symbol
            output_file: Output filename (default: {ticker}_analysis.txt)

        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = f"{ticker}_analysis.txt"

        with open(output_file, 'w') as f:
            f.write(f"AI Analysis for {ticker}\n")
            f.write("=" * 70 + "\n\n")
            f.write(analysis)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write(f"Generated using {self.model}\n")

        print(f"✓ Analysis saved to {output_file}")

        return output_file


def quick_analyze_with_ai(
    ticker: str,
    finviz_auth_file: str,
    openai_config_file: str,
    openai_prompt_file: str,
    days: int = 365,
    save_to_file: bool = True
) -> Optional[Dict[str, Any]]:
    """
        Convenience function for complete stock analysis with AI.

        Downloads stock data, calculates indicators, and performs AI analysis.

        Args:
            ticker: Stock ticker symbol
            finviz_auth_file: Path to Finviz Elite authentication file (required)
            openai_config_file: Path to OpenAI config file (required)
            openai_prompt_file: Path to OpenAI prompt template file (required)
            days: Number of days of historical data (default: 365)
            save_to_file: Whether to save analysis to file

        Returns:
            Dictionary with analysis results, or None if failed

        Example:
            >>> results = quick_analyze_with_ai(
            ...     'AAPL',
            ...     'finviz_auth.yaml',
            ...     'openai_config.yaml',
            ...     'openai_prompts/stock_analysis.txt',
            ...     days=365
            ... )
            >>> print(results['ai_analysis'])
            >>> df = results['data']  # DataFrame with indicators
    """
    try:
        analyzer = OpenAIAnalyzer(openai_config_file, openai_prompt_file)
        results = analyzer.analyze_stock(
            ticker=ticker,
            finviz_auth_file=finviz_auth_file,
            days=days,
            save_analysis=save_to_file
        )

        return results

    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return None


# Example usage
if __name__ == '__main__':
    """
    Command-line interface for OpenAI Stock Analyzer.

    Complete stock analysis workflow with AI insights.
    """
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Analyze stocks with technical indicators and AI insights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            Examples:
            %(prog)s AAPL                                      # Analyze AAPL with defaults
            %(prog)s AAPL --days 180                           # Analyze last 180 days
            %(prog)s TSLA --finviz-auth custom_finviz.yaml     # Use custom Finviz auth
            %(prog)s QQQ --days 90 --openai-config my_config.yaml

            Required Files:
            - finviz_auth.yaml: Finviz Elite authentication token
            - openai_config.yaml: OpenAI API configuration
            - openai_prompts/stock_analysis.txt: OpenAI analysis prompt template
        '''
    )

    parser.add_argument(
        'ticker',
        help='Stock ticker symbol (e.g., AAPL, TSLA, QQQ)'
    )

    parser.add_argument(
        '--finviz-auth',
        dest='finviz_auth_file',
        default='finviz_auth.yaml',
        help='Path to Finviz Elite authentication file (default: finviz_auth.yaml)'
    )

    parser.add_argument(
        '--openai-config',
        dest='openai_config_file',
        default='openai_config.yaml',
        help='Path to OpenAI config file (default: openai_config.yaml)'
    )

    parser.add_argument(
        '--openai-prompt',
        dest='openai_prompt_file',
        default='openai_prompts/stock_analysis.txt',
        help='Path to OpenAI prompt template file (default: openai_prompts/stock_analysis.txt)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data (default: 365)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save AI analysis to file'
    )

    parser.add_argument(
        '--rsi-period',
        type=int,
        default=14,
        help='RSI calculation period (default: 14)'
    )

    parser.add_argument(
        '--mfi-period',
        type=int,
        default=14,
        help='MFI calculation period (default: 14)'
    )

    args = parser.parse_args()

    print("OpenAI Stock Analyzer")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.days} days")
    print(f"Finviz auth: {args.finviz_auth_file}")
    print(f"OpenAI config: {args.openai_config_file}")
    print(f"OpenAI prompt: {args.openai_prompt_file}")
    print("=" * 70)

    try:
        # Complete analysis: download data, calculate indicators, AI analysis
        results = quick_analyze_with_ai(
            ticker=args.ticker,
            finviz_auth_file=args.finviz_auth_file,
            openai_config_file=args.openai_config_file,
            openai_prompt_file=args.openai_prompt_file,
            days=args.days,
            save_to_file=not args.no_save
        )

        if results:
            print("\n" + "=" * 70)
            print("Analysis Complete!")
            print("=" * 70)
            print(f"\nTicker: {results['ticker']}")
            print(f"Data records: {len(results['data'])}")
            print(f"\nLatest indicators:")
            for key, value in list(results['latest'].items())[:5]:
                print(f"  {key}: {value}")

            if 'ai_analysis_file' in results:
                print(f"\nAI analysis saved to: {results['ai_analysis_file']}")
        else:
            print("\n✗ Analysis failed")

    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nTo use this module:")
        print("1. Create finviz_auth.yaml with your Finviz Elite token")
        print("2. Create openai_config.yaml with your OpenAI API key and settings")
        print("3. Create openai_prompts/stock_analysis.txt with your analysis prompt template")
        print("4. Run this script again with --help for usage information")
