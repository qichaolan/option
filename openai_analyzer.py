#!/usr/bin/env python3
"""
OpenAI Analyzer Module

Connects to OpenAI API to analyze stock indicator data and provide insights.

Features:
- Loads API key from config file
- Loads prompts from prompts file
- Sends indicator data to OpenAI for analysis
- Returns AI-generated insights and recommendations

Usage:
    from openai_analyzer import OpenAIAnalyzer

    # Initialize
    analyzer = OpenAIAnalyzer('openai_config.yaml', 'prompts.yaml')

    # Analyze stock data
    analysis = analyzer.analyze_stock_indicators(
        ticker='AAPL',
        latest_data={'close': 150.0, 'rsi_14': 65.5, ...}
    )

    print(analysis)
"""

import os
import yaml
from typing import Dict, Any, Optional
import json


class OpenAIAnalyzer:
    """
    Connects to OpenAI API to analyze stock technical indicators.

    Attributes:
        api_key (str): OpenAI API key
        prompts (dict): Dictionary of prompt templates
        model (str): OpenAI model to use (default: gpt-4)
    """

    def __init__(
        self,
        config_file: str = 'openai_config.yaml',
        prompts_file: str = 'prompts.yaml',
        model: str = 'gpt-4'
    ):
        """
        Initialize OpenAI Analyzer.

        Args:
            config_file: Path to YAML config file with API key
            prompts_file: Path to YAML file with prompt templates
            model: OpenAI model to use (default: gpt-4)

        Raises:
            FileNotFoundError: If config or prompts file not found
            KeyError: If required keys missing in config
        """
        self.config_file = config_file
        self.prompts_file = prompts_file
        self.model = model

        # Load configuration
        self.api_key = self._load_api_key()
        self.prompts = self._load_prompts()

        # Initialize OpenAI client
        self._init_openai_client()

    def _load_api_key(self) -> str:
        """
        Load OpenAI API key from config file.

        Returns:
            API key string

        Raises:
            FileNotFoundError: If config file not found
            KeyError: If 'openai_api_key' not in config
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Config file '{self.config_file}' not found. "
                f"Create it with:\n"
                f"  openai_api_key: your_api_key_here"
            )

        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        if 'openai_api_key' not in config:
            raise KeyError(
                f"'openai_api_key' not found in {self.config_file}. "
                f"Add it to the config file."
            )

        return config['openai_api_key']

    def _load_prompts(self) -> Dict[str, str]:
        """
        Load prompt templates from file.

        Returns:
            Dictionary of prompt templates

        Raises:
            FileNotFoundError: If prompts file not found
        """
        if not os.path.exists(self.prompts_file):
            raise FileNotFoundError(
                f"Prompts file '{self.prompts_file}' not found. "
                f"Create it with your analysis prompts."
            )

        with open(self.prompts_file, 'r') as f:
            prompts = yaml.safe_load(f)

        return prompts

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
        historical_summary: Optional[Dict[str, Any]] = None,
        prompt_type: str = 'default_analysis'
    ) -> Optional[str]:
        """
        Send stock indicator data to OpenAI for analysis.

        Args:
            ticker: Stock ticker symbol
            latest_data: Dictionary with latest indicator values
            historical_summary: Optional historical statistics
            prompt_type: Type of analysis prompt to use (default: 'default_analysis')

        Returns:
            AI-generated analysis text, or None if failed
        """
        print(f"\nAnalyzing {ticker} with OpenAI ({self.model})...")

        # Format the data
        data_text = self.format_indicator_data(ticker, latest_data, historical_summary)

        # Get the prompt template
        if prompt_type not in self.prompts:
            print(f"Warning: Prompt type '{prompt_type}' not found. Using 'default_analysis'.")
            prompt_type = 'default_analysis'

        prompt_template = self.prompts.get(prompt_type, '')

        # Build the full prompt
        full_prompt = f"{prompt_template}\n\n{data_text}"

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional stock market analyst with expertise in technical analysis."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )

            # Extract the analysis
            analysis = response.choices[0].message.content

            print("✓ Analysis received from OpenAI")

            return analysis

        except Exception as e:
            print(f"✗ Error calling OpenAI API: {e}")
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
    latest_data: Dict[str, Any],
    config_file: str = 'openai_config.yaml',
    prompts_file: str = 'prompts.yaml',
    save_to_file: bool = True
) -> Optional[str]:
    """
    Convenience function for quick AI analysis.

    Args:
        ticker: Stock ticker symbol
        latest_data: Dictionary with latest indicator values
        config_file: Path to OpenAI config file
        prompts_file: Path to prompts file
        save_to_file: Whether to save analysis to file

    Returns:
        AI-generated analysis text, or None if failed

    Example:
        >>> latest = {
        ...     'close': 150.0,
        ...     'rsi_14': 65.5,
        ...     'mfi_14': 58.2,
        ...     'macd': 2.1,
        ...     'ma20': 148.5
        ... }
        >>> analysis = quick_analyze_with_ai('AAPL', latest)
        >>> print(analysis)
    """
    try:
        analyzer = OpenAIAnalyzer(config_file, prompts_file)
        analysis = analyzer.analyze_stock_indicators(ticker, latest_data)

        if analysis and save_to_file:
            analyzer.save_analysis(analysis, ticker)

        return analysis

    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return None


# Example usage
if __name__ == '__main__':
    """
    Example usage of OpenAI Analyzer.

    Note: Requires openai_config.yaml and prompts.yaml files.
    """
    print("OpenAI Stock Analyzer - Example Usage")
    print("=" * 70)

    # Example indicator data
    example_data = {
        'date': '2024-10-24',
        'close': 150.25,
        'volume': 50000000,
        'rsi_14': 65.32,
        'mfi_14': 58.45,
        'macd': 2.1234,
        'macd_signal': 1.9876,
        'macd_histogram': 0.1358,
        'ma20': 148.50,
        'ma50': 145.20,
        'ma200': 140.80
    }

    try:
        # Try to analyze
        analysis = quick_analyze_with_ai('AAPL', example_data)

        if analysis:
            print("\n" + "=" * 70)
            print("AI Analysis:")
            print("=" * 70)
            print(analysis)
        else:
            print("\n✗ Analysis failed")

    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nTo use this module:")
        print("1. Create openai_config.yaml with your API key")
        print("2. Create prompts.yaml with your analysis prompts")
        print("3. Run this script again")
