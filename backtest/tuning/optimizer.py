"""
Grid search parameter optimizer for backtesting strategies.

This module provides the main optimization logic that:
1. Loads parameter configurations defining search ranges
2. Generates all combinations via Cartesian product
3. Runs backtests for each parameter combination
4. Ranks results by performance metrics
5. Returns top N performing parameter sets
"""

import argparse
import itertools
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from backtest.engine import run_backtest
from backtest.tuning.exceptions import TuningError
from backtest.tuning.param_config import ParamConfig, ParameterSpec, load_param_config
from backtest.tuning.yaml_utils import clone_and_modify, load_yaml, save_yaml


# Maximum search space size before warning
MAX_SEARCH_SPACE_SIZE = 100000


@dataclass
class ParameterResult:
    """Result for a single parameter combination."""

    parameters: Dict[str, float]
    strategy_final_value: float
    total_return_pct: float
    excess_vs_lumpsum: float
    excess_vs_dca: float
    num_trades: int
    num_buys: int
    num_sells: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameters": self.parameters,
            "strategy_final_value": self.strategy_final_value,
            "total_return_pct": self.total_return_pct,
            "excess_vs_lumpsum": self.excess_vs_lumpsum,
            "excess_vs_dca": self.excess_vs_dca,
            "num_trades": self.num_trades,
            "num_buys": self.num_buys,
            "num_sells": self.num_sells,
        }


@dataclass
class SearchResult:
    """Result of parameter search optimization."""

    top_results: List[ParameterResult]
    total_combinations: int
    evaluated_combinations: int
    best_parameters: Dict[str, float]
    best_final_value: float
    search_space: Dict[str, List[float]]

    def summary(self) -> str:
        """Generate human-readable summary of search results."""
        lines = [
            "=" * 70,
            "PARAMETER SEARCH RESULTS",
            "=" * 70,
            "",
            f"Total combinations evaluated: {self.evaluated_combinations}/{self.total_combinations}",
            "",
            "SEARCH SPACE:",
        ]

        for param_name, values in self.search_space.items():
            lines.append(f"  {param_name}: {values[0]} to {values[-1]} ({len(values)} values)")

        lines.extend([
            "",
            "=" * 70,
            "TOP PERFORMING PARAMETER SETS",
            "=" * 70,
            "",
        ])

        for i, result in enumerate(self.top_results, 1):
            lines.append(f"Rank #{i}")
            lines.append("-" * 40)
            lines.append("Parameters:")
            for name, value in result.parameters.items():
                lines.append(f"  {name}: {value}")
            lines.append(f"Final Value: ${result.strategy_final_value:,.2f}")
            lines.append(f"Total Return: {result.total_return_pct:.2f}%")
            lines.append(f"Excess vs Lump Sum: {result.excess_vs_lumpsum:.2f}%")
            lines.append(f"Excess vs DCA: {result.excess_vs_dca:.2f}%")
            lines.append(f"Trades: {result.num_trades} (Buys: {result.num_buys}, Sells: {result.num_sells})")
            lines.append("")

        lines.extend([
            "=" * 70,
            "BEST PARAMETERS",
            "=" * 70,
        ])
        for name, value in self.best_parameters.items():
            lines.append(f"  {name}: {value}")
        lines.append(f"  -> Final Value: ${self.best_final_value:,.2f}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "top_results": [r.to_dict() for r in self.top_results],
            "total_combinations": self.total_combinations,
            "evaluated_combinations": self.evaluated_combinations,
            "best_parameters": self.best_parameters,
            "best_final_value": self.best_final_value,
            "search_space": self.search_space,
        }


def generate_parameter_combinations(
    param_config: ParamConfig,
) -> List[Dict[str, float]]:
    """
    Generate all parameter combinations from config.

    Uses Cartesian product of all parameter value ranges.

    Args:
        param_config: Parameter configuration with specs.

    Returns:
        List of dictionaries mapping parameter names to values.
    """
    if not param_config.parameters:
        return [{}]

    # Get names and value lists
    names = [p.name for p in param_config.parameters]
    value_lists = [p.get_values() for p in param_config.parameters]

    # Generate Cartesian product
    combinations = []
    for values in itertools.product(*value_lists):
        combo = dict(zip(names, values))
        combinations.append(combo)

    return combinations


def build_path_mapping(
    param_config: ParamConfig,
    parameter_values: Dict[str, float],
) -> Dict[str, float]:
    """
    Build path-to-value mapping for YAML modification.

    Args:
        param_config: Parameter configuration with path definitions.
        parameter_values: Dictionary of parameter name to value.

    Returns:
        Dictionary mapping YAML paths to values.
    """
    path_mapping = {}
    for spec in param_config.parameters:
        if spec.name in parameter_values:
            path_mapping[spec.path] = parameter_values[spec.name]
    return path_mapping


def run_single_backtest(
    data: Union[str, pd.DataFrame],
    strategy_yaml: Dict[str, Any],
    initial_capital: float,
    temp_dir: Optional[Path] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run a single backtest with modified strategy.

    Args:
        data: Path to indicator data CSV or DataFrame directly.
        strategy_yaml: Modified strategy YAML dictionary.
        initial_capital: Initial capital for backtest.
        temp_dir: Optional temp directory for strategy file.

    Returns:
        Tuple of (final_value, metrics_dict).
    """
    import tempfile

    # Create temp strategy file
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    temp_strategy = temp_dir / "temp_strategy.yaml"
    save_yaml(strategy_yaml, str(temp_strategy))

    try:
        # Run backtest - accepts DataFrame or path
        result = run_backtest(
            data_file=data,
            strategy_files=str(temp_strategy),
            initial_capital=initial_capital,
        )

        # Extract metrics
        portfolio = result.portfolio
        final_value = portfolio.final_value

        # Get benchmark comparison
        comparison = result.comparison
        strategy_row = comparison[comparison["Strategy"] == "Trading Strategy"]

        if not strategy_row.empty:
            total_return = strategy_row["Return %"].values[0]
        else:
            total_return = ((final_value - initial_capital) / initial_capital) * 100

        # Calculate excess returns vs benchmarks
        lumpsum_row = comparison[comparison["Strategy"] == "Lump-Sum Buy-and-Hold"]
        dca_row = comparison[comparison["Strategy"] == "Monthly DCA"]

        if not lumpsum_row.empty:
            lumpsum_return = lumpsum_row["Return %"].values[0]
            excess_vs_lumpsum = total_return - lumpsum_return
        else:
            excess_vs_lumpsum = 0.0

        if not dca_row.empty:
            dca_return = dca_row["Return %"].values[0]
            excess_vs_dca = total_return - dca_return
        else:
            excess_vs_dca = 0.0

        metrics = {
            "strategy_final_value": final_value,
            "total_return_pct": total_return,
            "excess_vs_lumpsum": excess_vs_lumpsum,
            "excess_vs_dca": excess_vs_dca,
            "num_trades": portfolio.num_trades,
            "num_buys": portfolio.num_buys,
            "num_sells": portfolio.num_sells,
        }

        return final_value, metrics

    finally:
        # Clean up temp file
        if temp_strategy.exists():
            temp_strategy.unlink()


def _validate_search_inputs(
    data_path: Optional[str],
    strategy_template_path: str,
    param_config_path: str,
    initial_capital: float,
    top_n: int,
) -> None:
    """Validate input parameters for parameter search."""
    # data_path is None when DataFrame is passed directly
    if data_path is not None and not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not Path(strategy_template_path).exists():
        raise FileNotFoundError(f"Strategy template not found: {strategy_template_path}")
    if not Path(param_config_path).exists():
        raise FileNotFoundError(f"Parameter config not found: {param_config_path}")

    if initial_capital <= 0:
        raise TuningError("initial_capital must be positive")
    if top_n < 1:
        raise TuningError("top_n must be at least 1")


def _check_search_space_size(total_combinations: int, verbose: bool = False) -> None:
    """Check if search space is too large and warn if so."""
    if total_combinations > MAX_SEARCH_SPACE_SIZE:
        warnings.warn(
            f"Search space is very large ({total_combinations:,} combinations). "
            f"This may take a long time. Consider reducing parameter ranges or "
            f"increasing step sizes. Maximum recommended: {MAX_SEARCH_SPACE_SIZE:,}",
            UserWarning,
        )
    elif total_combinations > 10000 and verbose:
        print(f"Note: Large search space ({total_combinations:,} combinations). This may take a while.")


def _run_single_combo(
    combo: Dict[str, float],
    data_path: str,
    base_yaml: Dict[str, Any],
    param_config: ParamConfig,
    initial_capital: float,
) -> Optional[ParameterResult]:
    """
    Worker function for parallel execution.

    Args:
        combo: Parameter combination to test.
        data_path: Path to data file (DataFrame not supported in parallel mode).
        base_yaml: Base strategy YAML dictionary.
        param_config: Parameter configuration.
        initial_capital: Initial capital for backtest.

    Returns:
        ParameterResult if successful, None if failed.
    """
    try:
        # Build path mapping and modify YAML
        path_mapping = build_path_mapping(param_config, combo)
        modified_yaml = clone_and_modify(base_yaml, path_mapping)

        # Run backtest
        final_value, metrics = run_single_backtest(
            data=data_path,
            strategy_yaml=modified_yaml,
            initial_capital=initial_capital,
        )

        return ParameterResult(
            parameters=combo,
            strategy_final_value=metrics["strategy_final_value"],
            total_return_pct=metrics["total_return_pct"],
            excess_vs_lumpsum=metrics["excess_vs_lumpsum"],
            excess_vs_dca=metrics["excess_vs_dca"],
            num_trades=metrics["num_trades"],
            num_buys=metrics["num_buys"],
            num_sells=metrics["num_sells"],
        )
    except Exception:
        return None


def run_parameter_search(
    data_path: Union[str, pd.DataFrame],
    strategy_template_path: str,
    param_config_path: str,
    initial_capital: float = 100000.0,
    top_n: int = 5,
    verbose: bool = False,
    n_jobs: int = 1,
) -> SearchResult:
    """
    Run grid search parameter optimization.

    This function:
    1. Loads parameter config defining search ranges
    2. Loads base strategy YAML template
    3. Generates all parameter combinations
    4. Runs backtest for each combination (optionally in parallel)
    5. Ranks by strategy_final_value
    6. Returns top N results

    Args:
        data_path: Path to indicator data CSV file, or DataFrame directly.
        strategy_template_path: Path to base strategy YAML file.
        param_config_path: Path to parameter configuration YAML.
        initial_capital: Initial capital for backtest.
        top_n: Number of top results to return.
        verbose: Whether to print progress.
        n_jobs: Number of parallel jobs (1 = sequential, >1 = parallel).
                Note: Parallel execution requires picklable data.

    Returns:
        SearchResult with top performing parameter sets.

    Raises:
        TuningError: If configuration is invalid or backtest fails.
        FileNotFoundError: If input files don't exist.
    """
    import tempfile

    # Handle DataFrame input - skip file validation for data_path
    if isinstance(data_path, pd.DataFrame):
        data = data_path  # Use DataFrame directly
        _validate_search_inputs(
            None, strategy_template_path, param_config_path,
            initial_capital, top_n
        )
    else:
        data = data_path  # Will be loaded by run_backtest
        _validate_search_inputs(
            data_path, strategy_template_path, param_config_path,
            initial_capital, top_n
        )

    # Load configurations
    param_config = load_param_config(param_config_path)
    base_yaml = load_yaml(strategy_template_path)

    # Generate all combinations
    combinations = generate_parameter_combinations(param_config)
    total_combinations = len(combinations)

    # Check search space size
    _check_search_space_size(total_combinations, verbose)

    if not param_config.parameters:
        warnings.warn(
            "No parameters to tune; running single backtest with original values",
            UserWarning,
        )

    if verbose:
        print(f"Starting parameter search with {total_combinations} combinations...")
        if n_jobs > 1:
            print(f"Using {n_jobs} parallel workers...")

    # Build search space for reporting
    search_space = {
        spec.name: spec.get_values() for spec in param_config.parameters
    }

    # Run backtests for each combination
    results: List[ParameterResult] = []

    if n_jobs > 1:
        # Parallel execution
        if isinstance(data, pd.DataFrame):
            raise TuningError(
                "Parallel execution (n_jobs > 1) requires a file path, not a DataFrame. "
                "DataFrames cannot be pickled across processes."
            )

        from functools import partial

        worker_func = partial(
            _run_single_combo,
            data_path=data,
            base_yaml=base_yaml,
            param_config=param_config,
            initial_capital=initial_capital,
        )

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_combo = {
                executor.submit(worker_func, combo): combo
                for combo in combinations
            }

            completed = 0
            for future in as_completed(future_to_combo):
                completed += 1
                combo = future_to_combo[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    elif verbose:
                        print(f"    Warning: Backtest failed for {combo}")
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Backtest failed for {combo}: {e}")

                if verbose and completed % 10 == 0:
                    print(f"  [{completed}/{total_combinations}] Progress...")
    else:
        # Sequential execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for i, combo in enumerate(combinations):
                if verbose:
                    print(f"  [{i+1}/{total_combinations}] Testing: {combo}")

                try:
                    # Build path mapping and modify YAML
                    path_mapping = build_path_mapping(param_config, combo)
                    modified_yaml = clone_and_modify(base_yaml, path_mapping)

                    # Run backtest
                    final_value, metrics = run_single_backtest(
                        data=data,
                        strategy_yaml=modified_yaml,
                        initial_capital=initial_capital,
                        temp_dir=temp_path,
                    )

                    # Create result
                    result = ParameterResult(
                        parameters=combo,
                        strategy_final_value=metrics["strategy_final_value"],
                        total_return_pct=metrics["total_return_pct"],
                        excess_vs_lumpsum=metrics["excess_vs_lumpsum"],
                        excess_vs_dca=metrics["excess_vs_dca"],
                        num_trades=metrics["num_trades"],
                        num_buys=metrics["num_buys"],
                        num_sells=metrics["num_sells"],
                    )
                    results.append(result)

                except Exception as e:
                    if verbose:
                        print(f"    Warning: Backtest failed for {combo}: {e}")
                    # Continue with other combinations
                    continue

    if not results:
        raise TuningError("All parameter combinations failed during backtesting")

    # Sort by final value (descending)
    results.sort(key=lambda r: r.strategy_final_value, reverse=True)

    # Get top N
    top_results = results[:top_n]

    # Build search result
    search_result = SearchResult(
        top_results=top_results,
        total_combinations=total_combinations,
        evaluated_combinations=len(results),
        best_parameters=top_results[0].parameters if top_results else {},
        best_final_value=top_results[0].strategy_final_value if top_results else 0.0,
        search_space=search_space,
    )

    if verbose:
        print(f"\nCompleted: {len(results)}/{total_combinations} combinations evaluated")
        print(f"Best final value: ${search_result.best_final_value:,.2f}")

    return search_result


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Run parameter grid search optimization for backtesting strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parameter search
  python -m backtest.tuning.optimizer -d data.csv -s strategy.yaml -p params.yaml

  # With custom capital and top N
  python -m backtest.tuning.optimizer -d data.csv -s strategy.yaml -p params.yaml -c 50000 -n 10

  # Verbose output
  python -m backtest.tuning.optimizer -d data.csv -s strategy.yaml -p params.yaml -v
        """,
    )

    parser.add_argument(
        "-d", "--data",
        required=True,
        help="Path to indicator data CSV file",
    )
    parser.add_argument(
        "-s", "--strategy",
        required=True,
        help="Path to base strategy YAML template",
    )
    parser.add_argument(
        "-p", "--params",
        required=True,
        help="Path to parameter configuration YAML",
    )
    parser.add_argument(
        "-c", "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress during search",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, sequential)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        result = run_parameter_search(
            data_path=args.data,
            strategy_template_path=args.strategy,
            param_config_path=args.params,
            initial_capital=args.capital,
            top_n=args.top_n,
            verbose=args.verbose,
            n_jobs=args.jobs,
        )

        # Print summary
        print(result.summary())

        # Save to JSON if requested
        if args.output:
            import json

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            print(f"\nResults saved to: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except TuningError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
