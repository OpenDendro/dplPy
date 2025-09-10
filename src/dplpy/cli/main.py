"""
Command line interface for dplpy package.

This module provides the main CLI functionality for dplpy, including
help system, argument parsing, and command dispatch.
"""

import argparse
import sys
import webbrowser
from typing import Optional, List


def help_system() -> None:
    """
    Display comprehensive help information for dplPy package.
    
    This function provides an interactive help menu with information about
    dplPy's main functionality, usage examples, and available functions
    for dendrochronological analysis.
    
    Notes
    -----
    The help system covers:
    - Data import/export (readers, writers)
    - Statistical analysis (stats, summary, report)
    - Detrending and standardization
    - Chronology development
    - Cross-dating and quality control
    - Autoregressive modeling
    
    Examples are provided for both command-line and interactive Python usage.
    
    See Also
    --------
    readme : Open online documentation
    """
    try:
        print("*Welcome to the dplPy Help Menu*")
        print("dplPy: A Python library for dendrochronological analysis")
        print("")
        print("....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⋮....:....⁞")
        print("")
        print("DOCUMENTATION")
        print("For complete documentation and tutorials:")
        print(">>> import dplpy as dpl")
        print(">>> dpl.readme()  # Opens web documentation")
        print(">>> help(dpl)     # Python help system")
        print("")
        print("CORE FUNCTIONS")
        print("")
        print("Data Import/Export:")
        print("  dpl.readers()   - Load tree ring data (RWL, CSV formats)")
        print("  dpl.write()     - Export data to various formats")
        print("")
        print("Statistical Analysis:")
        print("  dpl.stats()     - Comprehensive dendrochronological statistics")
        print("  dpl.summary()   - Basic descriptive statistics")
        print("  dpl.report()    - Data quality and missing ring analysis")
        print("")
        print("Standardization and Chronology:")
        print("  dpl.detrend()   - Remove growth trends from raw measurements")
        print("  dpl.chron()     - Build chronologies from detrended data")
        print("")
        print("Quality Control:")
        print("  dpl.xdate()     - Cross-dating analysis and verification")
        print("")
        print("Time Series Modeling:")
        print("  dpl.autoreg()   - Autoregressive modeling and prewhitening")
        print("")
        print("QUICK START EXAMPLE")
        print("")
        print(">>> import dplpy as dpl")
        print(">>> # Load tree ring data")
        print(">>> data = dpl.readers('path/to/data.rwl')")
        print(">>> # Generate statistics")
        print(">>> statistics = dpl.stats(data)")
        print(">>> # Detrend and build chronology")
        print(">>> rwi = dpl.detrend(data)")
        print(">>> chronology = dpl.chron(rwi)")
        print(">>> # Cross-date for quality control")
        print(">>> xdate_results = dpl.xdate(data)")
        print("")
        print("For detailed function help, use help(dpl.function_name)")
        print("Visit https://opendendro.org/dplpy for tutorials and examples")
        print("")
        print("END HELP MANUAL")
    except Exception as e:
        print(f"Error displaying help: {e}")


def readme() -> None:
    """
    Open the online dplPy documentation in web browser.
    
    This function launches the comprehensive dplPy documentation website
    which contains tutorials, API reference, examples, and theoretical
    background for dendrochronological analysis methods.
    
    Returns
    -------
    None
        Opens documentation website in default web browser.
    
    Notes
    -----
    The online documentation includes:
    - Getting started tutorials
    - Complete function API reference  
    - Worked examples with real data
    - Theoretical background on dendrochronology
    - Installation and setup instructions
    - Contributing guidelines
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> dpl.readme()  # Opens documentation in browser
    
    See Also
    --------
    help : Display help in terminal
    """
    try:
        print("Opening dplPy documentation...")
        webbrowser.open("https://opendendro.github.io/dplpy/")
        print("Documentation opened in web browser")
        print("If browser did not open, visit: https://opendendro.github.io/dplpy/")
    except Exception as e:
        print(f"Could not open browser: {e}")
        print("Please visit: https://opendendro.github.io/dplpy/")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="dplpy",
        description="dplPy: Dendrochronology Program Library for Python",
        epilog="For more information, visit: https://opendendro.github.io/dplpy/"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="dplpy 0.1.6"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Help command
    help_parser = subparsers.add_parser("help", help="Show help information")
    help_parser.set_defaults(func=lambda args: help_system())
    
    # Readme command  
    readme_parser = subparsers.add_parser("readme", help="Open documentation")
    readme_parser.set_defaults(func=lambda args: readme())
    
    # Analysis commands could be added here in the future
    # analyze_parser = subparsers.add_parser("analyze", help="Run analysis")
    # etc.
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the dplpy command line interface.
    
    Parameters
    ----------
    args : list of str, optional
        Command line arguments. If None, uses sys.argv[1:].
        
    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    
    if args is None:
        args = sys.argv[1:]
    
    # If no arguments provided, show help
    if not args:
        help_system()
        return 0
    
    try:
        parsed_args = parser.parse_args(args)
        
        # Execute the command
        if hasattr(parsed_args, 'func'):
            parsed_args.func(parsed_args)
        else:
            parser.print_help()
            
        return 0
        
    except SystemExit as e:
        return e.code if e.code is not None else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())