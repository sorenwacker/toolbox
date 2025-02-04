from __future__ import annotations
import os
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum


class SizeUnit(Enum):
    """Enumeration of supported file size units."""
    BYTES = ("B", 1)
    KIBIBYTES = ("KiB", 1024)
    MEBIBYTES = ("MiB", 1024 ** 2)
    GIBIBYTES = ("GiB", 1024 ** 3)
    TEBIBYTES = ("TiB", 1024 ** 4)

    def __init__(self, symbol: str, factor: int):
        self.symbol = symbol
        self.factor = factor

    @classmethod
    def from_size(cls, size_in_bytes: float) -> 'SizeUnit':
        """Determine the most appropriate unit based on file size."""
        for unit in reversed(cls):
            if size_in_bytes >= unit.factor:
                return unit
        return cls.BYTES


@dataclass
class FileInfo:
    """Data class to store information about a file."""
    extension: str
    size: int
    is_directory: bool


class FileExtensionAnalyzer:
    """Analyzes file information in a directory and its subdirectories."""

    def __init__(self, directory: Union[str, Path], unit: Optional[Union[SizeUnit, str]] = None):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"Path is not a directory: {self.directory}")
        
        if isinstance(unit, str):
            try:
                self.unit = next(u for u in SizeUnit if u.symbol == unit)
            except StopIteration:
                raise ValueError(f"Invalid unit: {unit}")
        else:
            self.unit = unit
        
        self.file_info: List[FileInfo] = []
        self.summary: Optional[pd.DataFrame] = None
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def analyze(self) -> pd.DataFrame:
        """
        Analyze the directory and generate a summary.
        
        Returns:
            DataFrame containing the analysis summary
        """
        self._collect_file_info()
        return self._generate_summary()

    def _collect_file_info(self) -> None:
        """Traverse the directory and collect information about each file."""
        self.file_info.clear()
        try:
            for root, _, files in tqdm(os.walk(self.directory), 
                                     desc="Analyzing files",
                                     unit="files"):
                for file in files:
                    try:
                        file_path = Path(root) / file
                        self.file_info.append(FileInfo(
                            extension=file_path.suffix.lower() or '(no extension)',
                            size=file_path.stat().st_size,
                            is_directory=file_path.is_dir()
                        ))
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Error accessing {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error walking directory {self.directory}: {e}")
            raise

    def _generate_summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of the collected file information."""
        if not self.file_info:
            self.logger.warning("No files found to analyze")
            return pd.DataFrame()

        df = pd.DataFrame(self.file_info)
        
        summary = df.groupby('extension').agg(
            count=('extension', 'count'),
            total_size=('size', 'sum'),
            max_size=('size', 'max'),
            min_size=('size', 'min'),
            mean_size=('size', 'mean'),
            median_size=('size', 'median')
        ).reset_index()

        if self.unit is None:
            total_volume = summary['total_size'].sum()
            self.unit = SizeUnit.from_size(total_volume)

        size_columns = ['total_size', 'max_size', 'min_size', 'mean_size', 'median_size']
        for col in size_columns:
            summary[col] = summary[col] / self.unit.factor

        total_row = pd.DataFrame([{
            'extension': 'Total',
            'count': summary['count'].sum(),
            'total_size': summary['total_size'].sum(),
            'max_size': summary['max_size'].max(),
            'min_size': summary['min_size'].min(),
            'mean_size': summary['mean_size'].mean(),
            'median_size': summary['median_size'].median()
        }])
        
        summary = pd.concat([summary, total_row], ignore_index=True)
        
        summary.rename(columns={
            col: f"{col} [{self.unit.symbol}]" for col in size_columns
        }, inplace=True)

        self.summary = summary
        return summary

    def save_summary(self, output_path: Union[str, Path], 
                    format: Optional[str] = None) -> None:
        """Save the summary to a file."""
        if self.summary is None:
            raise ValueError("No summary available. Run analyze() first.")

        output_path = Path(output_path)
        format = format or output_path.suffix.lower()[1:]
        
        save_methods = {
            'csv': lambda df, path: df.to_csv(path, index=False),
            'parquet': lambda df, path: df.to_parquet(path, index=False),
            'xlsx': lambda df, path: df.to_excel(path, index=False)
        }
        
        if format not in save_methods:
            raise ValueError(f"Unsupported format: {format}. Use csv, parquet, or xlsx")
        
        save_methods[format](self.summary, output_path)
        self.logger.info(f"Summary saved to {output_path}")

    def print_markdown(self) -> None:
        """Print the summary as a markdown table."""
        if self.summary is None:
            raise ValueError("No summary available. Run analyze() first.")
        print(self.summary.to_markdown(index=False, floatfmt=".2f"))


def main():
    """Command-line interface for the file extension analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze file extensions in a directory")
    parser.add_argument("directory", type=str, help="Directory to analyze")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--unit", type=str, choices=[u.symbol for u in SizeUnit],
                       help="Force specific size unit")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    unit = next((u for u in SizeUnit if u.symbol == args.unit), None)
    
    try:
        analyzer = FileExtensionAnalyzer(args.directory, unit)
        summary = analyzer.analyze()
        
        if args.output:
            analyzer.save_summary(args.output)
        else:
            analyzer.print_markdown()
            
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()