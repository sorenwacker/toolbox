import os
import pandas as pd
from tqdm import tqdm
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class DirectoryMapper:
    """Class to map and summarize file information in a directory and its subdirectories."""

    def __init__(self, directory: str):
        """
        Initialize the DirectoryMapper with the given directory.
        
        Args:
            directory (str): Path to the directory to be analyzed.
        """
        self.directory = directory
        self.file_info = []

    def map_directory(self) -> None:
        """
        Traverse the directory and gather information about each file.
        """
        for root, _, files in tqdm(os.walk(self.directory), desc="Traversing Directories"):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    file_extension = self.get_full_extension(file).lower()
                    self.file_info.append((file_extension, file_size))
                except FileNotFoundError as e:
                    logger.warning(f"File not found: {file_path}, {e}")

    @staticmethod
    def get_full_extension(file_name: str) -> str:
        """
        Get the full extension of a file, considering cases like '.tar.gz'.
        
        Args:
            file_name (str): Name of the file.
        
        Returns:
            str: The full extension without leading dot.
        """
        parts = file_name.split('.')
        if len(parts) > 2:
            return '.'.join(parts[-2:])
        elif len(parts) == 2:
            return parts[-1]
        return ''

    @staticmethod
    def get_best_unit(size_in_bytes: float) -> tuple:
        """
        Determine the best unit for displaying sizes.
        
        Args:
            size_in_bytes (float): Size in bytes.
        
        Returns:
            tuple: Unit name and conversion factor.
        """
        if size_in_bytes < 1024:
            return 'B', 1
        elif size_in_bytes < 1024 ** 2:
            return 'KiB', 1024
        elif size_in_bytes < 1024 ** 3:
            return 'MiB', 1024 ** 2
        elif size_in_bytes < 1024 ** 4:
            return 'GiB', 1024 ** 3
        else:
            return 'TiB', 1024 ** 4

    def summarize_file_info(self) -> pd.DataFrame:
        """
        Summarize the gathered file information by file type.

        Returns:
            pd.DataFrame: Summary DataFrame containing file extensions, counts, and total volumes.
        """
        df = pd.DataFrame(self.file_info, columns=['Extension', 'Size'])
        summary = df.groupby('Extension').agg(
            Count=('Extension', 'count'),
            Volume=('Size', 'sum'),
            Max_Size=('Size', 'max'),
            Mean_Size=('Size', 'mean'),
            Median_Size=('Size', 'median')
        ).reset_index()

        total_volume_bytes = summary['Volume'].sum()
        best_unit, factor = self.get_best_unit(total_volume_bytes)
        
        summary['Volume'] = summary['Volume'] / factor
        summary['Max_Size'] = summary['Max_Size'] / factor
        summary['Mean_Size'] = summary['Mean_Size'] / factor
        summary['Median_Size'] = summary['Median_Size'] / factor

        total_files = summary['Count'].sum()
        total_volume = summary['Volume'].sum()
        
        total_row = pd.DataFrame([{
            'Extension': 'Total',
            'Count': total_files,
            'Volume': total_volume,
            'Max_Size': summary['Max_Size'].max(),
            'Mean_Size': summary['Mean_Size'].mean(),
            'Median_Size': summary['Median_Size'].median()
        }])
        
        summary = pd.concat([summary, total_row], ignore_index=True)
        
        summary.rename(columns={
            'Volume': f'Volume [{best_unit}]',
            'Max_Size': f'Max_Size [{best_unit}]',
            'Mean_Size': f'Mean_Size [{best_unit}]',
            'Median_Size': f'Median_Size [{best_unit}]'
        }, inplace=True)
        
        return summary

    def get_summary(self) -> pd.DataFrame:
        """
        Get the summary of the directory's file information.

        Returns:
            pd.DataFrame: Summary DataFrame containing file extensions, counts, and total volumes.
        """
        self.map_directory()
        return self.summarize_file_info()

def save_summary(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the summary DataFrame to the specified file format.
    
    Args:
        df (pd.DataFrame): DataFrame containing the summary.
        output_file (str): Path to the output file.
    """
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.parquet'):
        df.to_parquet(output_file, index=False)
    elif output_file.endswith('.xlsx'):
        df.to_excel(output_file, index=False)
    else:
        raise ValueError("Unsupported file format. Please use .csv, .parquet, or .xlsx.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map and summarize file information in a directory.")
    parser.add_argument('directory', type=str, help="Path to the directory to be analyzed.")
    parser.add_argument('-o', '--output', type=str, help="Path to the output file (.csv, .parquet, or .xlsx).")
    
    args = parser.parse_args()
    
    directory_mapper = DirectoryMapper(args.directory)
    summary_df = directory_mapper.get_summary().round(3)
    
    if args.output:
        save_summary(summary_df, args.output)
        print(f"Summary saved to {args.output}")
    else:
        print(summary_df)
