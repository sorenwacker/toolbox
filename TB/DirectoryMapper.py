import os
import pandas as pd
from tqdm import tqdm
import argparse

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
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                file_extension = self.get_full_extension(file).lower()
                self.file_info.append((file_extension, file_size))

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
        summary['Volume [GiB]'] = summary['Volume'] / (1024 ** 3)  # Convert bytes to GiB
        summary['Volume [TiB]'] = summary['Volume'] / (1024 ** 4)  # Convert bytes to TiB
        summary['Max_Size [GiB]'] = summary['Max_Size'] / (1024 ** 3)  # Convert bytes to GiB
        summary['Mean_Size [GiB]'] = summary['Mean_Size'] / (1024 ** 3)  # Convert bytes to GiB
        summary['Median_Size [GiB]'] = summary['Median_Size'] / (1024 ** 3)  # Convert bytes to GiB
        
        total_files = summary['Count'].sum()
        total_volume_gib = summary['Volume [GiB]'].sum()
        total_volume_tib = summary['Volume [TiB]'].sum()
        
        total_row = pd.DataFrame([{
            'Extension': 'Total',
            'Count': total_files,
            'Volume [GiB]': total_volume_gib,
            'Volume [TiB]': total_volume_tib,
            'Max_Size [GiB]': summary['Max_Size [GiB]'].max(),
            'Mean_Size [GiB]': summary['Mean_Size [GiB]'].mean(),
            'Median_Size [GiB]': summary['Median_Size [GiB]'].median()
        }])
        
        summary = pd.concat([summary, total_row], ignore_index=True)
        summary = summary.drop(columns=['Volume', 'Max_Size', 'Mean_Size', 'Median_Size'])  # Remove raw size columns in bytes
        
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
