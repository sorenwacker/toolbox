import os
import hashlib
import logging
from typing import List, Dict, Optional, Union
from urllib.parse import urlparse
import ftplib

import requests


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ENADownloader:
    """
    A utility for downloading datasets from the European Nucleotide Archive (ENA).

    Manages downloading of genomic datasets with support for various accession types.
    Creates a folder structure to organize the downloaded files:
    
    data/
    ├── raw/
    │   ├── sequencing/
    │   │   ├── sample1_R1.fastq.gz
    │   │   ├── sample1_R2.fastq.gz
    │   │   ├── sample2_R1.fastq.gz
    │   │   ├── sample2_R2.fastq.gz
    │   │   └── ...
    │   └── metadata/
    │       ├── sample_metadata.csv
    │       └── ...
    ├── processed/
    │   ├── aligned/
    │   │   ├── sample1.bam
    │   │   ├── sample2.bam
    │   │   └── ...
    │   ├── filtered/
    │   │   ├── sample1_filtered.bam
    │   │   ├── sample2_filtered.bam
    │   │   └── ...
    │   ├── qc/
    │   │   ├── fastqc/
    │   │   │   ├── sample1_R1_fastqc.html
    │   │   │   ├── sample1_R2_fastqc.html
    │   │   │   ├── sample2_R1_fastqc.html
    │   │   │   ├── sample2_R2_fastqc.html
    │   │   │   └── ...
    │   │   └── multiqc/
    │   │       └── multiqc_report.html
    │   ├── counts/
    │   │   ├── sample1_counts.txt
    │   │   ├── sample2_counts.txt
    │   │   └── ...
    │   └── assemblies/
    │       ├── sample1/
    │       │   ├── sample1_contigs.fasta
    │       │   ├── sample1_scaffolds.fasta
    │       │   └── ...
    │       ├── sample2/
    │       │   ├── sample2_contigs.fasta
    │       │   ├── sample2_scaffolds.fasta
    │       │   └── ...
    │       └── ...
    └── reference/
        ├── genome/
        │   ├── genome.fa
        │   └── genome.fa.fai
        └── annotation/
            ├── genes.gtf
            └── ...
            
    Folder Structure Explanation:
    - 'data/': The root directory for all data files.
        - 'raw/': Contains raw data files.
            - 'sequencing/': Raw sequencing data (e.g., FASTQ files).
            - 'metadata/': Metadata files associated with the samples.
        - 'processed/': Contains processed data files.
            - 'aligned/': Aligned sequencing data (e.g., BAM files).
            - 'filtered/': Filtered aligned data (e.g., filtered BAM files).
            - 'qc/': Quality control reports and metrics.
                - 'fastqc/': FastQC reports for individual samples.
                - 'multiqc/': MultiQC report summarizing all samples.
            - 'counts/': Gene expression count matrices.
            - 'assemblies/': Genome assembly files for each sample.
                - 'sample1/', 'sample2/', ...: Directories for each sample.
                    - 'sample1_contigs.fasta': Assembled contigs for the sample.
                    - 'sample1_scaffolds.fasta': Assembled scaffolds for the sample.
        - 'reference/': Contains reference genome and annotation files.
            - 'genome/': Reference genome files (e.g., FASTA and index files).
            - 'annotation/': Gene annotation files (e.g., GTF files).
    """

    ENA_SEARCH_URL: str = "https://www.ebi.ac.uk/ena/portal/api/search"
    ENA_DOWNLOAD_URL: str = "https://www.ebi.ac.uk/ena/portal/api/download"

    def __init__(self, output_dir: str = 'data'):
        """
        Initialize the ENA downloader.

        Args:
            output_dir: Directory to save downloaded files.
                        Creates the folder structure under this directory.
        """
        self.output_dir = output_dir
        self.create_folder_structure()

    def create_folder_structure(self):
        """
        Create the folder structure for organizing the downloaded files.
        """
        # Raw data directories
        os.makedirs(os.path.join(self.output_dir, 'raw', 'sequencing'), exist_ok=True)
        
        # Processed data directories
        os.makedirs(os.path.join(self.output_dir, 'processed', 'aligned'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'processed', 'filtered'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'processed', 'qc', 'fastqc'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'processed', 'qc', 'multiqc'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'processed', 'counts'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'processed', 'assemblies'), exist_ok=True)
        
        # Reference data directories
        os.makedirs(os.path.join(self.output_dir, 'reference', 'genome'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'reference', 'annotation'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'reference', 'metadata'), exist_ok=True)

    def search_dataset(
        self, 
        accession: str, 
        result_type: str = 'read_run'
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Search for dataset details in ENA.

        Args:
            accession: Study or project accession (e.g., PRJNA123456).
            result_type: Type of result to retrieve.

        Returns:
            List of dataset entries.
        """
        params = {
            'result': result_type,
            'query': 'study_accession={}'.format(accession),
            'fields': 'study_accession,sample_accession,run_accession,fastq_ftp,submitted_ftp,fastq_md5',
            'format': 'JSON'
        }
        
        try:
            logger.info("Searching for dataset: %s", accession)
            response = requests.get(self.ENA_SEARCH_URL, params=params)
            response.raise_for_status()
            
            dataset_info = response.json()
            logger.info("Found %d entries for %s", len(dataset_info), accession)
            
            return dataset_info
        except requests.RequestException as e:
            logger.error("Error searching dataset: %s", e)
            logger.error("Request URL: %s", e.request.url if e.request else 'N/A')
            logger.error("Response: %s", e.response.text if e.response else 'N/A')
            return []

    def download_file_ftp(
        self, 
        url: str, 
        filename: Optional[str] = None,
        subdir: str = 'sequencing'
    ) -> Optional[str]:
        """
        Download a file using FTP protocol.

        Args:
            url: FTP URL of the file to download.
            filename: Optional custom filename.
            subdir: Subdirectory under 'raw' to save the file.

        Returns:
            Path to the downloaded file or None if download fails.
        """
        try:
            # Parse the FTP URL
            parsed_url = urlparse(url)
            host = parsed_url.hostname
            remote_path = parsed_url.path

            # Extract filename if not provided
            if not filename:
                filename = os.path.basename(remote_path)
            
            full_path = os.path.join(self.output_dir, 'raw', subdir, filename)
            
            # Establish FTP connection
            with ftplib.FTP(host) as ftp:
                ftp.login()  # anonymous login
                
                # Change to the directory containing the file
                ftp.cwd(os.path.dirname(remote_path))
                
                # Download the file
                with open(full_path, 'wb') as local_file:
                    ftp.retrbinary('RETR {}'.format(os.path.basename(remote_path)), 
                                   local_file.write)
            
            logger.info("Successfully downloaded: %s", filename)
            return full_path
        
        except Exception as e:
            logger.error("Failed to download %s: %s", url, e)
            return None

    def download_file(
        self, 
        url: str, 
        filename: Optional[str] = None,
        subdir: str = 'sequencing'
    ) -> Optional[str]:
        """
        Download a single file from a given URL.

        Args:
            url: URL of the file to download.
            filename: Optional custom filename.
            subdir: Subdirectory under 'raw' to save the file.

        Returns:
            Path to the downloaded file or None if download fails.
        """
        try:
            # Check if it's an FTP URL
            if url.startswith('ftp://'):
                return self.download_file_ftp(url, filename, subdir)
            
            # For HTTP/HTTPS URLs
            if not url.startswith(('http://', 'https://')):
                url = 'https://{}'.format(url)
            
            # Extract filename if not provided
            if not filename:
                filename = os.path.basename(urlparse(url).path)
            
            full_path = os.path.join(self.output_dir, 'raw', subdir, filename)
            
            logger.info("Attempting to download: %s", url)
            
            # Use requests for HTTP/HTTPS downloads
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(full_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info("Successfully downloaded: %s", filename)
            return full_path
        
        except Exception as e:
            logger.error("Failed to download %s: %s", url, e)
            return None

    def download_dataset(
        self, 
        accession: str
    ) -> List[str]:
        """
        Download entire dataset for a given accession.

        Args:
            accession: Study or project accession.

        Returns:
            List of paths to downloaded files.
        """
        # Search for dataset details
        dataset_info = self.search_dataset(accession)
        
        # Track downloaded files
        downloaded_files: List[str] = []
        
        # Download each file
        for entry in dataset_info:
            if 'fastq_ftp' in entry and entry['fastq_ftp']:
                ftp_urls = entry['fastq_ftp'].split(';')
                
                for url in ftp_urls:
                    downloaded_file = self.download_file(url, subdir='sequencing')
                    if downloaded_file:
                        downloaded_files.append(downloaded_file)
            
            if 'submitted_ftp' in entry and entry['submitted_ftp']:
                sample_url = entry['submitted_ftp'].split(';')[0]  # Assume first file is sample metadata
                sample_filename = f"{entry['sample_accession']}.txt"
                downloaded_file = self.download_file(sample_url, filename=sample_filename, subdir='metadata')
                if downloaded_file:
                    downloaded_files.append(downloaded_file)
        
        return downloaded_files

    def verify_md5(
        self, 
        file_path: str, 
        expected_md5: str
    ) -> bool:
        """
        Verify MD5 checksum of a downloaded file.

        Args:
            file_path: Path to the downloaded file.
            expected_md5: Expected MD5 checksum.

        Returns:
            Boolean indicating if MD5 matches.
        """
        # Calculate MD5 of the file
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read the file in chunks
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        
        calculated_md5 = md5_hash.hexdigest()
        
        return calculated_md5 == expected_md5


def main():
    """
    Example usage of the ENADownloader.
    """
    # Initialize downloader
    downloader = ENADownloader()
    
    # Example: Download dataset from a specific project
    # Try multiple project accessions to diagnose issues
    project_accessions = [
        'PRJEB63589',
    ]
    
    for project_accession in project_accessions:
        try:
            logger.info("Attempting to download dataset: %s", project_accession)
            
            # Download entire dataset
            downloaded_files = downloader.download_dataset(project_accession)
            
            # Print downloaded files
            if downloaded_files:
                logger.info("Downloaded Files:")
                for file in downloaded_files:
                    logger.info("%s", file)
            else:
                logger.warning("No files downloaded for %s", project_accession)
        
        except Exception as e:
            logger.error("Error downloading dataset %s: %s", project_accession, e)


if __name__ == '__main__':
    main()
