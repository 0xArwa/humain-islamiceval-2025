import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

@dataclass
class Config:
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_model: str
    dev_dataset_xml: str
    test_dataset_xml: str
    dev_dataset_tsv: str
    max_workers: int
    enable_multiprocessing: bool
    environment: str
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY', ''),
            azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
            azure_openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview'),
            azure_openai_model=os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-900ptu'),
            dev_dataset_xml=os.getenv('DEV_DATASET_XML', 'dev_SubtaskA.xml'),
            test_dataset_xml=os.getenv('TEST_DATASET_XML', 'test_SubtaskA.xml'),
            dev_dataset_tsv=os.getenv('DEV_DATASET_TSV', 'dev_SubtaskA.tsv'),
            max_workers=int(os.getenv('MAX_WORKERS', '8')),
            enable_multiprocessing=os.getenv('ENABLE_MULTIPROCESSING', 'false').lower() == 'true',
            environment=os.getenv('ENVIRONMENT', 'dev')
        )
    
    def get_dataset_xml(self) -> str:
        """Get the appropriate dataset XML file based on environment"""
        return self.test_dataset_xml if self.environment == 'test' else self.dev_dataset_xml
    
    def get_openai_client(self) -> AzureOpenAI:
        """Get OpenAI client instance"""
        return AzureOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            api_version=self.azure_openai_api_version
        )