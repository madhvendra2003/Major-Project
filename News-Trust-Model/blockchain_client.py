from web3 import Web3
import json
from solcx import compile_source, install_solc
install_solc('0.8.0')

class BlockchainClient:
    def __init__(self, provider_url="http://127.0.0.1:7545"):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ganache at {provider_url}")
        self.w3.eth.default_account = self.w3.eth.accounts[0]
        self.contract = None

    def deploy_contract(self, contract_file="contracts/SourceReputation.sol"):
        with open(contract_file, 'r') as f:
            contract_source = f.read()
        compiled = compile_source(
            contract_source,
            output_values=['abi', 'bin'],
            solc_version='0.8.0'
        )
        _, contract_interface = compiled.popitem()
        abi = contract_interface['abi']
        bytecode = contract_interface['bin']
        
        SourceReputation = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        tx_hash = SourceReputation.constructor().transact()
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = tx_receipt.contractAddress
        self.contract = self.w3.eth.contract(address=contract_address, abi=abi)
        print(f"Contract 'SourceReputation' deployed at address: {contract_address}")
        return self.contract

    def set_trust(self, source_id, trust_score):
        if self.contract is None:
            raise Exception("Contract not deployed")
        tx = self.contract.functions.setTrust(source_id, trust_score).transact()
        self.w3.eth.wait_for_transaction_receipt(tx)

    def get_trust(self, source_id):
        if self.contract is None:
            raise Exception("Contract not deployed")
        return self.contract.functions.getTrust(source_id).call()