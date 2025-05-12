import os
import time
import autogen
from autogen import ConversableAgent
import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv
# from main import socketio
from socket_io_setup import socketio
from web3 import Web3
from typing import List, Dict, Optional

_ = load_dotenv(find_dotenv())  # Load from .env file if it exists


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

generation_config = {
            "temperature":0.9,
            "top_p":1,
            "top_k":0,
            "max_output_tokens":4096
        }
safety_settings = [
{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
},
{
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
},
{
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
},
{
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
},
]
genai.configure(api_key=GOOGLE_API_KEY)  # This is the correct way

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)


# LLM Configurations
config_list = [{'model': 'gemini-1.5-flash', 'api_key': GOOGLE_API_KEY, "api_type": "google"}]

llm_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

Role = """
    **Role** :  You are a agent facilitating transactions on Soneium
                """

checkAccountBalancePrompt="""
    **Task** :  Given a Soneium Account address, return the balance in ETH. Use function getBalance and retun only the number as response without any additional text.
"""

executeSoneiumPayoutPrompt = """
    **Task** :  Given a sender account address, receiver account address and amount to transfer in ether, perform transaction. 
    Use function executeTransaction and return only transaction hash as response without any additional text.
"""

checkHistoricalTransactionPrompt = """
    **Task** :  Given a Soneium Account address, use function fetchAccountTransactions and return list of all transation history in form of JSON.
"""

def print_messages(recipient, messages, sender, config):
    # Print the message immediately
    print(
        f"Custom Response Sender: {sender.name} | Recipient: {recipient.name} "
    )

    socketio.emit(f'{sender.name}', f"Executing : {sender.name} ")
    time.sleep(4)
    return False, None  # Required to ensure the agent communication flow continues

user_agent = ConversableAgent(
    name="User_Agent",
    system_message="Return the message as Input of message of agent called",
    llm_config=llm_config,
    max_consecutive_auto_reply=0,
    human_input_mode="NEVER",
)


checkAccountBalanceAgent = ConversableAgent(
    name="CheckAccountBalanceAgent",
    system_message=Role+checkAccountBalancePrompt,
    llm_config=llm_config,
    human_input_mode="NEVER"
)

executeSoneiumPayountAgent = ConversableAgent(
    name="ExecuteSoneiumPayountAgent",
    system_message=Role+executeSoneiumPayoutPrompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

checkHistoricalTransactionAgent = ConversableAgent(
    name="CheckHistoricalTransactionAgent",
    system_message=Role+checkHistoricalTransactionPrompt ,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

function_executor = ConversableAgent(
    name="Function Executor",
    system_message="Execute function",
    llm_config=False,
    human_input_mode="NEVER"
)

user_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
)

checkAccountBalanceAgent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 

executeSoneiumPayountAgent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
) 
checkHistoricalTransactionAgent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages, 
    config={"callback": None},
)


rpc_url = "https://soneium-minato.rpc.scs.startale.com?apikey=8x9PYOaxpO29GqMRcWSQEB0lS6sy"

function_executor.register_for_execution()
@checkAccountBalanceAgent.register_for_llm(name="getBalance", description="Fetch account balance in ETH")
def getBalance(account_address):
    
    # Connect to the Soneium
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    # Check if the connection is successful
    if w3.is_connected():
        print("Successfully connected to Soneium Testnet (Minato)")
    else:
        print("Failed to connect to Soneium Testnet (Minato)")
        exit()

    try:
        # Get the balance of the account in Wei
        balance_wei = w3.eth.get_balance(account_address)

        # Convert the balance from Wei to Ether
        balance_ether = w3.from_wei(balance_wei, 'ether')
        return balance_ether

    except Exception as e:
        print(f"An error occurred: {e}")

function_executor.register_for_execution()
@executeSoneiumPayountAgent.register_for_llm(name="executeTransaction", description="Trannsfer Soneium from sender to receiver account")
def executeTransaction(sender_address: str, receiver_address: str, amount_ether: float) -> Optional[Dict]:
    """
    Connects to the Soneium Testnet, creates an unsigned transaction dictionary
    for transferring ETH, prints it, and returns the dictionary.
    This transaction needs to be signed and sent using an external wallet like MetaMask.

    Args:
        rpc_url: The RPC URL for the Soneium Testnet (Minato).
        sender_address: The Ethereum address of the sender account (managed by MetaMask).
        receiver_address: The Ethereum address of the recipient.
        amount_ether: The amount of ETH to transfer.

    Returns:
        Optional[Dict]: An unsigned transaction dictionary if successful, None otherwise.
    """
    try:
        # Connect to the Soneium Testnet
        w3 = Web3(Web3.HTTPProvider(rpc_url))

        # Check if the connection is successful
        if not w3.is_connected():
            print("Failed to connect to Soneium Testnet (Minato)")
            return None

        # Convert the amount from Ether to Wei
        amount_wei = w3.to_wei(amount_ether, 'ether')

        # Get the current nonce for the sender's account
        nonce = w3.eth.get_transaction_count(sender_address)

        # Get the current gas price
        gas_price = w3.eth.gas_price

        # Construct the unsigned transaction dictionary
        transaction = {
            'to': receiver_address,
            'value': amount_wei,
            'gas': 21000,  # Standard gas limit for a simple ETH transfer
            'gasPrice': gas_price,
            'nonce': nonce,
            'chainId': 1946,  # Chain ID for Soneium Testnet (Minato)
            'from': sender_address,  # Explicitly include the sender address
        }
        return transaction

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

function_executor.register_for_execution()
@checkHistoricalTransactionAgent.register_for_llm(name="fetchAccountTransactions", description="Return list of transactions")
def fetchAccountTransactions(account_address: str, start_block: int = 0, end_block: str = 'latest') -> Optional[List[Dict]]:
    """
    Fetches a list of incoming and outgoing transactions for a given account address
    on the Soneium Testnet (Minato).

    Args:
        w3: A Web3 instance connected to the Soneium Testnet.
        account_address: The Ethereum address to fetch transactions for.
        start_block: The block number to start searching from (inclusive). Defaults to 0.
        end_block: The block number to stop searching at (inclusive).
                   Can be an integer or the string 'latest'. Defaults to 'latest'.

    Returns:
        Optional[List[Dict]]: A list of transaction dictionaries, or None if an error occurs.
    """
    try:
        transactions = []
        # Connect to the Soneium Testnet
        w3 = Web3(Web3.HTTPProvider(rpc_url))

        # Check if the connection is successful
        if not w3.is_connected():
            print("Failed to connect to Soneium Testnet (Minato)")
            return None
        current_block = w3.eth.block_number if end_block == 'latest' else end_block

        for block_number in range(start_block, current_block + 1):
            block = w3.eth.get_block(block_number, full_transactions=True)
            if block and 'transactions' in block:
                for tx in block['transactions']:
                    if tx['from'] == account_address or tx['to'] == account_address:
                        transactions.append(tx)
        return transactions

    except Exception as e:
        print(f"An error occurred while fetching transactions: {e}")
        return None

