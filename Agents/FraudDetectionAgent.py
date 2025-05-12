import os
import time
import autogen
from autogen import ConversableAgent
import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv
from socket_io_setup import socketio  # Import socketio instance
from web3 import Web3
from typing import List, Dict, Optional

_ = load_dotenv(find_dotenv())  # Load from .env file if it exists

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Gemini configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 0,
    "max_output_tokens": 4096,
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
genai.configure(api_key=GOOGLE_API_KEY)
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

# Role definitions for agents
Role = """
    **Role** :  You are an agent identifying and evaluation risk and fraud given list of transactions.
"""

invoice_risk_evaluation_prompt = """
    **Task**: Evaluate the risk associated with a given invoice.  
    Consider factors like the invoice amount, the reputation of the involved parties, and any available historical data.  
    Return a risk score (0-100, where 0 is lowest risk and 100 is highest) and a brief explanation of the risk factors. 
    Use function 'evaluateInvoiceRisk'.
"""

transaction_pattern_analysis_prompt = """
    **Task**: Analyze the transaction patterns of a given Soneium account. 
    Identify any unusual or suspicious activity, such as large or frequent transactions, or transactions with known high-risk accounts.
    Return a summary of the analysis. Use function 'analyzeTransactionPatterns'.
"""

cross_reference_oracles_prompt = """
    **Task**: Cross-reference data from multiple oracles to verify the legitimacy of a transaction or invoice.  
    This might include checking the reputation of the involved parties, verifying the existence of assets, or confirming the accuracy of pricing data. 
    Return a confidence score (0-100) in the legitimacy and a brief explanation of how the oracles were used.  
    Use function 'crossReferenceOracles'.
"""

flag_transaction_anomaly_prompt = """
    **Task**:  Identify anomalous transactions that deviate from expected patterns.  
    This could include sudden changes in transaction volume, unexpected counterparties, or other unusual behavior.  
    Return "True" if an anomaly is detected, "False" otherwise, and a description of the anomaly. 
    Use function 'flagTransactionAnomaly'.
"""

perform_automated_action_prompt = """
    **Task**: Perform an automated action based on the output of other agents.  
    This could include approving or rejecting a transaction, triggering an alert, or initiating a dispute resolution process. 
    Return a confirmation message indicating the action taken. Use function 'performAutomatedAction'.
"""

collect_evidence_document_prompt = """
    **Task**: Collect and document evidence related to a potentially fraudulent or risky transaction.  
    This could include gathering transaction records, invoices, communications, and other relevant information.  
    Return a summary of the collected evidence. Use function 'collectEvidenceDocument'.
"""

evaluate_contract_terms_prompt = """
    **Task**: Evaluate the terms of a smart contract related to a transaction.  
    Identify any potential risks or ambiguities in the contract language. 
    Return a risk assessment of the contract (High, Medium, Low) and a summary of the key terms and potential risks. 
    Use function 'evaluateContractTerms'.
"""

propose_predictive_solution_prompt = """
    **Task**: Propose a predictive solution to mitigate future risks based on the analysis of past transactions and patterns.  
    This could include suggesting changes to transaction processes, implementing new security measures, or developing new risk models.  
    Return a description of the proposed solution. Use function 'proposePredictiveSolution'.
"""

execute_soneium_payout_prompt = """
    **Task**: Execute a payout on the Soneium network. 
    Given a recipient address and an amount, use function 'executeSoneiumPayout' to perform the transaction. Return the transaction hash.
"""


# Custom print_messages function to emit messages via socketio
def print_messages(recipient, messages, sender, config):
    """
    Custom reply function to print messages and emit them via socketio.
    """
    print(
        f"Custom Response Sender: {sender.name} | Recipient: {recipient.name} "
    )
    # Emit the message through the socket
    if socketio:  # Check if socketio is initialized
        socketio.emit(f'{sender.name}', f"Executing : {sender.name} ")
    time.sleep(4)
    return False, None  # Important: Return False to continue agent interaction

# Initialize agents
user_agent = ConversableAgent(
    name="User_Agent",
    system_message="Return the message as Input of message of agent called",
    llm_config=llm_config,
    max_consecutive_auto_reply=0,
    human_input_mode="NEVER",
)

invoice_risk_evaluation_agent = ConversableAgent(
    name="InvoiceRiskEvaluationAgent",
    system_message=Role + invoice_risk_evaluation_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

transaction_pattern_analysis_agent = ConversableAgent(
    name="TransactionPatternAnalysisAgent",
    system_message=Role + transaction_pattern_analysis_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

cross_reference_oracles_agent = ConversableAgent(
    name="CrossReferenceOraclesAgent",
    system_message=Role + cross_reference_oracles_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

flag_transaction_anomaly_agent = ConversableAgent(
    name="FlagTransactionAnomalyAgent",
    system_message=Role + flag_transaction_anomaly_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

perform_automated_action_agent = ConversableAgent(
    name="PerformAutomatedActionAgent",
    system_message=Role + perform_automated_action_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

collect_evidence_document_agent = ConversableAgent(
    name="CollectEvidenceDocumentAgent",
    system_message=Role + collect_evidence_document_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

evaluate_contract_terms_agent = ConversableAgent(
    name="EvaluateContractTermsAgent",
    system_message=Role + evaluate_contract_terms_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

propose_predictive_solution_agent = ConversableAgent(
    name="ProposePredictiveSolutionAgent",
    system_message=Role + propose_predictive_solution_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)
execute_soneium_payout_agent = ConversableAgent(
    name="ExecuteSoneiumPayoutAgent",
    system_message=Role + execute_soneium_payout_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

function_executor = ConversableAgent(
    name="Function Executor",
    system_message="Execute function",
    llm_config=False,
    human_input_mode="NEVER",
)

# Register custom reply functions
user_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

invoice_risk_evaluation_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

transaction_pattern_analysis_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

cross_reference_oracles_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

flag_transaction_anomaly_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

perform_automated_action_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

collect_evidence_document_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

evaluate_contract_terms_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

propose_predictive_solution_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
execute_soneium_payout_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

rpc_url = "https://soneium-minato.rpc.scs.startale.com?apikey=8x9PYOaxpO29GqMRcWSQEB0lS6sy"

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
    
# Register functions for the Function Executor Agent
function_executor.register_for_execution()

@invoice_risk_evaluation_agent.register_for_llm(name="evaluateInvoiceRisk", description="Evaluate the risk associated with an invoice")
def evaluateInvoiceRisk(invoice_amount: float, sender_reputation: int, receiver_reputation: int, historical_data: str) -> Dict:
    """
    Evaluates the risk associated with an invoice based on several factors.

    Args:
        invoice_amount (float): The amount of the invoice.
        sender_reputation (int): The reputation score of the sender (0-100).
        receiver_reputation (int): The reputation score of the receiver (0-100).
        historical_data (str):  A summary of historical transaction data.

    Returns:
        Dict: A dictionary containing the risk score (0-100) and an explanation.
    """
    risk_score = 0
    explanation = ""

    if invoice_amount > 10000:
        risk_score += 30
        explanation += "Large invoice amount. "
    elif invoice_amount > 5000:
        risk_score += 15
        explanation += "Medium invoice amount. "

    if sender_reputation < 50:
        risk_score += 40
        explanation += "Low sender reputation. "
    elif sender_reputation < 70:
        risk_score += 20
        explanation += "Medium sender reputation. "

    if receiver_reputation < 50:
        risk_score += 30
        explanation += "Low receiver reputation. "
    elif receiver_reputation < 70:
        risk_score += 15
        explanation += "Medium receiver reputation. "

    if "fraudulent" in historical_data.lower():
        risk_score = 100
        explanation = "Historical data indicates fraud."
    elif "late payment" in historical_data.lower():
        risk_score += 20
        explanation += "History of late payments."

    risk_score = min(risk_score, 100)  # Cap at 100
    if not explanation:
      explanation = "Low risk"
    return {"risk_score": risk_score, "explanation": explanation}

function_executor.register_for_execution()
@transaction_pattern_analysis_agent.register_for_llm(name="analyzeTransactionPatterns", description="Analyze transaction patterns of an account")
def analyzeTransactionPatterns(account_address: str) -> str:
    """
    Analyzes the transaction patterns of a given Soneium account.

    Args:
        account_address (str): The Soneium account address to analyze.

    Returns:
        str: A summary of the transaction pattern analysis.
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        return "Failed to connect to Soneium Testnet."

    try:
        transactions = fetchAccountTransactions(account_address)  # Reuse the function
        if transactions is None or len(transactions) == 0:
            return "No transactions found for this account."

        num_transactions = len(transactions)
        total_value = 0
        unique_counterparties = set()
        for tx in transactions:
            total_value += tx['value']
            if tx['from'] == account_address:
                unique_counterparties.add(tx['to'])
            else:
                unique_counterparties.add(tx['from'])
        average_value = total_value / num_transactions if num_transactions else 0

        report = f"Analysis of {account_address}:\n"
        report += f"- Number of transactions: {num_transactions}\n"
        report += f"- Total value: {w3.from_wei(total_value, 'ether')} ETH\n"
        report += f"- Average transaction value: {w3.from_wei(average_value, 'ether')} ETH\n"
        report += f"- Unique counterparties: {len(unique_counterparties)}\n"

        if num_transactions > 100:
            report += "High transaction volume detected. "
        if average_value > w3.to_wei(10, 'ether'):  # Example threshold: 10 ETH
            report += "Large average transaction value detected. "
        if len(unique_counterparties) > 50:
            report += "Interacting with many different accounts. "

        return report

    except Exception as e:
        return f"Error analyzing transaction patterns: {e}"

function_executor.register_for_execution()
@cross_reference_oracles_agent.register_for_llm(name="crossReferenceOracles", description="Cross-reference oracles to verify transaction legitimacy")
def crossReferenceOracles(transaction_details: Dict) -> Dict:
    """
    Cross-references data from multiple oracles to verify transaction legitimacy.

    Args:
        transaction_details (Dict): A dictionary containing transaction details.
            Example: {
                "sender": "0x123...",
                "receiver": "0x456...",
                "amount": 1.0,
                "asset": "ETH",
                "invoice_hash": "0xabc...",
            }

    Returns:
        Dict: A dictionary containing a confidence score (0-100) and an explanation.
    """
    confidence_score = 100
    explanation = "All oracles confirm transaction details."

    if "sender" not in transaction_details or "receiver" not in transaction_details:
        confidence_score -= 20
        explanation = "Missing sender or receiver information."
    if transaction_details.get("amount", 0) <= 0:
        confidence_score -= 30
        explanation = "Invalid transaction amount."
    if transaction_details.get("asset") != "ETH":
        confidence_score -= 10
        explanation = "Asset is not ETH (Soneium only supports ETH)."

    sender_reputation = get_reputation(transaction_details["sender"])
    receiver_reputation = get_reputation(transaction_details["receiver"])

    if sender_reputation < 30:
        confidence_score -= 40
        explanation += " Sender reputation is low."
    elif sender_reputation < 60:
        confidence_score -= 20
        explanation += " Sender reputation is medium."

    if receiver_reputation < 30:
        confidence_score -= 30
        explanation += " Receiver reputation is low."
    elif receiver_reputation < 60:
        confidence_score -= 15
        explanation += " Receiver reputation is medium."
    confidence_score = max(0, confidence_score)
    return {"confidence_score": confidence_score, "explanation": explanation}

def get_reputation(address: str) -> int:
  """
  Simulates fetching reputation from an external source.
  """
  if address in ["0xbad1", "0xbad2"]: #Example of bad addresses
    return 10
  elif address in ["0xgood1", "0xgood2"]:
    return 95
  else:
    return 70

function_executor.register_for_execution()
@flag_transaction_anomaly_agent.register_for_llm(name="flagTransactionAnomaly", description="Identify anomalous transactions")
def flagTransactionAnomaly(transaction: Dict, historical_transactions: List[Dict]) -> Dict:
    """
    Flags anomalous transactions based on deviation from expected patterns.

    Args:
        transaction (Dict): The transaction to check.
        historical_transactions (List[Dict]): Historical transactions for the same account.

    Returns:
        Dict: A dictionary containing whether an anomaly was detected (True/False)
              and a description of the anomaly if found.
    """
    anomaly = False
    description = "No anomaly detected."

    if not historical_transactions:
        return {"anomaly": False, "description": "No historical data available."}

    avg_amount = sum(tx['value'] for tx in historical_transactions) / len(historical_transactions)
    std_dev = (sum((tx['value'] - avg_amount) ** 2 for tx in historical_transactions) / len(historical_transactions)) ** 0.5

    amount_ether = Web3().from_wei(transaction['value'], 'ether')
    avg_amount_ether = Web3().from_wei(avg_amount, 'ether')
    std_dev_ether = Web3().from_wei(std_dev, 'ether')
    # Check if the transaction amount is significantly different from the average
    if abs(amount_ether - avg_amount_ether) > 2 * std_dev_ether:  
        anomaly = True
        description = f"Transaction amount ({amount_ether:.2f} ETH) is significantly different from the average ({avg_amount_ether:.2f} ETH)."

    # Check for new or unusual counterparties.
    historical_counterparties = set()
    for tx in historical_transactions:
        if tx['from'] == transaction['from']:
            historical_counterparties.add(tx['to'])
        else:
            historical_counterparties.add(tx['from'])

    if transaction['from'] not in historical_counterparties and transaction['to'] not in historical_counterparties:
        anomaly = True
        description += " New counterparty detected."

    return {"anomaly": anomaly, "description": description}

function_executor.register_for_execution()
@perform_automated_action_agent.register_for_llm(name="performAutomatedAction", description="Perform an automated action based on analysis")
def performAutomatedAction(action_type: str, details: Dict) -> str:
    """
    Performs an automated action based on the output of other agents.

    Args:
        action_type (str): The type of action to perform (e.g., "approve", "reject", "alert").
        details (Dict):  Details relevant to the action.
            e.g., for "reject": {"transaction_id": "0x...", "reason": "High risk"}
            e.g., for "alert": {"account_address": "0x...", "message": "Unusual activity"}

    Returns:
        str: A confirmation message indicating the action taken.
    """
    if action_type == "approve":
        transaction_id = details.get("transaction_id")
        if transaction_id:
            return f"Transaction {transaction_id} approved."
        else:
            return "Transaction approved."

    elif action_type == "reject":
        transaction_id = details.get("transaction_id")
        reason = details.get("reason", "No reason provided")
        if transaction_id:
          return f"Transaction {transaction_id} rejected. Reason: {reason}"
        else:
          return f"Transaction rejected. Reason: {reason}"

    elif action_type == "alert":
        account_address = details.get("account_address")
        message = details.get("message", "No message provided")
        if socketio:
            socketio.emit("alert", {"account_address": account_address, "message": message})
        return f"Alert sent for account {account_address}. Message: {message}"

    elif action_type == "escalate":
        case_id = details.get("case_id")
        return f"Case {case_id} escalated for review."

    else:
        return f"Unknown action type: {action_type}"

function_executor.register_for_execution()
@collect_evidence_document_agent.register_for_llm(name="collectEvidenceDocument", description="Collect evidence related to a transaction")
def collectEvidenceDocument(transaction_id: str) -> str:
    """
    Collects and documents evidence related to a potentially fraudulent or risky transaction.

    Args:
        transaction_id (str): The ID of the transaction to collect evidence for.

    Returns:
        str: A summary of the collected evidence.
    """
    evidence = {
        "transaction_id": transaction_id,
        "timestamp": time.time(),
        "sender_address": "0xSenderAddress",  # Placeholder
        "receiver_address": "0xReceiverAddress",  # Placeholder
        "amount": 1.234,  # Placeholder
        "invoice_hash": "0xInvoiceHash",  # Placeholder
        "sender_reputation": 65,  # Placeholder
        "receiver_reputation": 80,  # Placeholder
        "notes": "Transaction flagged for potential risk due to large amount and new counterparty."  # Placeholder
    }
    # Convert the dictionary to a string representation (e.g., JSON)
    evidence_string = str(evidence)
    return f"Evidence collected for transaction {transaction_id}: {evidence_string}"

function_executor.register_for_execution()
@evaluate_contract_terms_agent.register_for_llm(name="evaluateContractTerms", description="Evaluate the terms of a smart contract")
def evaluateContractTerms(contract_address: str) -> Dict:
    """
    Evaluates the terms of a smart contract.

    Args:
        contract_address (str): The address of the smart contract.

    Returns:
        Dict: A dictionary containing a risk assessment ("High", "Medium", "Low")
              and a summary of key terms and potential risks.
    """
    if contract_address == "0xRiskyContract":
        return {
            "risk_assessment": "High",
            "summary": "Contract contains unusual clauses, including variable fees and unclear dispute resolution process.",
            "risks": "Potential for unexpected fees, lack of clarity on dispute resolution."
        }
    elif contract_address == "0xStandardContract":
        return {
            "risk_assessment": "Low",
            "summary": "Standard contract terms, clear payment schedule, and dispute resolution.",
            "risks": "Minimal risks identified."
        }
    else:
        return {
            "risk_assessment": "Medium",
            "summary": "Contract contains standard terms but with some ambiguities in the payment terms.",
            "risks": "Potential for disputes over payment timing."
        }

function_executor.register_for_execution()
@propose_predictive_solution_agent.register_for_llm(name="proposePredictiveSolution", description="Propose a solution to mitigate future risks")
def proposePredictiveSolution(historical_data: List[Dict]) -> str:
    """
    Proposes a predictive solution to mitigate future risks based on historical data.

    Args:
        historical_data (List[Dict]): A list ofhistorical transaction dictionaries.

    Returns:
        str: A description of the proposed solution.
    """
    if not historical_data:
        return "No historical data available to propose a solution."

    high_risk_transactions = 0
    for tx in historical_data:
      if tx.get("risk_score", 0) > 70:
        high_risk_transactions +=1
    if high_risk_transactions / len(historical_data) > 0.2:
      return "Implement stricter screening for new counterparties and transactions exceeding 5000 ETH.  Consider using a multi-signature approval process for high-value transactions."

    return "Implement real-time risk scoring and automated alerts for transactions exceeding a threshold of 1000 ETH.  Provide training to users on identifying and reporting suspicious activity."

function_executor.register_for_execution()
@execute_soneium_payout_agent.register_for_llm(name="executeSoneiumPayout", description="Execute a payout on the Soneium network")
def executeSoneiumPayout(sender_address:str, recipient_address: str, amount_ether: float) -> Optional[str]:
    """
    Executes a payout on the Soneium network.

    Args:
        sender_address (str): The Soneium address of the sender
        recipient_address (str): The Soneium address of the recipient.
        amount_ether (float): The amount of ETH to send.

    Returns:
        Optional[str]: The transaction hash if successful, None otherwise.
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
            'to': recipient_address,
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
