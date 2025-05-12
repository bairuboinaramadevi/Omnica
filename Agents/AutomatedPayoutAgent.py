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
import psycopg2  # Import the psycopg2 module for PostgreSQL

_ = load_dotenv(find_dotenv())  # Load from .env file if it exists

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")  # Default PostgreSQL port

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
    **Role** :  You are an agent facilitating transactions on Soneium
"""

check_payment_calendar_prompt = """
    **Task**: Given a contract address, retrieve the payment schedule, including due dates and amounts for each installment, from the database. 
    Return the payment schedule in JSON format.  Use function 'getPaymentCalendar'.
"""

execute_installment_payment_prompt = """
    **Task**: Execute an installment payment according to the payment schedule.  
    Given the sender address, recipient address and the installment number, use function 'executeInstallmentPayment' 
    to send the payment and update the database to record the payment. Return the transaction hash.
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

check_payment_calendar_agent = ConversableAgent(
    name="CheckPaymentCalendarAgent",
    system_message=Role + check_payment_calendar_prompt,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

execute_installment_payment_agent = ConversableAgent(
    name="ExecuteInstallmentPaymentAgent",
    system_message=Role + execute_installment_payment_prompt,
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

check_payment_calendar_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

execute_installment_payment_agent.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)


rpc_url = "https://soneium-minato.rpc.scs.startale.com?apikey=8x9PYOaxpO29GqMRcWSQEB0lS6sy"

# Register functions for the Function Executor Agent
function_executor.register_for_execution()

def connect_to_db():
    """
    Connects to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            port=POSTGRES_PORT
        )
        return conn, conn.cursor()
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None, None

@check_payment_calendar_agent.register_for_llm(name="getPaymentCalendar", description="Retrieve the payment schedule for a contract")
def getPaymentCalendar(sender_address: str) -> Optional[List[Dict]]:
    """
    Retrieves the payment schedule for a given contract from the PostgreSQL database.

    Args:
        contract_address (str): The address of the smart contract.

    Returns:
        Optional[List[Dict]]: A list of dictionaries representing the payment schedule.
                        Each dictionary contains 'dueDate' (timestamp) and 'amount' (in ETH).
                        Returns None on error.
    """
    conn, cursor = connect_to_db()
    if conn is None:
        return None

    try:
        cursor.execute(
            "SELECT due_date, amount FROM payment_schedules WHERE sender_address = %s", (sender_address)
        )
        results = cursor.fetchall()
        payment_schedule = [{"dueDate": row[0], "amount": row[1]} for row in results]
        return payment_schedule

    except Exception as e:
        print(f"Error retrieving payment schedule: {e}")
        return None
    finally:
        if conn:
            conn.close()  # Ensure the connection is closed

function_executor.register_for_execution()
@execute_installment_payment_agent.register_for_llm(name="executeInstallmentPayment", description="Execute an installment payment")
def executeInstallmentPayment(sender_address:str, recipient_address: str,amount_ether: float, installment_number: int) -> Optional[str]:
    """
    Executes an installment payment for a given contract on the Soneium network, using PostgreSQL data.
    This function now uses the logic from the executeTransaction function and updates the database.

    Args:
        sender_address (str): The Soneium address of the sender
        recipient_address (str): The Soneium address of the recipient.
        amount_ether (float): The amount of ETH to send.
        installment_number (int): The number of the installment to pay.

    Returns:
        Optional[str]: The transaction hash if successful, None otherwise.
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print("Failed to connect to Soneium Testnet (Minato)")
        return None

    conn, cursor = connect_to_db()
    if conn is None:
        return None

    try:
        # 1. Get the installment details from the database
        cursor.execute(
            "SELECT due_date, amount FROM payment_schedules WHERE sender_address = %s AND installment_number = %s",
            (sender_address, installment_number),
        )
        result = cursor.fetchone()
        if result is None:
            if conn:
                conn.close()
            return f"Installment {installment_number} not found for sender {sender_address}."

        due_date, amount_ether = result
        amount_wei = w3.to_wei(amount_ether, 'ether')

        # 2. Check if the installment is due
        if time.time() < due_date:
            if conn:
                conn.close()
            return f"Installment {installment_number} is not yet due.  Due on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(due_date))}."

        # 3. Build the transaction (using logic from executeTransaction)
        nonce = w3.eth.get_transaction_count(sender_address)
        gas_price = w3.eth.gas_price
        tx = {
            'to': recipient_address,  # Send to the contract
            'value': amount_wei,
            'gas': 100000,  # Adjust as needed
            'gasPrice': gas_price,
            'nonce': nonce,
            'chainId': 1946,  # Soneium Chain ID
            'from': sender_address,
        }

        
        # 5. Update the database to record the payment
        cursor.execute(
            "UPDATE payment_schedules SET paid = TRUE WHERE contract_address = %s AND installment_number = %s",
            (sender_address, installment_number),
        )
        conn.commit()
        if conn:
            conn.close()
        return tx

    except Exception as e:
        print(f"Error executing installment payment: {e}")
        if conn:
            conn.close()
        return None
