import telebot
from dotenv import load_dotenv
import yaml
import os
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from rag import RAG

load_dotenv()

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Inicializar o modelo
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Criar instância do RAG
# Carregar documento
rag = RAG(llm)

# Inicializar o bot
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
PJe_bot = telebot.TeleBot(TELEGRAM_API_KEY)

@PJe_bot.message_handler(commands=['start'])
def send_start_message(message):
    PJe_bot.reply_to(message, config['commands']['start'])

@PJe_bot.message_handler(commands=['help'])
def send_help_message(message):
    PJe_bot.reply_to(message, config['commands']['help'])

@PJe_bot.message_handler(func = lambda message: True)
def send_message(message):
    response = rag.generate_response(message.text, message.from_user.id)
    PJe_bot.reply_to(message, response["answer"])

@PJe_bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Acessar a lista de fotos recebidas (diferentes resoluções)
    photo_sizes = message.json.get('photo', [])
    
    if photo_sizes:
        # Selecionar a maior resolução disponível (última da lista)
        largest_photo = photo_sizes[-1]
        file_id = largest_photo['file_id']
        
        # Obter informações do arquivo
        file_info = PJe_bot.get_file(file_id)
        file_path = file_info.file_path
        
        # Fazer download do arquivo
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_API_KEY}/{file_path}"
        response = requests.get(file_url)

        if response.status_code == 200:
            # Salvar o arquivo localmente
            with open("imagem_recebida.jpg", "wb") as f:
                f.write(response.content)
            PJe_bot.reply_to(message, "Imagem salva com sucesso!")
        else:
            PJe_bot.reply_to(message, "Não foi possível baixar a imagem.")

PJe_bot.infinity_polling()