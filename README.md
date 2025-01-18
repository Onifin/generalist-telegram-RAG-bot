# Telegram RAG Assistant

Este repositório contém um assistente virtual para Telegram, desenvolvido utilizando a abordagem **RAG (Retrieval-Augmented Generation)**. O bot combina IA generativa com busca eficiente em documentos para responder a perguntas de forma precisa e contextualizada.

## Tecnologias
- **Linguagem:** Python
- **Frameworks:** LangChain, Gemini API, Telebot
- **Armazenamento:** A definir

# Como Usar

Para configurar e executar o bot, siga os passos abaixo:

1. **Clone o repositório**
   ```bash
   git clone https://github.com/Onifin/PJe-Telegram-RAG-Bot.git
   cd PJe-Telegram-RAG-Bot
   ```

2. **Crie um ambiente virtual Python**

   No Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   No Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instale as dependências do programa**
   ```bash
   pip install -r requirements.txt
   ```

4. **Crie o arquivo `.env` e adicione as credenciais**

   No diretório raiz do projeto, crie um arquivo chamado `.env`. 
   
   Adicione a chave da API do Telegram no seguinte formato:
   ```env
   TELEGRAM_API_KEY=seu_token_do_telegram
   ```

   Adicione a chave da API do Gemini no seguinte formato:
   ```env
    GOOGLE_API_KEY=seu_token_do_gemini
   ```

   Texto explicando como conseguir as chaves: ainda em produção.

5. **Configure os dados**

   Crie um arquivo chamado `extracted_text.txt` com o texto que será fornecido para o bot.

   Caso seu texto venha de uma página web, considere usar o WebDataDrill que consegue extrair os textos e estruturar no formato correto para o bot.

6. **Configure o modelo de embeddings**

   No Linux:
   ```bash
   sudo apt update
   sudo apt install curl
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull mxbai-embed-large
   ```

7. **Execute a aplicação**

   No Linux:
   ```bash
   python3 main.py
   ```

   No Windows:
   ```bash
   python main.py
   ```

Agora o bot estará pronto para ser usado!

# Fluxo de Mensagens da Aplicação

A aplicação processa diferentes tipos de mensagens de acordo com o conteúdo recebido. O fluxo funciona da seguinte forma:

1. **Comandos**  
   - Quando um comando é enviado, a aplicação verifica se ele está registrado.  
   - Caso seja um comando cadastrado, uma resposta apropriada é fornecida.

2. **Mensagem de Texto**  
   - Mensagens de texto são processadas utilizando o modelo **Gemini-1.5-pro**, que gera uma resposta adequada baseada no conteúdo enviado.

