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

4. **Crie o arquivo `.env` e adicione a chave do bot Telegram**

   No diretório raiz do projeto, crie um arquivo chamado `.env` e adicione a chave da API do Telegram no seguinte formato:
   ```env
   TELEGRAM_API_KEY=seu_token_do_telegram
   ```

5. **Execute a aplicação**
   ```bash
   python main.py
   ```

Agora o bot estará pronto para ser usado!

# Fluxo de Mensagens da Aplicação

A aplicação processa diferentes tipos de mensagens de acordo com o conteúdo recebido. O fluxo funciona da seguinte forma:

1. **Comando**  
   - Quando um comando é enviado, a aplicação verifica se ele está registrado.  
   - Caso seja um comando cadastrado, uma resposta apropriada é fornecida.

2. **Mensagem de Texto**  
   - Mensagens de texto são processadas utilizando o modelo **Gemini-1.5-pro**, que gera uma resposta adequada baseada no conteúdo enviado.

3. **Imagem**  
   - Imagens recebidas são processadas para gerar uma descrição textual, também criada pelo modelo **Gemini-1.5-pro**.

4. **Imagem com Texto**  
   - A aplicação converte o texto embutido na imagem para formato textual.  
   - Em seguida, utiliza o texto extraído e a descrição da imagem para formular uma resposta completa.

5. **Vídeo ou Vídeo com Texto**  
   - Para mensagens contendo vídeos ou vídeos com texto, a aplicação informa que este tipo de dado não é suportado no momento.
