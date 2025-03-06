import yaml

# Carregar o arquivo YAML
with open('config.yaml', 'r', encoding='utf-8') as f:
    dados = yaml.safe_load(f)

# Acessar o prompt_template
contexto = "Documentos sobre inteligência artificial..."
pergunta = "Quais são as aplicações práticas de IA hoje?"
prompt_template = dados['prompt_template']
prompt_final = prompt_template.format(
    context=contexto,
    question=pergunta
)
print(prompt_final)