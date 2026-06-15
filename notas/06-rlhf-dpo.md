# 6. De State of GPT a DPO

Na palestra "State of GPT", Karpathy organizou o pipeline em estágios:
pré-treino, SFT, modelagem de recompensa e RLHF. A pergunta desta nota é como um
modelo base cru, que só completa texto, vira um assistente que segue instruções e
recusa pedidos problemáticos.

RLHF (aprendizado por reforço com feedback humano) foi a resposta clássica.
Coletam-se comparações humanas (resposta A é melhor que B), treina-se um modelo
de recompensa que aprende essa preferência, e então otimiza-se o LLM para
maximizar essa recompensa, geralmente com PPO. Funciona, mas é uma engenharia
pesada: treinar e manter um modelo de recompensa separado e rodar RL é instável e
caro.

DPO (Direct Preference Optimization) é a simplificação que ganhou tração. Ele
mostra que dá para pular o modelo de recompensa explícito e o loop de RL, e
otimizar diretamente uma perda de classificação sobre os pares de preferência. O
resultado é um treino mais estável e mais simples, com qualidade comparável. A
lição para a trilha é bonita: o mesmo objetivo de preferência pode ser alcançado
por um caminho de otimização muito mais barato, que é a história recorrente de
todo este percurso, de RLHF a DPO assim como de full fine-tuning a LoRA.
