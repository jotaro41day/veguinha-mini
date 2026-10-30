# Arquitetura

## Circuito e ciclo físico

`config.py` configura escalas, ganhos, clocks e caminhos. `neurons.py`, `regions.py` e `network.py` constroem neurônios Izhikevich e projeções regionais. A arquitetura inclui lobo antenal, células de Kenyon/APL/saídas do corpo pedunculado, complexo central, descendentes, módulos ventrais, valor, dopamina e octopamina.

No mk-1 modular, cada perna tem pools flexor/extensor para coxa, fêmur e tíbia. `leg_circuit.py`, `cord.py` e `network.py` contêm os circuitos motores; `motor.py` transforma a diferença das taxas filtradas em torque limitado. Conexões inatas e inibição recíproca dão capacidade de oscilação; não equivalem a uma marcha aprendida.

`sim.py` executa a rede e atualiza sensores/corpo em blocos de 20 ms. `mujoco_body.py` integra o corpo em subpassos de 2 ms; a rede central usa 0,5 ms na configuração modular. O torque é mantido entre atualizações motoras. O corpo antigo em `body.py` é histórico; o app usa os contatos e a gravidade do MuJoCo.

## Sensores

`sensory.py` encaminha ângulos, carga/contato, olfato, fluxo corporal e energia às populações correspondentes. A cópia de eferência chega à região comparadora.

`perception.py` produz imagens a partir da cena física e mede contato oral/dor. `binocular.py` combina os sinais visuais em detectores pulsados de disparidade. `optical_flow.py` estima deslocamento local de textura e confiança somente com pares de imagens. `somatic.py` contém transdução gustativa/tátil e o detector de aproximação visual. Processamento NumPy participa da transdução e leitura; nem todo processamento sensorial é uma população de neurônios.

O fluxo óptico e a odometria física são modalidades separadas. Regiões sem correspondência visual confiável ficam desconhecidas; não são tratadas como profundidade medida.

## Aprendizagem, necessidade e ambiente

`learning.py` mantém traços pré/pós-sinápticos e de elegibilidade; a modulação dopaminérgica vem do disparo dos DANs no caminho usado pelo app. Há limites e normalização de pesos e proteção/poda experimental. Um ramo histórico preserva uma formulação anterior de recompensa para reprodução; não representa o caminho sensorial atual do app.

`intent.py` contém a competição neural que modula a liberação dos motores. Energia, estímulos, contato oral e ameaça alteram correntes neurais; movimento não é selecionado por um comando de velocidade do navegador. `ecology.py` e `maintenance.py` acrescentam presas, poeira, fadiga e circuitos experimentais. Esses mecanismos ainda não demonstram caça, limpeza ou busca de abrigo confiáveis.

`world.py` mantém alimento e odor no referencial do mundo. `server.py` publica snapshots e recebe comandos; `mjviser_bridge.py` desenha o estado do MuJoCo em um viewer separado.
