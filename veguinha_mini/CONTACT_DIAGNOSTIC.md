# Diagnóstico de contato e ponte motora

Os ensaios deste protocolo usam `contact_diagnostic.py` e salvam linhas completas
atomicamente em `results/contact_*_v1.json`. Cada invocação tem orçamento de
230 s e deve ser protegida com `timeout 240s`. Repetir o comando retoma as linhas
ausentes. Nenhuma plasticidade é usada.

## Três modelos separados, sem sobrescrever o histórico

| Modelo | Contato e carga | Arquivo de convergência |
|---|---|---|
| Legado | Máscara `knee < 0` antes do passo; limite de atrito fixo 65 | `contact_convergence_v1.json` |
| Altura fixa | Altura do pé, carga unilateral, limite proporcional à carga; altura corporal fixa | `contact_contact_convergence_v1.json` |
| Sustentação vertical | Mesmo contato unilateral; altura corporal e velocidade vertical integradas sob gravidade e cargas dos pés | `contact_loaded_convergence_v1.json` |

O segundo modelo é uma ablação mecânica: não exige sustentação do peso. O
terceiro é a física usada na nova comparação neural. A projeção horizontal das
pernas permanece igual; a elevação é uma aproximação 2,5D, sem roll/pitch.
Isto não representa uma validação completa de equilíbrio tridimensional.

## Equações e parâmetros

| Quantidade | Definição |
|---|---|
| Altura do pé | `h_corpo - 24 + 24*sin(q_joelho)` |
| Velocidade vertical do pé | `v_vertical + 24*cos(q_joelho)*omega_joelho` |
| Carga normal | `max(0, -50*altura - 1.5*velocidade_vertical)` somente se altura ≤ 0; zero acima do solo |
| Atrito | Viscoso anisotrópico original (1,4 longitudinal; 4 lateral), com norma limitada a `2*carga_normal` |
| Reação na junta | Carga normal × derivada da altura em relação ao ângulo do joelho |
| Corpo vertical | Massa 3; gravidade 9,81; amortecimento 1,8; soma das cargas dos pés |
| Integração | Corpo 2 ms; rede 1 ms em blocos de 20 ms; torque constante em todos os substeps do bloco |

`Body.step(..., contacts=None, ...)` calcula contato físico. Com
`body.vertical_support=True`, integra também a sustentação vertical. Máscaras
explícitas são mantidas para reproduzir os ensaios históricos e para o modo
manual de teste. O aplicativo/rede completa não é promovido automaticamente
para este protocolo experimental antes de vencer os controles.

`fast_body.pyx` implementa as mesmas equações para baratear os ensaios. Testes
comparam seus estados com `Body.step`, incluindo altura e velocidade vertical,
nos quatro passos solicitados. Não usa trajetórias ou decisões comportamentais.

## Sonda e rede

| Ensaio | Definição |
|---|---|
| Sonda imposta | Período 0,38 s; fases interpernas A/B 0/180°; varredura quadril–joelho 0–330° a cada 30° (360° repete 0°); torque RMS calibrado antes da correção física |
| Acoplamento local experimental | Interneurônios I do quadril inibem E homólogos do joelho; I do joelho inibem E antagonistas do quadril; 3 alvos por fonte; pesos negativos `ganho/sqrt(grau)`; atraso 2 ms e corrente de 40 ms |
| Reflexo de carga | Carga medida dividida por `massa*g/6`, limitada a [0,2]; entrada nos I extensores do quadril e I flexores do joelho; portanto recruta extensão de quadril e flexão de joelho por inibição recíproca |
| Ausência de fase prescrita na rede | Não há relógio de marcha, seno, fase alvo ou torque externo no circuito. Apenas spikes, conexões e aferências mecânicas |
| Controle sem acoplamento | Remove apenas o acoplamento **entre pernas**; mantém o mesmo circuito local e reflexo de carga selecionados |
| Embaralhado | Mantém acoplamentos; permuta os dois sinais motores de cada perna juntos, com uma permutação sem pontos fixos por seed; sensores mantêm a anatomia |

As seeds 42–44 servem para a sonda/calibração curta. A comparação confirmatória
usa seeds novas 52–71, pareadas entre condições, sem reajustar parâmetros depois
de observar seus resultados.

O circuito local com ganho 10 e reflexo de carga 1 reduziu o torque RMS para
~0,18, ante ~0,33 sem o novo acoplamento. Dois controles curtos de ganho motor
foram feitos **somente** nas seeds de calibração: ganho 0,27 recuperou o RMS
para ~0,29, mas a velocidade frontal média foi −0,280; ganho 0,32 elevou o
RMS a ~0,36 e a velocidade a −0,134. Nenhum gerou avanço consistente ou
alternância de apoio. Por isso, esses ganhos não substituíram o protocolo
confirmatório de 20 seeds, e não iniciamos Brian2 completo ou aprendizado.

## Métrica principal

Velocidade frontal líquida é a média temporal **assinada** de
`vx*cos(heading) + vy*sin(heading)`. Também reportamos distância euclidiana
inicial–final e integral frontal assinada. A razão deslocamento/caminho dos
relatórios antigos não é o desfecho principal deste protocolo.

Fase e apoio são avaliados na janela final, com exclusão das bordas da análise
de fase; média circular e concentração são salvas por perna. A comparação usa
diferenças pareadas e IC95% t de Student. A sonda não treina nem define uma
marcha alvo. A rede Brian2 completa e o aprendizado continuam condicionados à
vantagem confirmatória sobre os controles.
