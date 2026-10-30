# Notas do Veguinha mini

## Escopo desta publicação

Esta versão reúne o código atual, os modelos visuais/físicos, os testes e uma seleção pequena de medições. O diário operacional do aparelho, credenciais, dados de sessão, checkpoints, processos, caches e treinamentos extensos não fazem parte da publicação.

## Caminho percorrido

1. Rede regional com neurônios Izhikevich, balanço de excitação/inibição, Kenyon/APL, testes de olfato, servidor e painel.
2. Corpo 2D, acoplamento neural, decodificação de torque e instrumentação. Tentativas iniciais de aprendizagem não convergiram; não são apresentadas como sucesso.
3. Diagnósticos de osciladores e coordenação, correção do contato e comparação com sondas impostas usadas somente nos ensaios de física.
4. Física MuJoCo com gravidade, quedas e controle de duas juntas por perna.
5. Modelo mk-1 criado pelo autor do projeto, segmentado em um corpo de 18 juntas, convenções explícitas de sinais e teste de suporte passivo.
6. Investigações de half-centers, pools motores, feedback e regularidade; integração modular ao circuito principal e ensaios curtos com controles.
7. Alimento no referencial do mundo, ingestão, percepção binocular, fluxo óptico e circuitos de necessidade/repouso.
8. Presas e manutenção experimentais, paladar, tato, proteção/poda e detector de aproximação visual.
9. Visão periférica com anel intermediário e transição suave, preservando o centro nativo do estéreo.

## Decisões mantidas

- Sem controlador externo de velocidade ou trajetória no app.
- Sondas periódicas e tripés impostos ficam em ferramentas de diagnóstico, não substituem a rede na execução normal.
- Corpo de duas juntas mantido como controle.
- Resultados devem informar seeds, janelas, versão da física, controles e limitações.
- Nenhum treino longo deve ser iniciado automaticamente ao abrir o projeto.
- Marcha aprendida robusta e navegação eficiente continuam pendentes de demonstração.

## Atualização do visual

O visual antigo foi retirado. O controle de duas juntas usa primitivas geométricas e mantém sua física de referência. O mk-1 é um modelo próprio, criado por jotaro41day; não é um asset de terceiros. Caminhada observada e melhoria causada pela plasticidade são avaliações distintas.
